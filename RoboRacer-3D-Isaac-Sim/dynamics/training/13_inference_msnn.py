"""
D1: MS-NN inference API for Isaac Sim integration.

Usage (Isaac Sim):
    from inference_msnn import MSNNInference
    infer = MSNNInference(model_dir='trained_models/msnn_exp8_combined_lowlr')
    infer.reset()
    yr_pred, ax_pred = infer.step(delta=0.1, vel=2.0, acc=0.5, accy=-0.3, crt=50.0)

Model inputs at each timestep (20ms steps assumed):
    delta : steering_cmd [rad]
    vel   : odom_speed [m/s]
    acc   : IMU_linear_acceleration_x [m/s²]
    accy  : IMU_linear_acceleration_y [m/s²]
    crt   : current_motor [A or raw]

Outputs:
    yr_pred : predicted yaw_rate for next step [rad/s]
    ax_pred : predicted longitudinal acceleration for next step [m/s²]
"""
import itertools
import os, sys
import numpy as np

# Allows importing when cwd is different from script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class MSNNInference:
    # Window sizes matching training configuration
    NUM_SAMPLES_PAST = 20  # lateral model lookback
    CURRENT_WINDOW   = 15  # longitudinal model lookback

    # Class-level (not per-instance) so it stays unique across multiple
    # reset() calls and multiple instances within one process -- see the
    # bugfix note in reset() below.
    _dataset_name_counter = itertools.count()

    def __init__(self, model_dir: str, data_stats: dict | None = None):
        """
        Args:
            model_dir   : path to trained_models/msnn_exp* directory (contains net.json)
            data_stats  : UNUSED, kept only for backward-compatible call signatures.
                          See _build_and_load()'s docstring for why.
        """
        sys.path.insert(0, _SCRIPT_DIR)
        self._stats = data_stats
        self._model_dir = os.path.abspath(model_dir) if not os.path.isabs(model_dir) else model_dir

        self._dyn = self._build_and_load()
        self.reset()

    def _build_and_load(self):
        from nnodely import nnodely

        # BUGFIX 2026-08-13 (final version): two earlier attempts here hand-
        # reconstructed the training-time nnodely graph in Python (first the
        # wrong "noax" architecture, then a corrected "withax" one matching
        # exp9_withax_combined.py line-for-line) before calling loadModel().
        # Both produced BYTE-IDENTICAL predictions to the ORIGINAL buggy
        # code, which was the tell: none of that hand-built graph was ever
        # actually used. Read Loader.loadModel()'s source directly --
        #   self._model_def = ModelDef(model_def)   # wholesale replace
        #   self._model = None
        #   self._neuralized = False
        # it discards whatever graph existed before it entirely and replaces
        # it with the loaded definition, which is fully self-describing
        # (verified directly: a bare `nnodely(...).loadModel()`, with NO
        # prior addModel/Input/Fuzzify calls at all, already reconstructs
        # Inputs=['acc','accy','crt','delta','vel','yaw_rate'] correctly).
        # So none of the manual graph construction was ever necessary, and
        # -- more importantly -- it was never the actual bug either; it just
        # couldn't have been, given loadModel() ignores it. Root cause of
        # the 0.4908-vs-0.1158 gap is therefore still OPEN as of this
        # comment; simplifying to the minimal correct sequence here to stop
        # chasing a red herring, see the caller for the real fix.
        dyn = nnodely(visualizer='Standard', seed=42,
                      workspace=self._model_dir, save_history=False)
        dyn.loadModel()
        dyn.neuralizeModel()

        # BUGFIX 2026-08-13 (THE real root cause of the 0.408-vs-0.1158 gap,
        # confirmed empirically against diag_eval.py's batch-mode getSamples()
        # eval, which DOES reproduce 0.1158 exactly against this same
        # checkpoint -- see report/Fortschrittsbericht_VehicleDynamics_V1_
        # 20260813_DE.txt). The loaded model's Inputs carry their own exact
        # backward/forward window requirement (net.json 'ns': acc/delta/vel
        # need ns=[20,1], i.e. 20 backward + 1 forward = 21 rows total per
        # sample). reset()'s buffer used to be sized max(20,15)+5=25 rows --
        # 4 rows too many. getSamples(name, index=0, window=1) returns the
        # OLDEST valid window in whatever series it's given, i.e. rows
        # [0:21] of a 25-row buffer -- 4 timesteps STALE relative to the
        # value just pushed at the tail. Sizing the buffer to exactly
        # ns_backward+ns_forward makes index=0 the ONLY valid window, which
        # is then necessarily the freshest one.
        ns_backward, ns_forward = dyn._model_def['Info']['ns']
        self._ns_backward = ns_backward
        self._ns_forward = ns_forward
        return dyn

    def reset(self):
        """Clear the sliding-window buffers. Call at episode start."""
        w = self._ns_backward + self._ns_forward
        self._buf = {k: [0.0]*w for k in ('yaw_rate','vel','delta','acc','accy','crt')}
        # BUGFIX 2026-08-13: nnodely's loadData() raises NameError on the 2nd+ call if
        # given the same dataset name twice in one process (registry is process-global,
        # no overwrite/unload API -- confirmed via nnodely.operators.loader.Loader, only
        # filterData/getSamples/loadData/resamplingData/resetStates exist). The original
        # step() below used a constant name 'infer_buf', so calling step() more than once
        # per process crashed -- this was never caught because nothing had actually run
        # step() in a loop end-to-end before (14_eval_and_plot.py's DATA_DIR default also
        # pointed nowhere, so this path was untested). Same documented pattern as the
        # 'eval_set already used as Dataset' issue elsewhere in this project: use a
        # per-call unique name via a counter.

    def step(self, delta: float, vel: float, acc: float,
             accy: float, crt: float) -> tuple[float, float]:
        """
        Push one timestep of measurements and return next-step predictions.

        Returns:
            (yr_pred, ax_pred) — yaw_rate [rad/s] and longitudinal acceleration [m/s²]
        """
        buf = self._buf
        for k, v in (('delta',delta),('vel',vel),('acc',acc),('accy',accy),('crt',crt)):
            buf[k].append(v); buf[k].pop(0)
        buf['yaw_rate'].append(0.0); buf['yaw_rate'].pop(0)  # placeholder; model predicts next

        ds = {k: np.array(v, dtype=np.float64) for k,v in buf.items()}
        ds_name = f'infer_buf_{next(self._dataset_name_counter)}'
        self._dyn.loadData(name=ds_name, source=ds)
        s = self._dyn.getSamples(ds_name, index=0, window=1)
        o = self._dyn(s, sampled=True)
        yr_pred = float(np.asarray(o['yaw_rate_']).flatten()[0])
        ax_pred = float(np.asarray(o['acceleration']).flatten()[0])
        return yr_pred, ax_pred

    @staticmethod
    def available_models(base_dir: str = None) -> list[str]:
        """Return list of trained model directories."""
        if base_dir is None:
            base_dir = os.path.join(_SCRIPT_DIR, 'trained_models')
        if not os.path.isdir(base_dir):
            return []
        return [d for d in os.listdir(base_dir)
                if d.startswith('msnn_') and
                   os.path.exists(os.path.join(base_dir, d, 'net.json'))]


if __name__ == '__main__':
    # Quick smoke test: load exp8 model if available, run 5 steps
    import os
    base = os.path.join(_SCRIPT_DIR, 'trained_models')
    available = MSNNInference.available_models(base)
    print(f"Available MS-NN models: {available}")
    if available:
        m = sorted(available)[0]
        print(f"Loading: {m}")
        infer = MSNNInference(model_dir=os.path.join(base, m))
        infer.reset()
        for i in range(5):
            yr, ax = infer.step(delta=0.05*i, vel=2.0, acc=0.2, accy=-0.1, crt=60.0)
            print(f"  step {i}: yr_pred={yr:.4f} rad/s  ax_pred={ax:.4f} m/s²")
        print("inference_msnn.py smoke test OK")
    else:
        print("No trained models found — train first.")
