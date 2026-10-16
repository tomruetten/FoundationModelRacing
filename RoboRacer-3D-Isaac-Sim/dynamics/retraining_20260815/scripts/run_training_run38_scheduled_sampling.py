"""
run_training_run38_scheduled_sampling.py — GRU exposure-bias experiment
(2026-08-15, written on the lab machine)

Motivation:
  V1 reported GRU OL-RMSE=0.434 vs CL-RMSE=0.607 (yr) on the Day1 cross-day
  replay, framed as "exposure bias". Investigated the actual Run30 training
  loop (training.py) before assuming scheduled sampling is the right fix:
  training already does short-horizon autoregression within each 500-step
  (10s) sample (see training.py:431, next_inputs overwritten with the
  model's own detached predictions from step H onward) -- it is NOT pure
  teacher forcing. The likely real gap driver is a train/eval ROLLOUT
  HORIZON mismatch (train: <=490 self-fed steps per sample, always
  restarted from a fresh real-history window every batch; eval: thousands
  of steps, never reset, continuous over the whole Day1 file) rather than
  classic exposure bias in the Bengio et al. 2015 sense.

  Given that, this run tests textbook scheduled sampling anyway (Task 2 in
  training_v2.py, never tested on the Run22/Run30 baseline config before)
  as one candidate direction, run independently for thesis verification
  purposes, alongside the training-loop-mechanism finding as a documented
  alternative/competing explanation. Do not assume this will beat Run30 --
  report the number honestly either way.

  Exactly Run30's config otherwise (same architecture, same data, same all
  other hyperparameters) -- the ONLY variable changed is
  scheduled_sampling=True (via training_v2 instead of training v1).
  training_v2's docstring confirms "All new params are optional with safe
  defaults, so existing run scripts work unchanged" -- i.e. this is a
  minimal, single-variable swap, not a rewrite.

Data exclusion (verified before writing this script, see investigation
report): TRAIN_FILES/VAL_FILE/TEST_FILE copied verbatim from
run_training_run30.py -- session_20260721 (Day1) only ever appears in
TEST_FILE/auto_regression_files (never train/val), and training_v2.py's
train loop only backprops through train_files, matching the same
no-Day1-in-gradient-path guarantee training.py has (test_dataset is only
used post-hoc in trainAndAnalyze-style eval, never in the optimizer step).
"""
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'msnn' / 'src'))

wandb_stub = types.ModuleType('wandb')
class _Dict:
    def __getitem__(self, k): return None
    def __setitem__(self, k, v): pass
    def update(self, d=None, **kw): pass
class _FakeConfig:
    @staticmethod
    def update(d=None, allow_val_change=False): pass
class _FakeRun:
    name = 'run38_schedsamp'
    summary = _Dict()
    config = _FakeConfig()
    def log(self, d): pass
    def define_metric(self, *a, **kw): pass
    def finish(self): pass
class _FakeTable:
    def __init__(self, data=None, columns=None): pass
class _FakePlot:
    @staticmethod
    def line_series(*a, **kw): return None
    @staticmethod
    def bar(*a, **kw): return None
    def __getattr__(self, name): return lambda *a, **kw: None
wandb_stub.run = _FakeRun(); wandb_stub.init = lambda **kw: None
wandb_stub.finish = lambda: None; wandb_stub.log = lambda d: None
wandb_stub.config = _FakeConfig; wandb_stub.define_metric = lambda *a, **kw: None
wandb_stub.Table = _FakeTable; wandb_stub.plot = _FakePlot()
wandb_stub.Image = lambda *a, **kw: None
sys.modules['wandb'] = wandb_stub

import training_v2 as training
from run_training_run30 import TRAIN_FILES, VAL_FILE, TEST_FILE

params = {
    "model_name":               'RNN_jomoh_1',
    "random_seed":              42,
    "learning_rate":            0.003,
    "batch_size":               700,
    "num_epochs":               80,      # Run22-scale budget (not Run30's 200) -- see
                                          # module docstring: a single Run30-scale (200ep)
                                          # job was measured to cost too much wall-clock
                                          # time given the number of experiments queued.
                                          # Run22-scale (80ep/patience=15) is what most of
                                          # Felix's own ablations (Run23-28) used too.
    "criterion":                'Huber',
    "optimizer":                'AdamW',
    "weight_decay":             0.000242,
    "gradient_clipping_value":  0.0,
    "patience":                 15,
    "history":                  10,
    "seq_length":               500,
    "stride":                   5,
    "train_files":              TRAIN_FILES,
    "val_files":                [VAL_FILE],
    "test_files":               [TEST_FILE],
    "auto_regression_files":    [TEST_FILE],
    "outputs_columns": ["v_x_local", "v_y_local", "mocap_angular_velocity",
                        "a_x_local", "a_y_local"],
    "inputs_columns":  ["v_x_local", "v_y_local", "mocap_angular_velocity",
                        "a_x_local", "a_y_local", "speed_cmd", "steering_cmd"],
    "normed":          False,
    "num_workers":     2,
    "prefetch_factor": 2,
    "scheduled_sampling": True,   # <-- the ONLY change vs run_training_run30.py
}

print("=" * 60)
print("  Run38 — Scheduled Sampling (exposure-bias experiment)")
print("  Identical to Run30 except scheduled_sampling=True")
print("  Run30 baseline: 5ch CL-RMSE=0.7761, 3ch=0.4448")
print("=" * 60)

training.train_network(params)
