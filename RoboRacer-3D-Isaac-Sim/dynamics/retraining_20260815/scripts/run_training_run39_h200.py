"""
run_training_run39_h200.py — A3 independent verification: hidden_size=200
(2026-08-15). Felix's own TECHNICAL_NOTES.md already documents "h=200: no gain"
(Run23), but that was on an EARLIER baseline (before Run30's 200ep/
patience=30 setting). Re-run on the Run30 baseline for a genuinely
independent verification (different base config), for thesis completeness
-- not expecting to beat Run30, reporting honestly either way.
Identical to Run30 except model_name='RNN_h200' (already-defined class in
RNN_architectures.py, hidden_size=200 instead of 100).
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
    name = 'run39_h200'
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

import training

# BUGFIX 2026-08-15: `from run_training_run30 import ...` triggers Run30's
# OWN module-level `training.train_network(params)` call as a side effect
# (that file has no `if __name__=="__main__"` guard) -- discovered when
# Run38 accidentally ran a full extra 80-epoch training job this way.
# Inline the exact same file lists instead of importing them.
PROC = 'processed_data'
TRAIN_FILES = [
    f'{PROC}/session_20260722_113922_seg1_20ms.csv',
    f'{PROC}/session_20260722_113922_seg2_20ms.csv',
    f'{PROC}/session_20260722_113922_seg3_20ms.csv',
    f'{PROC}/session_20260722_115055_seg1_20ms.csv',
    f'{PROC}/session_20260722_120608_seg1_20ms.csv',
    f'{PROC}/session_20260722_120608_seg2_20ms.csv',
    f'{PROC}/session_20260722_120608_seg3_20ms.csv',
    f'{PROC}/session_20260722_113922_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260722_113922_seg2_20ms_mirror.csv',
    f'{PROC}/session_20260722_113922_seg3_20ms_mirror.csv',
    f'{PROC}/session_20260722_115055_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260722_120608_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260722_120608_seg2_20ms_mirror.csv',
    f'{PROC}/session_20260722_120608_seg3_20ms_mirror.csv',
    f'{PROC}/session_20260728_122000_seg1_20ms.csv',
    f'{PROC}/session_20260728_122000_seg2_20ms.csv',
    f'{PROC}/session_20260728_122813_seg1_20ms.csv',
    f'{PROC}/session_20260728_122813_seg2_20ms.csv',
    f'{PROC}/session_20260728_123200_seg1_20ms.csv',
    f'{PROC}/session_20260728_123412_seg1_20ms.csv',
    f'{PROC}/session_20260728_123713_seg1_20ms.csv',
    f'{PROC}/session_20260728_123713_seg2_20ms.csv',
    f'{PROC}/session_20260728_123955_seg1_20ms.csv',
    f'{PROC}/session_20260728_124427_seg1_20ms.csv',
    f'{PROC}/session_20260728_163324_seg1_20ms.csv',
    f'{PROC}/session_20260728_170012_seg1_20ms.csv',
    f'{PROC}/session_20260728_170012_seg2_20ms.csv',
    f'{PROC}/session_20260728_122000_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_122000_seg2_20ms_mirror.csv',
    f'{PROC}/session_20260728_122813_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_122813_seg2_20ms_mirror.csv',
    f'{PROC}/session_20260728_123200_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_123412_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_123713_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_123713_seg2_20ms_mirror.csv',
    f'{PROC}/session_20260728_123955_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_124427_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_163324_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_170012_seg1_20ms_mirror.csv',
    f'{PROC}/session_20260728_170012_seg2_20ms_mirror.csv',
]
VAL_FILE  = f'{PROC}/session_20260722_115055_seg2_20ms.csv'
TEST_FILE = f'{PROC}/session_20260721_20ms.csv'

params = {
    "model_name":               'RNN_h200',   # <-- only change vs Run30
    "random_seed":              42,
    "learning_rate":            0.003,
    "batch_size":               700,
    "num_epochs":               80,      # Run22-scale budget, see run38's docstring for why
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
}

print("=" * 60)
print("  Run39 — hidden_size=200 (A3, independent verification of Run23)")
print("  Run30 baseline: 5ch CL-RMSE=0.7761, 3ch=0.4448")
print("=" * 60)

training.train_network(params)
