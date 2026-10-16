"""
run_training_run30.py — Run30: längere Trainingszeit (200ep, patience=30)

Motivation:
  Run22 hatte num_epochs=80, patience=15.
  Auf dem großen Day2+Day3-Datensatz ist das Modell nach 80 Epochen
  vielleicht noch nicht vollständig konvergiert. Mehr Epochen + mehr
  Geduld (patience) geben dem Modell mehr Zeit, den Parameterraum zu
  erkunden.

  Annahme: bei großer Datenmenge tritt Overfitting langsamer auf,
  längeres Training könnte sich also weiter verbessern.
  Kosten: ~2.5× längere Trainingszeit.

Alle anderen Hyperparameter identisch zu Run22, einzige Variable =
Trainingsdauer.
"""
import sys, types

wandb_stub = types.ModuleType('wandb')
class _Dict:
    def __getitem__(self, k): return None
    def __setitem__(self, k, v): pass
    def update(self, d=None, **kw): pass
class _FakeConfig:
    @staticmethod
    def update(d=None, allow_val_change=False): pass
class _FakeRun:
    name = 'local_run'
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
    "model_name":               'RNN_jomoh_1',
    "random_seed":              42,
    "learning_rate":            0.003,
    "batch_size":               700,
    "num_epochs":               200,     # was 80 in Run22
    "criterion":                'Huber',
    "optimizer":                'AdamW',
    "weight_decay":             0.000242,
    "gradient_clipping_value":  0.0,
    "patience":                 30,      # was 15 in Run22
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
print("  Run30 — Extended Training (200ep, patience=30)")
print("  Run22 baseline: CL-RMSE=0.7802 @ 80ep/patience=15")
print("  Hypothesis: more data needs more training epochs")
print("=" * 60)

training.train_network(params)
