"""
run_training_run37.py — Run37: Onboard-sensors-only GRU (fair comparison with MS-NN).

Motivation:
  MS-NN (Felix paper) uses only onboard sensors: odom_speed, IMU_psi_dot,
  IMU_ax, IMU_ay, steering_cmd, current_motor — NO MoCap signals.
  Our GRU (Run30-35) uses MoCap v_x, v_y, psi_dot which are unavailable
  on the real car. This is an unfair advantage for GRU.

  Run37 levels the playing field: same inputs as MS-NN paper.

Inputs  (6ch): [odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x,
                IMU_linear_acceleration_y, steering_cmd, current_motor]
Outputs (4ch): [odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x,
                IMU_linear_acceleration_y]

Evaluation: compare IMU_psi_dot RMSE and IMU_ax RMSE with MS-NN directly.
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
    "num_epochs":               200,
    "criterion":                'Huber',
    "optimizer":                'AdamW',
    "weight_decay":             0.000242,
    "gradient_clipping_value":  0.0,
    "patience":                 30,
    "history":                  10,
    "seq_length":               500,
    "stride":                   5,
    "train_files":              TRAIN_FILES,
    "val_files":                [VAL_FILE],
    "test_files":               [TEST_FILE],
    "auto_regression_files":    [TEST_FILE],
    "outputs_columns": ["odom_speed", "IMU_angular_velocity_z",
                        "IMU_linear_acceleration_x", "IMU_linear_acceleration_y"],
    "inputs_columns":  ["odom_speed", "IMU_angular_velocity_z",
                        "IMU_linear_acceleration_x", "IMU_linear_acceleration_y",
                        "steering_cmd", "current_motor"],
    "normed":          False,
    "num_workers":     2,
    "prefetch_factor": 2,
}

print("=" * 60)
print("  Run37 — Onboard sensors only (fair vs MS-NN)")
print("  Inputs: odom_speed, IMU_psi, IMU_ax, IMU_ay, steer, current")
print("  NO MoCap signals — real-world deployable")
print("  Compare: IMU_psi RMSE and ax RMSE vs MS-NN 0.6687 / 1.5109")
print("=" * 60)

training.train_network(params)
