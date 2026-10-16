"""
MS-NN-full training for F1TENTH Car02.
Trains MS-NN-lat (yaw rate) and MS-NN-long (longitudinal accel) open-loop,
then fine-tunes jointly in closed loop.

Based on Mungiello/Jahncke et al. 2026 (nnodely_roboracer).
Column mapping: steering_cmd→steer/delta, odom_speed→vx/vel,
IMU_linear_acceleration_x→ax/acc, IMU_angular_velocity_z→yaw_rate,
current_motor→crt, IMU_linear_acceleration_y→accy.

CRITICAL: session_20260721 = TEST ONLY. Never in training.
"""

import os, sys, glob
import numpy as np
import pandas as pd
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

from nnodely import *
from nnodely.support import earlystopping
from nnodely.support.initializer import *

DATA_DIR   = os.path.join(WORK, 'processed_data')
MODEL_PATH = os.path.join(WORK, 'trained_models', 'msnn_full_car02')

# ── Load data ────────────────────────────────────────────────────────────────

IMU_COLS = ['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y']
EXCLUDE_SESSIONS = ['124427']  # 100% NaN IMU data in this session

def _concat(patterns):
    dfs = []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(DATA_DIR, pat))):
            bn = os.path.basename(f)
            if 'mirror' in bn:
                continue
            if any(s in bn for s in EXCLUDE_SESSIONS):
                continue
            df = pd.read_csv(f, index_col=0)
            # Drop rows where any IMU column is NaN
            imu_present = [c for c in IMU_COLS if c in df.columns]
            if imu_present:
                df = df.dropna(subset=imu_present)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_train = _concat(['session_20260722_*.csv'])   # Day2 → training
df_val   = _concat(['session_20260728_*.csv'])   # Day3 → validation
df_test  = pd.read_csv(os.path.join(DATA_DIR, 'session_20260721_20ms.csv'), index_col=0)

print(f"Train (Day2): {len(df_train)} samples = {len(df_train)*0.02:.1f}s")
print(f"Val   (Day3): {len(df_val)}   samples = {len(df_val)*0.02:.1f}s")
print(f"Test  (Day1): {len(df_test)}  samples = {len(df_test)*0.02:.1f}s")

def _to_dict(df):
    return {
        'steer':    df['steering_cmd'].values.astype(np.float64),
        'vx':       df['odom_speed'].values.astype(np.float64),
        'yaw_rate': df['IMU_angular_velocity_z'].values.astype(np.float64),
        'ax':       df['IMU_linear_acceleration_x'].values.astype(np.float64),
        'vel':      df['odom_speed'].values.astype(np.float64),
        'crt':      df['current_motor'].values.astype(np.float64),
        'acc':      df['IMU_linear_acceleration_x'].values.astype(np.float64),
        'delta':    df['steering_cmd'].values.astype(np.float64),
        'accy':     df['IMU_linear_acceleration_y'].values.astype(np.float64),
    }

ds_train = _to_dict(df_train)
ds_val   = _to_dict(df_val)
ds_test  = _to_dict(df_test)

# Training data stats for fuzzy channel initialization
all_steer = np.concatenate([df_train['steering_cmd'].values, df_val['steering_cmd'].values])
all_vx    = np.concatenate([df_train['odom_speed'].values,   df_val['odom_speed'].values])
all_ax    = np.concatenate([df_train['IMU_linear_acceleration_x'].values, df_val['IMU_linear_acceleration_x'].values])
all_accy  = np.concatenate([df_train['IMU_linear_acceleration_y'].values, df_val['IMU_linear_acceleration_y'].values])

# ── LATERAL MODEL (MS-NN-lat) ─────────────────────────────────────────────────
# Inputs shared between lat and long: vel=vx, delta=steer, acc=ax
# These are declared with the COMBINED names (vel, delta, acc) for unified dataset

yaw_rate = Input('yaw_rate')   # [rad/s] — lateral model target
vel      = Input('vel')        # [m/s]   — longitudinal speed (shared)
delta    = Input('delta')      # [rad]   — steering angle (shared)
acc      = Input('acc')        # [m/s²]  — longitudinal accel (shared)

# Fuzzy channel centers computed from training data
num_samples_past   = 20
num_channels_vx    = 2
num_channels_delta = 5
num_channels_ax    = 2

min_vx, max_vx     = float(np.nanmin(all_vx)),    float(np.nanmax(all_vx))
min_delta, max_delta = float(np.nanmin(all_steer)), float(np.nanmax(all_steer))
min_ax, max_ax     = float(np.nanmin(all_ax)),    float(np.nanmax(all_ax))

chan_centers_vx    = list(np.linspace(min_vx - 0.3,    max_vx + 0.3,    num=num_channels_vx))
chan_centers_delta  = list(np.linspace(min_delta - 0.1, max_delta + 0.1, num=num_channels_delta))
chan_centers_ax    = list(np.linspace(min_ax - 1.0,    max_ax + 1.0,    num=num_channels_ax))

chan_centers_steer_list, chan_centers_speed_list, chan_centers_ax_list = [], [], []
for i in range(num_channels_delta):
    chan_centers_steer_list.append(Constant('steer_center_' + str(i), values=chan_centers_delta[i]))
for i in range(num_channels_vx):
    chan_centers_speed_list.append(Constant('vx_center_' + str(i), values=chan_centers_vx[i]))
for i in range(num_channels_ax):
    chan_centers_ax_list.append(Constant('ax_center_' + str(i), values=chan_centers_ax[i]))

def understeer_corr_local(inp, vx, inp_0, vx_0, ax, ax_0, k1, k2, k3, k4):
    return vx * (k1 + k2*(inp - inp_0) + k3*(vx - vx_0) + k4*(ax - ax_0))

understeer_corr = ParamFun(understeer_corr_local)

def understeer_corr_gen(idx_list):
    def understeer_corr_map(inp, vx, ax):
        i_ax    = idx_list[2]
        i_steer = idx_list[1]
        i_speed = idx_list[0]
        k1 = Parameter('k1_' + str(i_speed) + '_' + str(i_steer) + '_' + str(i_ax), values=[[1e-5]])
        k2 = Parameter('k2_' + str(i_speed) + '_' + str(i_steer) + '_' + str(i_ax), values=[[1e-5]])
        k3 = Parameter('k3_' + str(i_speed) + '_' + str(i_steer) + '_' + str(i_ax), values=[[1e-5]])
        k4 = Parameter('k4_' + str(i_speed) + '_' + str(i_steer) + '_' + str(i_ax), values=[[1e-5]])
        inp_0 = chan_centers_steer_list[i_steer]
        vx_0  = chan_centers_speed_list[i_speed]
        ax_0  = chan_centers_ax_list[i_ax]
        return understeer_corr(inp, vx, inp_0, vx_0, ax, ax_0, k1, k2, k3, k4)
    return understeer_corr_map

fuzz_lat_transient = Fuzzify(centers=chan_centers_vx, functions='Triangular')(vel.sw(num_samples_past))
fuzz_ax_transient  = Fuzzify(centers=chan_centers_ax, functions='Triangular')(acc.sw(num_samples_past))

fuzz_lat_ss1 = Fuzzify(centers=chan_centers_vx,    functions='Triangular')(vel.sw(num_samples_past))
fuzz_lat_ss2 = Fuzzify(centers=chan_centers_delta,  functions='Triangular')(delta.sw(num_samples_past))
fuzz_lat_ss3 = Fuzzify(centers=chan_centers_ax,    functions='Triangular')(acc.sw(num_samples_past))

local_model_ss = LocalModel(input_function=understeer_corr_gen, pass_indexes=True)(
    (delta.sw(num_samples_past), vel.sw(num_samples_past), acc.sw(num_samples_past)),
    (fuzz_lat_ss1, fuzz_lat_ss2, fuzz_lat_ss3)
)

local_model_transient = LocalModel(
    output_function=lambda: Fir(
        W_init=init_exp,
        W_init_params={'size_index': 0, 'max_value': 1, 'lambda': 5, 'monotonicity': 'increasing'}
    )
)(local_model_ss, (fuzz_lat_transient, fuzz_ax_transient))

yaw_rate_out  = Output('yaw_rate_', local_model_transient)
accy_computed = Output('accy_computed', local_model_transient * vel.last())

# ── LONGITUDINAL MODEL (MS-NN-long) ──────────────────────────────────────────

crt  = Input('crt')    # [A] motor current
accy = Input('accy')   # [m/s²] lateral accel

# Physics model for longitudinal acceleration
def acc_model_based(Ty, v, F_y, delta_in,
                    r1, mass, Kd, Cv, Cr, Iw1):
    g_acc = 9.81
    return (
        (1.0/mass) * (2*(Ty/r1) - Kd * v**2 - Cv * v - F_y * torch.sin(delta_in)) - Cr * g_acc
    ) / (1.0 + (2.0/mass) * (2*(Iw1/r1**2.0)))

def acc_pos(T):
    return torch.mul(T, torch.gt(T, 0))

def acc_neg(T):
    return torch.mul(T, ~torch.gt(T, 0))

def long_corr_lat_local(accy_in, accy_0, k1, k2):
    return k1 + k2*(accy_in - accy_0)

# Longitudinal model hyperparams
n              = 10
current_window = 15
num_channels_accy = 4
num_channels_long_vx = 2

min_accy, max_accy = float(np.nanmin(all_accy)), float(np.nanmax(all_accy))
chan_centers_accy    = list(np.linspace(min_accy, max_accy, num=num_channels_accy))
chan_centers_long_vx = list(np.linspace(min_vx, max_vx, num=num_channels_long_vx))

# Torque network
pos_torque = ParamFun(acc_pos)(crt.sw(current_window))
neg_torque = ParamFun(acc_neg)(crt.sw(current_window))

bias_value = 0.0001
torque_layer_11 = Fir(
    output_dimension=n, b=True,
    b_init=init_negexp, b_init_params={'size_index': 0, 'first_value': bias_value, 'lambda': 3},
    W_init=init_negexp, W_init_params={'size_index': 0, 'first_value': 0.0001, 'lambda': 3},
    dropout=0.05
)(pos_torque)
torque_layer_21 = Tanh(torque_layer_11)
torque_pos = Linear(torque_layer_21)

torque_layer_12 = Fir(
    output_dimension=n, b=True,
    b_init=init_negexp, b_init_params={'size_index': 0, 'first_value': bias_value, 'lambda': 3},
    W_init=init_negexp, W_init_params={'size_index': 0, 'first_value': 0.0001, 'lambda': 3},
    dropout=0.05
)(neg_torque)
torque_layer_22 = Tanh(torque_layer_12)
torque_neg = Linear(torque_layer_22)

# Learnable physics parameters (initial guesses, car02: mass=3.788kg, wheel_r≈0.05m)
r1_p    = Parameter('r1',   values=[[5e-3]])
mass_p  = Parameter('mass', values=[[3.5]])
Kd_p    = Parameter('Kd',   values=[[1e-4]])
Cv_p    = Parameter('Cv',   values=[[1e-4]])
Cr_p    = Parameter('Cr',   values=[[1e-4]])
Iw1_p   = Parameter('Iw1',  values=[[1e-4]])

chan_centers_accy_list = [
    Constant('ay_center_' + str(i), values=chan_centers_accy[i])
    for i in range(num_channels_accy)
]

long_corr_lat = ParamFun(long_corr_lat_local)

def long_corr_lat_gen(idx_list):
    def long_corr_lat_map(accy_in):
        i_accy = idx_list[0]
        k1 = Parameter('k1_ay_' + str(i_accy), values=[[1e-5]])
        k2 = Parameter('k2_ay_' + str(i_accy), values=[[1e-5]])
        accy_0 = chan_centers_accy_list[i_accy]
        return long_corr_lat(accy_in, accy_0, k1, k2)
    return long_corr_lat_map

fuzz_long_vx   = Fuzzify(centers=chan_centers_long_vx,  functions='Triangular')(vel.sw(current_window))
fuzz_long_accy = Fuzzify(centers=chan_centers_accy, functions='Triangular')(accy.sw(current_window))

local_model_long_ss = LocalModel(input_function=long_corr_lat_gen, pass_indexes=True)(
    (accy.sw(current_window),),
    (fuzz_long_accy,)
)

Fy = LocalModel(
    output_function=lambda: Fir(
        W_init=init_negexp,
        W_init_params={'size_index': 0, 'first_value': 0.0001, 'lambda': 5, 'monotonicity': 'increasing'}
    )
)(local_model_long_ss, fuzz_long_vx)

model_based_out = ParamFun(
    acc_model_based,
    parameters_and_constants=[r1_p, mass_p, Kd_p, Cv_p, Cr_p, Iw1_p]
)(torque_pos + torque_neg, vel.last(), Fy, delta.last())

acc_out = Output('acceleration', model_based_out)

# ── Build combined nnodely model ──────────────────────────────────────────────

dynamics = nnodely(visualizer='Standard', seed=12, workspace=MODEL_PATH, save_history=False)

dynamics.addModel('yaw_rate', [yaw_rate_out, accy_computed])
dynamics.addMinimize('yaw_rate_error', yaw_rate.next(), yaw_rate_out, loss_function='rmse')

dynamics.addModel('acc', [acc_out])
dynamics.addMinimize('acc_error', acc.next(), acc_out, loss_function='rmse')

dynamics.neuralizeModel()

dynamics.loadData(name='training_set',   source=ds_train)
dynamics.loadData(name='validation_set', source=ds_val)
dynamics.loadData(name='test_set',       source=ds_test)

# ── Phase 1: Train MS-NN-lat (yaw rate model) ─────────────────────────────────

print("\n" + "="*60)
print("Phase 1: Training MS-NN-lat (yaw rate)")
print("="*60)

lat_params = {
    'num_of_epochs':    1500,
    'val_batch_size':   100,
    'train_batch_size': 100,
    'lr':               1e-3,
    'models':           'yaw_rate',
}

dynamics.trainAndAnalyze(
    train_dataset='training_set',
    validation_dataset='validation_set',
    test_dataset='test_set',
    training_params=lat_params,
    optimizer='Adam',
    shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience': 300, 'error': 'yaw_rate_error'},
)

# ── Phase 2: Train MS-NN-long (acceleration model) ────────────────────────────

print("\n" + "="*60)
print("Phase 2: Training MS-NN-long (longitudinal acceleration)")
print("="*60)

long_params = {
    'num_of_epochs':    3000,
    'val_batch_size':   100,
    'train_batch_size': 100,
    'lr':               1e-3,
    'models':           'acc',
}

dynamics.trainAndAnalyze(
    train_dataset='training_set',
    validation_dataset='validation_set',
    test_dataset='test_set',
    training_params=long_params,
    optimizer='Adam',
    shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience': 500, 'error': 'acc_error'},
)

# ── Phase 3: Closed-loop fine-tuning ─────────────────────────────────────────

print("\n" + "="*60)
print("Phase 3: Closed-loop fine-tuning")
print("="*60)

dynamics.addConnect(accy_computed, accy)
dynamics.addClosedLoop(acc_out, accy)
dynamics.neuralizeModel()

cl_params = {
    'num_of_epochs':    1000,
    'val_batch_size':   100,
    'train_batch_size': 100,
    'lr':               1e-3,
    'connect':      {'accy': 'accy_computed'},
    'closed_loop':  {'acc': 'acceleration'},
}

dynamics.trainAndAnalyze(
    train_dataset='training_set',
    validation_dataset='validation_set',
    test_dataset='test_set',
    training_params=cl_params,
    optimizer='Adam',
    shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience': 200, 'error': 'yaw_rate_error'},
)

dynamics.saveModel()
print(f"\nModel saved to: {MODEL_PATH}")

# ── Quick validation RMSE ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("Validation RMSE (open-loop, one-step)")
print("="*60)

dynamics.neuralizeModel()

num_use = len(ds_val['yaw_rate']) - num_samples_past - 5
samples = dynamics.getSamples('validation_set', index=0, window=num_use)
out     = dynamics(samples, sampled=True)

yaw_pred  = np.asarray(out['yaw_rate_']).flatten()
yaw_true  = np.array([samples['yaw_rate'][i] for i in range(len(yaw_pred))]).flatten()
acc_pred   = np.asarray(out['acceleration']).flatten()
acc_true   = np.array([samples['acc'][i][current_window] for i in range(len(acc_pred))]).flatten()

rmse_yaw = np.sqrt(np.mean((yaw_pred - yaw_true)**2))
rmse_acc = np.sqrt(np.mean((acc_pred  - acc_true)**2))

print(f"  Ω  RMSE (val): {rmse_yaw:.4f} rad/s")
print(f"  ax RMSE (val): {rmse_acc:.4f} m/s²")
print("="*60)
print("MS-NN-full training complete. Run eval_msnn_car02.py for test-set evaluation.")
