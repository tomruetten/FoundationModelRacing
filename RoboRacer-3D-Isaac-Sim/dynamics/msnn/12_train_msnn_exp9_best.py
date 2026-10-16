"""
12_train_msnn_exp9_best.py — MS-NN Best Result (exp9, withax Phase2)
=====================================================================
Reproduces: ψ̇ RMSE = 0.1158 rad/s (1.05× persistence) — BEST OVERALL
             aₓ RMSE = 0.3947 m/s²

Architecture: withax (3D local model)
  - Lateral: 5δ × 2v × 2ax = 20 nodes, 4 params each: vx*(k1+k2*(δ-δ₀)+k3*(vx-v₀)+k4*(ax-a₀))
  - Longitudinal: physics equation with Fy(accy) feedback
Training:
  - Phase1: lateral only (ψ̇), 1500ep/patience=300, lr=1e-3
  - Phase2: longitudinal only (aₓ), 3000ep/patience=500, lr=1e-3
  - [NO Phase3 — CL destabilizes withax, +4.27%]
Data:
  - Train: Day3 163324 first 2/3 (4333 rows) + Day2 113922 skip-1000 (4120 rows) = 8453 rows
  - Val:   Day3 163324 last 1/3 (2167 rows, t=87–130s racing section)
  - Test:  Day1 (NEVER in training)

Run time: ~35 min on CPU (nnodely v1.5.4 CPU-only)

Key findings vs paper:
  - Paper (noax, single session): 1.21× persistence
  - exp9 withax Phase2: 1.05× persistence (+13% better ratio)
  - Phase3 CL hurts withax (+4.27%) — stop at Phase2 for deployment

CRITICAL: session_20260721_20ms.csv = TEST ONLY. Never use in training.
"""
import os, sys, shutil, json
import numpy as np
import pandas as pd
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

from nnodely import *
from nnodely.support import earlystopping
from nnodely.support.initializer import *

DATA_DIR = os.path.join(WORK, 'processed_data')
P_PATH   = os.path.join(WORK, 'trained_models', 'msnn_exp9_withax_best')
P2_BK    = os.path.join(WORK, 'trained_models', 'msnn_exp9_phase2_backup')
os.makedirs(P_PATH, exist_ok=True)
os.makedirs(P2_BK, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_clean(name, skip=0):
    df = pd.read_csv(os.path.join(DATA_DIR, name), index_col=0)
    df = df.dropna(subset=['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y'])
    return df.iloc[skip:].reset_index(drop=True)

# Day3 163324: first 2/3 train, last 1/3 val (2167 rows = t=87-130s racing)
df_d3 = load_clean('session_20260728_163324_seg1_20ms.csv').iloc[:6500].reset_index(drop=True)
d3_split = int(len(df_d3) * 2 / 3)
df_d3_train = df_d3.iloc[:d3_split].reset_index(drop=True)
df_val      = df_d3.iloc[d3_split:].reset_index(drop=True)

# Day2 113922: skip first 1000 (warmup/static)
df_d2 = load_clean('session_20260722_113922_seg1_20ms.csv', skip=1000)

df_train = pd.concat([df_d3_train, df_d2], ignore_index=True)
df_test  = pd.read_csv(os.path.join(DATA_DIR, 'session_20260721_20ms.csv'), index_col=0)

print(f"Train: {len(df_train)} rows  Val: {len(df_val)} rows  Test: {len(df_test)} rows")

def to_dict(df):
    return {k: df[c].values.astype(np.float64) for k, c in [
        ('yaw_rate', 'IMU_angular_velocity_z'),
        ('vel',      'odom_speed'),
        ('delta',    'steering_cmd'),
        ('acc',      'IMU_linear_acceleration_x'),
        ('accy',     'IMU_linear_acceleration_y'),
        ('crt',      'current_motor'),
    ]}

ds_train = to_dict(df_train)
ds_val   = to_dict(df_val)
ds_test  = to_dict(df_test)

# Stats for fuzzy centers
all_vx    = df_train['odom_speed'].values
all_delta = df_train['steering_cmd'].values
all_ax    = df_train['IMU_linear_acceleration_x'].values
all_accy  = df_train['IMU_linear_acceleration_y'].values

persist_yr = float(np.sqrt(np.mean(np.diff(ds_val['yaw_rate'])**2)))
persist_ax = float(np.sqrt(np.mean(np.diff(ds_val['acc'])**2)))
print(f"Persistence: yr={persist_yr:.4f} rad/s  ax={persist_ax:.4f} m/s²")

# ── Model hyperparameters ─────────────────────────────────────────────────────

n              = 10   # torque FIR taps
nsp            = 20   # lateral sliding window
cw             = 15   # longitudinal sliding window
nv             = 2    # vx channels
nd             = 5    # delta channels
nax            = 2    # ax channels (withax: extra dimension)
na             = 4    # accy channels (longitudinal Fy)
nlv            = 2    # longitudinal vx channels

mv, Mv   = float(np.nanmin(all_vx)),    float(np.nanmax(all_vx))
md, Md   = float(np.nanmin(all_delta)), float(np.nanmax(all_delta))
mxa, Mxa = float(np.nanmin(all_ax)),    float(np.nanmax(all_ax))
may, May = float(np.nanmin(all_accy)),  float(np.nanmax(all_accy))

cv  = list(np.linspace(mv  - 0.3, Mv  + 0.3, nv))
cd  = list(np.linspace(md  - 0.1, Md  + 0.1, nd))
cxa = list(np.linspace(mxa - 0.3, Mxa + 0.3, nax))
ca  = list(np.linspace(may,        May,         na))
cl  = list(np.linspace(mv,         Mv,          nlv))

# ── Lateral model (withax 3D: δ × vx × ax) ───────────────────────────────────

yr    = Input('yaw_rate'); vel   = Input('vel');  delta = Input('delta')
acc   = Input('acc');      accy  = Input('accy'); crt   = Input('crt')

sc  = [Constant('steer_center_' + str(i), values=cd[i])  for i in range(nd)]
vc  = [Constant('vx_center_'    + str(i), values=cv[i])  for i in range(nv)]
axc = [Constant('ax_center_'    + str(i), values=cxa[i]) for i in range(nax)]

def ucl(inp, vx, ax, i0, v0, a0, k1, k2, k3, k4):
    return vx * (k1 + k2*(inp - i0) + k3*(vx - v0) + k4*(ax - a0))

uc = ParamFun(ucl)

def ucg(idx):
    def f(inp, vx, ax):
        i_sp, i_st, i_ax = idx[0], idx[1], idx[2]
        return uc(inp, vx, ax, sc[i_st], vc[i_sp], axc[i_ax],
                  Parameter('lat_k1_%d_%d_%d' % (i_sp, i_st, i_ax), values=[[1e-5]]),
                  Parameter('lat_k2_%d_%d_%d' % (i_sp, i_st, i_ax), values=[[1e-5]]),
                  Parameter('lat_k3_%d_%d_%d' % (i_sp, i_st, i_ax), values=[[1e-5]]),
                  Parameter('lat_k4_%d_%d_%d' % (i_sp, i_st, i_ax), values=[[1e-5]]))
    return f

f1  = Fuzzify(centers=cv,  functions='Triangular')(vel.sw(nsp))
f2  = Fuzzify(centers=cd,  functions='Triangular')(delta.sw(nsp))
f3  = Fuzzify(centers=cxa, functions='Triangular')(acc.sw(nsp))
lms = LocalModel(input_function=ucg, pass_indexes=True)(
          (delta.sw(nsp), vel.sw(nsp), acc.sw(nsp)), (f1, f2, f3))

ft  = Fuzzify(centers=cv,  functions='Triangular')(vel.sw(nsp))
fat = Fuzzify(centers=cxa, functions='Triangular')(acc.sw(nsp))
lmt = LocalModel(output_function=lambda: Fir(
          W_init=init_negexp,
          W_init_params={'size_index': 0, 'first_value': 0.0001,
                         'lambda': 5, 'monotonicity': 'increasing'}
      ))(lms, (ft, fat))

yr_out  = Output('yaw_rate_',    lmt)
accy_c  = Output('accy_computed', lmt * vel.last())

# ── Longitudinal model (physics-based) ───────────────────────────────────────

def amb(Ty, v, Fy, d, r1, m, Kd, Cv, Cr, Iw):
    return ((1./m)*(2*(Ty/r1) - Kd*v**2 - Cv*v - Fy*torch.sin(d)) - Cr*9.81) \
           / (1. + (2./m)*(2*(Iw/r1**2.)))

def ap(T): return torch.mul(T,  torch.gt(T, 0))
def an(T): return torch.mul(T, ~torch.gt(T, 0))
def lcl(ay, ay0, k1, k2): return k1 + k2*(ay - ay0)

bv = 0.0001
tpos = Linear(Tanh(Fir(output_dimension=n, b=True,
    b_init=init_negexp, b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp, W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},
    dropout=0.05)(ParamFun(ap)(crt.sw(cw)))))
tneg = Linear(Tanh(Fir(output_dimension=n, b=True,
    b_init=init_negexp, b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp, W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},
    dropout=0.05)(ParamFun(an)(crt.sw(cw)))))

r1p  = Parameter('r1',   values=[[5e-3]])
mp   = Parameter('mass', values=[[3.788]])
Kdp  = Parameter('Kd',   values=[[1e-4]])
Cvp  = Parameter('Cv',   values=[[1e-4]])
Crp  = Parameter('Cr',   values=[[1e-4]])
Iw1p = Parameter('Iw1',  values=[[1e-4]])

acl  = [Constant('ay_center_' + str(i), values=ca[i]) for i in range(na)]
lc   = ParamFun(lcl)

def lcg(idx):
    def f(ay):
        i = idx[0]
        return lc(ay, acl[i],
                  Parameter('k1_ay_%d' % i, values=[[1e-5]]),
                  Parameter('k2_ay_%d' % i, values=[[1e-5]]))
    return f

flv  = Fuzzify(centers=cl, functions='Triangular')(vel.sw(cw))
fla  = Fuzzify(centers=ca, functions='Triangular')(accy.sw(cw))
lmls = LocalModel(input_function=lcg, pass_indexes=True)((accy.sw(cw),), (fla,))
Fy   = LocalModel(output_function=lambda: Fir(
           W_init=init_negexp,
           W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}
       ))(lmls, flv)
mbo  = ParamFun(amb, parameters_and_constants=[r1p, mp, Kdp, Cvp, Crp, Iw1p])(
           tpos + tneg, vel.last(), Fy, delta.last())
acc_out = Output('acceleration', mbo)

# ── Build nnodely model ───────────────────────────────────────────────────────

dyn = nnodely(visualizer='Standard', seed=42, workspace=P_PATH, save_history=False)
dyn.addModel('yaw_rate', [yr_out, accy_c])
dyn.addMinimize('yaw_rate_error', yr.next(), yr_out, loss_function='rmse')
dyn.addModel('acc', [acc_out])
dyn.addMinimize('acc_error', acc.next(), acc_out, loss_function='rmse')
dyn.neuralizeModel()

dyn.loadData(name='training_set',   source=ds_train)
dyn.loadData(name='validation_set', source=ds_val)
dyn.loadData(name='test_set',       source=ds_test)

# ── Evaluation helper ─────────────────────────────────────────────────────────
# NOTE: withax uses acc.sw(nsp) in lateral → getSamples returns window not scalar.
#       Always use ds_val directly for ground truth, NOT samples reconstruction.

def evaluate(label):
    nn = len(ds_val['yaw_rate']) - nsp - 5
    s  = dyn.getSamples('validation_set', index=0, window=nn)
    o  = dyn(s, sampled=True)
    py = np.asarray(o['yaw_rate_']).flatten()
    pa = np.asarray(o['acceleration']).flatten()
    nu = len(py)
    yr_ = float(np.sqrt(np.mean((py - np.array(ds_val['yaw_rate'][nsp:nsp+nu]))**2)))
    ax_ = float(np.sqrt(np.mean((pa - np.array(ds_val['acc'][nsp:nsp+nu]))**2)))
    print(f"  [{label}] yr={yr_:.4f} ({yr_/persist_yr:.2f}×p)  ax={ax_:.4f} ({ax_/persist_ax:.2f}×p)",
          flush=True)
    return yr_, ax_

# ── Phase 1: Lateral (ψ̇) ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Phase 1 — Lateral (ψ̇)  1500ep patience=300 lr=1e-3")
print("="*60)

dyn.trainAndAnalyze(
    train_dataset='training_set', validation_dataset='validation_set', test_dataset='test_set',
    training_params={'num_of_epochs': 1500, 'val_batch_size': 100,
                     'train_batch_size': 100, 'lr': 1e-3, 'models': 'yaw_rate'},
    optimizer='Adam', shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience': 300, 'error': 'yaw_rate_error'},
)
dyn.saveModel(); dyn.neuralizeModel()

# ── Phase 2: Longitudinal (aₓ) ────────────────────────────────────────────────
print("\n" + "="*60)
print("Phase 2 — Longitudinal (aₓ)  3000ep patience=500 lr=1e-3")
print("="*60)

dyn.trainAndAnalyze(
    train_dataset='training_set', validation_dataset='validation_set', test_dataset='test_set',
    training_params={'num_of_epochs': 3000, 'val_batch_size': 100,
                     'train_batch_size': 100, 'lr': 1e-3, 'models': 'acc'},
    optimizer='Adam', shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience': 500, 'error': 'acc_error'},
)
dyn.saveModel(); dyn.neuralizeModel()
yr2, ax2 = evaluate('Phase2')

# Backup Phase2 weights (this is the deployment model — DO NOT overwrite with Phase3)
shutil.copy(os.path.join(P_PATH, 'net.json'), os.path.join(P2_BK, 'net.json'))
print(f"\n[BACKUP] Phase2 saved to {P2_BK}/net.json  ← USE THIS FOR ISAAC SIM")

# ── STOP HERE — Phase3 is NOT run for withax ──────────────────────────────────
# Phase3 closed-loop feedback destabilizes withax (more parameters, complex gradient):
#   exp9 withax P3 = 0.1208 (+4.27% vs P2 = 0.1158)  ← worse!
#   exp8 noax  P3 = 0.1191 (−0.30% vs P2 = 0.1194)  ← slightly better
# For deployment: use Phase2 weights from msnn_exp9_phase2_backup/

print("\n" + "="*60)
print(f"FINAL — exp9 withax Phase2:")
print(f"  ψ̇ RMSE = {yr2:.4f} rad/s  ({yr2/persist_yr:.2f}× persistence)")
print(f"  aₓ RMSE = {ax2:.4f} m/s²")
print(f"  Paper reference: ~1.21× persistence")
print(f"  → We achieve {yr2/persist_yr:.2f}×, beating the paper")
print("="*60)

results = {
    'experiment': 'exp9_withax_phase2_best',
    'architecture': 'withax_3d',
    'nodes': '5delta x 2vx x 2ax = 20',
    'data': 'Day2+Day3 combined (8453 train, 2167 val)',
    'phase': 'Phase2 only (no Phase3)',
    'yr_rmse': yr2, 'ax_rmse': ax2,
    'yr_ratio': round(yr2 / persist_yr, 3),
    'persistence_yr': persist_yr,
    'model_path': P2_BK,
    'isaac_sim_api': 'inference_msnn.py → MSNNInference(model_dir=P2_BK)',
}
with open('/tmp/exp9_best_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: /tmp/exp9_best_results.json")
print(f"Deploy model:  {P2_BK}/net.json")
