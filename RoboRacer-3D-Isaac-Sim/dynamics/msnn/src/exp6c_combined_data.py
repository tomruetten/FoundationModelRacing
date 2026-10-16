"""
Exp6c: Combined Day2+Day3 training data — Phase1+2 (noax).
Train: Day2 113922 (t=20-102s) + Day3 163324 (t=0-87s) combined
Val:   Day3 163324 (t=87-130s) — same as exp5 val
Test:  Day1 cross-day
Goal:  Test if more diverse training data improves yaw_rate and ax.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)
from nnodely import *
from nnodely.support import earlystopping
from nnodely.support.initializer import *

DATA_DIR = os.path.join(WORK, 'processed_data')
P_PATH   = os.path.join(WORK, 'trained_models', 'msnn_exp6c_combined')
OUT_JSON = '/tmp/exp6c_combined_results.json'
os.makedirs(P_PATH, exist_ok=True)

# ── Day3 val (same as exp5) ───────────────────────────────────────────────────
df_d3 = pd.read_csv(os.path.join(DATA_DIR, 'session_20260728_163324_seg1_20ms.csv'), index_col=0)
df_d3 = df_d3.dropna(subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y'])
RACING_ROWS = 6500
df_d3_racing = df_d3.iloc[:RACING_ROWS].reset_index(drop=True)
d3_train_rows = int(len(df_d3_racing) * 2/3)
df_d3_train = df_d3_racing.iloc[:d3_train_rows].reset_index(drop=True)
df_val = df_d3_racing.iloc[d3_train_rows:].reset_index(drop=True)

# ── Day2 train ────────────────────────────────────────────────────────────────
df_d2 = pd.read_csv(os.path.join(DATA_DIR, 'session_20260722_113922_seg1_20ms.csv'), index_col=0)
df_d2 = df_d2.dropna(subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y'])
df_d2_train = df_d2.iloc[1000:].reset_index(drop=True)  # skip startup t=0-20s

# Combined train
df_train = pd.concat([df_d3_train, df_d2_train], ignore_index=True)
df_test  = pd.read_csv(os.path.join(DATA_DIR, 'session_20260721_20ms.csv'), index_col=0)

print(f"[EXP6c] Combined Day2+Day3 training", flush=True)
print(f"Train Day3: {len(df_d3_train)} rows = {len(df_d3_train)*0.02:.1f}s", flush=True)
print(f"Train Day2: {len(df_d2_train)} rows = {len(df_d2_train)*0.02:.1f}s", flush=True)
print(f"Train total: {len(df_train)} rows = {len(df_train)*0.02:.1f}s  speed={df_train['odom_speed'].mean():.2f} m/s", flush=True)
print(f"Val (Day3): {len(df_val)} rows = {len(df_val)*0.02:.1f}s  speed={df_val['odom_speed'].mean():.2f} m/s", flush=True)

def to_dict(df):
    return {'yaw_rate': df['IMU_angular_velocity_z'].values.astype(np.float64),
            'vel':      df['odom_speed'].values.astype(np.float64),
            'delta':    df['steering_cmd'].values.astype(np.float64),
            'acc':      df['IMU_linear_acceleration_x'].values.astype(np.float64),
            'accy':     df['IMU_linear_acceleration_y'].values.astype(np.float64),
            'crt':      df['current_motor'].values.astype(np.float64)}

ds_train = to_dict(df_train)
ds_val   = to_dict(df_val)
ds_test  = to_dict(df_test)

all_vx    = df_train['odom_speed'].values
all_delta = df_train['steering_cmd'].values
all_accy  = df_train['IMU_linear_acceleration_y'].values

persist_yr = float(np.sqrt(np.mean(np.diff(ds_val['yaw_rate'])**2)))
persist_ax = float(np.sqrt(np.mean(np.diff(ds_val['acc'])**2)))
print(f"Val persistence: yaw_rate={persist_yr:.4f}  ax={persist_ax:.4f}", flush=True)

# ── noax architecture (same as exp5) ─────────────────────────────────────────
num_samples_past=20; current_window=15; n=10
num_channels_vx=2; num_channels_delta=5; num_channels_accy=4; num_channels_long_vx=2

min_vx,max_vx = float(np.nanmin(all_vx)), float(np.nanmax(all_vx))
min_delta,max_delta = float(np.nanmin(all_delta)), float(np.nanmax(all_delta))
min_accy,max_accy = float(np.nanmin(all_accy)), float(np.nanmax(all_accy))

chan_centers_vx      = list(np.linspace(min_vx-0.3,    max_vx+0.3,    num=num_channels_vx))
chan_centers_delta   = list(np.linspace(min_delta-0.1, max_delta+0.1, num=num_channels_delta))
chan_centers_accy    = list(np.linspace(min_accy, max_accy, num=num_channels_accy))
chan_centers_long_vx = list(np.linspace(min_vx, max_vx, num=num_channels_long_vx))

yaw_rate=Input('yaw_rate'); vel=Input('vel'); delta=Input('delta')
acc=Input('acc'); accy=Input('accy'); crt=Input('crt')

sc_list=[Constant('sc_'+str(i),values=chan_centers_delta[i]) for i in range(num_channels_delta)]
vc_list=[Constant('vc_'+str(i),values=chan_centers_vx[i])   for i in range(num_channels_vx)]

def uc_local(inp,vx,inp0,vx0,k1,k2,k3): return vx*(k1+k2*(inp-inp0)+k3*(vx-vx0))
uc=ParamFun(uc_local)
def uc_gen(idx):
    def f(inp,vx):
        s,sp=idx[1],idx[0]
        k1=Parameter('k1_%d_%d'%(sp,s),values=[[1e-5]])
        k2=Parameter('k2_%d_%d'%(sp,s),values=[[1e-5]])
        k3=Parameter('k3_%d_%d'%(sp,s),values=[[1e-5]])
        return uc(inp,vx,sc_list[s],vc_list[sp],k1,k2,k3)
    return f

f1=Fuzzify(centers=chan_centers_vx,   functions='Triangular')(vel.sw(num_samples_past))
f2=Fuzzify(centers=chan_centers_delta, functions='Triangular')(delta.sw(num_samples_past))
lm_ss=LocalModel(input_function=uc_gen,pass_indexes=True)(
    (delta.sw(num_samples_past),vel.sw(num_samples_past)),(f1,f2))
ft=Fuzzify(centers=chan_centers_vx,functions='Triangular')(vel.sw(num_samples_past))
lm_t=LocalModel(output_function=lambda: Fir(W_init=init_exp,
    W_init_params={'size_index':0,'max_value':1,'lambda':5,'monotonicity':'increasing'}))(lm_ss,ft)
yaw_rate_out=Output('yaw_rate_',lm_t)
accy_computed=Output('accy_computed',lm_t*vel.last())

def amb(Ty,v,Fy,d,r1,m,Kd,Cv,Cr,Iw1):
    return ((1./m)*(2*(Ty/r1)-Kd*v**2-Cv*v-Fy*torch.sin(d))-Cr*9.81)/(1.+(2./m)*(2*(Iw1/r1**2.)))
def ap(T): return torch.mul(T,torch.gt(T,0))
def an(T): return torch.mul(T,~torch.gt(T,0))
def lcl(ay,ay0,k1,k2): return k1+k2*(ay-ay0)

bv=0.0001
pt=ParamFun(ap)(crt.sw(current_window)); nt_=ParamFun(an)(crt.sw(current_window))
tpos=Linear(Tanh(Fir(output_dimension=n,b=True,
    b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},dropout=0.05)(pt)))
tneg=Linear(Tanh(Fir(output_dimension=n,b=True,
    b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},dropout=0.05)(nt_)))

r1p=Parameter('r1',values=[[5e-3]]); mp=Parameter('mass',values=[[3.788]])
Kdp=Parameter('Kd',values=[[1e-4]]); Cvp=Parameter('Cv',values=[[1e-4]])
Crp=Parameter('Cr',values=[[1e-4]]); Iw1p=Parameter('Iw1',values=[[1e-4]])
acl=[Constant('ay_center_'+str(i),values=chan_centers_accy[i]) for i in range(num_channels_accy)]
lcf=ParamFun(lcl)
def lcg(idx):
    def f(ay):
        i=idx[0]
        return lcf(ay,acl[i],Parameter('k1_ay_%d'%i,values=[[1e-5]]),Parameter('k2_ay_%d'%i,values=[[1e-5]]))
    return f
flv=Fuzzify(centers=chan_centers_long_vx,functions='Triangular')(vel.sw(current_window))
fla=Fuzzify(centers=chan_centers_accy,   functions='Triangular')(accy.sw(current_window))
lmls=LocalModel(input_function=lcg,pass_indexes=True)((accy.sw(current_window),),(fla,))
Fy=LocalModel(output_function=lambda:Fir(W_init=init_negexp,
    W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}))(lmls,flv)
mbo=ParamFun(amb,parameters_and_constants=[r1p,mp,Kdp,Cvp,Crp,Iw1p])(tpos+tneg,vel.last(),Fy,delta.last())
acc_out=Output('acceleration',mbo)

dynamics=nnodely(visualizer='Standard',seed=42,workspace=P_PATH,save_history=False)
dynamics.addModel('yaw_rate',[yaw_rate_out,accy_computed])
dynamics.addMinimize('yaw_rate_error',yaw_rate.next(),yaw_rate_out,loss_function='rmse')
dynamics.addModel('acc',[acc_out])
dynamics.addMinimize('acc_error',acc.next(),acc_out,loss_function='rmse')
dynamics.neuralizeModel()
dynamics.loadData(name='training_set', source=ds_train)
dynamics.loadData(name='validation_set', source=ds_val)
dynamics.loadData(name='test_set', source=ds_test)

def ev(lbl):
    nn=len(ds_val['yaw_rate'])-num_samples_past-5
    s=dynamics.getSamples('validation_set',index=0,window=nn)
    o=dynamics(s,sampled=True)
    yr=float(np.sqrt(np.mean((np.asarray(o['yaw_rate_']).flatten()-np.array([s['yaw_rate'][i] for i in range(nn)]).flatten())**2)))
    ax=float(np.sqrt(np.mean((np.asarray(o['acceleration']).flatten()-np.array([s['acc'][i] for i in range(nn)]).flatten())**2)))
    print(f"  [{lbl}] yr={yr:.4f}({yr/persist_yr:.1f}×p)  ax={ax:.4f}({ax/persist_ax:.1f}×p)",flush=True)
    return yr,ax

print("\n"+"="*60+"\nPhase1 (1500ep p=300)\n"+"="*60,flush=True)
dynamics.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',
    test_dataset='test_set',
    training_params={'num_of_epochs':1500,'val_batch_size':100,'train_batch_size':100,'lr':1e-3,'models':'yaw_rate'},
    optimizer='Adam',shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience':300,'error':'yaw_rate_error'})
dynamics.saveModel(); dynamics.neuralizeModel()
yr1,ax1=ev('P1')

print("\n"+"="*60+"\nPhase2 (3000ep p=500)\n"+"="*60,flush=True)
dynamics.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',
    test_dataset='test_set',
    training_params={'num_of_epochs':3000,'val_batch_size':100,'train_batch_size':100,'lr':1e-3,'models':'acc'},
    optimizer='Adam',shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience':500,'error':'acc_error'})
dynamics.saveModel(); dynamics.neuralizeModel()
yr2,ax2=ev('P2')

print("\n"+"="*60+"\nPhase3 (1000ep p=200)\n"+"="*60,flush=True)
dynamics.addConnect(accy_computed,accy)
dynamics.addClosedLoop(acc_out,accy)
dynamics.neuralizeModel()
dynamics.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',
    test_dataset='test_set',
    training_params={'num_of_epochs':1000,'val_batch_size':100,'train_batch_size':100,'lr':1e-3,
                     'connect':{'accy':'accy_computed'},'closed_loop':{'acc':'acceleration'}},
    optimizer='Adam',shuffle_data=True,
    early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience':200,'error':'yaw_rate_error'})
dynamics.saveModel(); dynamics.neuralizeModel()
yr3,ax3=ev('P3')

print("\n"+"="*60+"\nFINAL — Exp6c Combined Day2+Day3\n"+"="*60,flush=True)
print(f"  P1 yr={yr1:.4f}  P2 yr={yr2:.4f} ax={ax2:.4f}  P3 yr={yr3:.4f} ax={ax3:.4f}",flush=True)
result={'session':'day2+day3_combined','model':'noax',
        'train_rows':len(df_train),'val_rows':len(df_val),
        'yr1':yr1,'yr2':yr2,'ax2':ax2,'yr3':yr3,'ax3':ax3,
        'persist_yr':persist_yr,'persist_ax':persist_ax}
with open(OUT_JSON,'w') as f: json.dump(result,f,indent=2)
print(f"Saved: {OUT_JSON}",flush=True)
