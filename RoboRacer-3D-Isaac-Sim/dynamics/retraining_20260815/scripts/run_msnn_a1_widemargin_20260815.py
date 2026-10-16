"""
run_msnn_a1_widemargin_20260815.py — A1: MS-NN cross-day robustness attempt
(2026-08-15).

TECHNICAL_NOTES.md's own conclusion on cross-day gap: "Gap is not from model
quality -- it's tire temperature, track condition, battery voltage. Cannot
be fixed by better models. Only more diverse training data helps." Also:
"3-session (15637 rows) worse than 2-session (8453 rows) -- diversity
hurts" (adding a 3rd whole session already tried and failed). No new
recording sessions are available, so this does NOT attempt to add more
data (already shown to hurt) or claim to fix the fundamental physical
domain-shift cause (Felix's own conclusion is that's not fixable by
modeling).

Instead, tests ONE untried, cheap, single-variable hypothesis: exp9's
fuzzy LocalModel node centers are placed with a narrow fixed extrapolation
margin beyond the training data's own min/max (vx +/-0.3, delta +/-0.1,
ax +/-0.3 -- see original exp9_withax_combined.py). If Day1's (cross-day)
vx/delta/ax distribution extends even slightly beyond Day2+Day3's observed
range, the model has to extrapolate past its outermost fuzzy node with no
margin, which could plausibly WORSEN cross-day error independent of any
"more diverse data" question. Widened margins here (vx/ax +/-1.0, delta
+/-0.3, roughly 3x wider) to test whether giving the model more room to
extrapolate gracefully helps. Same architecture, same data, same
everything else as exp9_withax_combined.py -- only the margin constants
changed. Saves to a NEW, clearly separate checkpoint location -- does NOT
touch or overwrite the deployed msnn_exp9_phase2_backup.

Data/exclusion: identical to exp9_withax_combined.py (session_20260721
Day1 only in test_set, post-hoc eval, never in train_dataset).
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
P_PATH   = os.path.join(WORK, 'trained_models', 'msnn_a1_widemargin_20260815')
P2_BK    = os.path.join(WORK, 'trained_models', 'msnn_a1_widemargin_phase2_backup_20260815')
OUT_JSON = '/tmp/msnn_a1_widemargin_results.json'
os.makedirs(P_PATH,exist_ok=True); os.makedirs(P2_BK,exist_ok=True)

df_d3=pd.read_csv(os.path.join(DATA_DIR,'session_20260728_163324_seg1_20ms.csv'),index_col=0)
df_d3=df_d3.dropna(subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y']).iloc[:6500].reset_index(drop=True)
d3_tr=int(len(df_d3)*2/3)
df_val=df_d3.iloc[d3_tr:].reset_index(drop=True)
df_d3_train=df_d3.iloc[:d3_tr].reset_index(drop=True)
df_d2=pd.read_csv(os.path.join(DATA_DIR,'session_20260722_113922_seg1_20ms.csv'),index_col=0)
df_d2=df_d2.dropna(subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y']).iloc[1000:].reset_index(drop=True)
df_train=pd.concat([df_d3_train,df_d2],ignore_index=True)
df_test=pd.read_csv(os.path.join(DATA_DIR,'session_20260721_20ms.csv'),index_col=0)
print(f"[A1] Day2+Day3 WITHAX wide-margin (A1 cross-day robustness test)  Train={len(df_train)} Val={len(df_val)}",flush=True)

def to_dict(df):
    return {k:df[c].values.astype(np.float64) for k,c in [('yaw_rate','IMU_angular_velocity_z'),
        ('vel','odom_speed'),('delta','steering_cmd'),('acc','IMU_linear_acceleration_x'),
        ('accy','IMU_linear_acceleration_y'),('crt','current_motor')]}

ds_train=to_dict(df_train); ds_val=to_dict(df_val); ds_test=to_dict(df_test)
all_vx=df_train['odom_speed'].values; all_delta=df_train['steering_cmd'].values
all_accy=df_train['IMU_linear_acceleration_y'].values; all_ax=df_train['IMU_linear_acceleration_x'].values
persist_yr=float(np.sqrt(np.mean(np.diff(ds_val['yaw_rate'])**2)))
persist_ax=float(np.sqrt(np.mean(np.diff(ds_val['acc'])**2)))
print(f"Persistence: yr={persist_yr:.4f} ax={persist_ax:.4f}",flush=True)

n=10; nsp=20; cw=15; nv=2; nd=5; nax=2; na=4; nlv=2
mv,Mv=float(np.nanmin(all_vx)),float(np.nanmax(all_vx))
md,Md=float(np.nanmin(all_delta)),float(np.nanmax(all_delta))
mxa,Mxa=float(np.nanmin(all_ax)),float(np.nanmax(all_ax))
may,May=float(np.nanmin(all_accy)),float(np.nanmax(all_accy))
# A1 EXPERIMENT: margins widened ~3x vs exp9's original (vx/ax 0.3->1.0,
# delta 0.1->0.3) -- the only variable changed in this whole script.
cv=list(np.linspace(mv-1.0,Mv+1.0,nv)); cd=list(np.linspace(md-0.3,Md+0.3,nd))
cxa=list(np.linspace(mxa-1.0,Mxa+1.0,nax)); ca=list(np.linspace(may,May,na)); cl=list(np.linspace(mv,Mv,nlv))

yr=Input('yaw_rate'); vel=Input('vel'); delta=Input('delta')
acc=Input('acc'); accy=Input('accy'); crt=Input('crt')
sc=[Constant('steer_center_'+str(i),values=cd[i]) for i in range(nd)]
vc=[Constant('vx_center_'+str(i),values=cv[i]) for i in range(nv)]
axc=[Constant('ax_center_'+str(i),values=cxa[i]) for i in range(nax)]
def ucl(inp,vx,ax,i0,v0,a0,k1,k2,k3,k4): return vx*(k1+k2*(inp-i0)+k3*(vx-v0)+k4*(ax-a0))
uc=ParamFun(ucl)
def ucg(idx):
    def f(inp,vx,ax):
        i_sp,i_st,i_ax=idx[0],idx[1],idx[2]
        return uc(inp,vx,ax,sc[i_st],vc[i_sp],axc[i_ax],
                  Parameter('lat_k1_%d_%d_%d'%(i_sp,i_st,i_ax),values=[[1e-5]]),
                  Parameter('lat_k2_%d_%d_%d'%(i_sp,i_st,i_ax),values=[[1e-5]]),
                  Parameter('lat_k3_%d_%d_%d'%(i_sp,i_st,i_ax),values=[[1e-5]]),
                  Parameter('lat_k4_%d_%d_%d'%(i_sp,i_st,i_ax),values=[[1e-5]]))
    return f
f1=Fuzzify(centers=cv,functions='Triangular')(vel.sw(nsp))
f2=Fuzzify(centers=cd,functions='Triangular')(delta.sw(nsp))
f3=Fuzzify(centers=cxa,functions='Triangular')(acc.sw(nsp))
lms=LocalModel(input_function=ucg,pass_indexes=True)((delta.sw(nsp),vel.sw(nsp),acc.sw(nsp)),(f1,f2,f3))
ft=Fuzzify(centers=cv,functions='Triangular')(vel.sw(nsp))
fat=Fuzzify(centers=cxa,functions='Triangular')(acc.sw(nsp))
lmt=LocalModel(output_function=lambda:Fir(W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}))(lms,(ft,fat))
yr_out=Output('yaw_rate_',lmt); accy_c=Output('accy_computed',lmt*vel.last())
def amb(Ty,v,Fy,d,r1,m,Kd,Cv,Cr,Iw): return ((1./m)*(2*(Ty/r1)-Kd*v**2-Cv*v-Fy*torch.sin(d))-Cr*9.81)/(1.+(2./m)*(2*(Iw/r1**2.)))
def ap(T): return torch.mul(T,torch.gt(T,0))
def an(T): return torch.mul(T,~torch.gt(T,0))
def lcl(ay,ay0,k1,k2): return k1+k2*(ay-ay0)
bv=0.0001
tpos=Linear(Tanh(Fir(output_dimension=n,b=True,b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},dropout=0.05)(ParamFun(ap)(crt.sw(cw)))))
tneg=Linear(Tanh(Fir(output_dimension=n,b=True,b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},dropout=0.05)(ParamFun(an)(crt.sw(cw)))))
r1p=Parameter('r1',values=[[5e-3]]); mp=Parameter('mass',values=[[3.788]])
Kdp=Parameter('Kd',values=[[1e-4]]); Cvp=Parameter('Cv',values=[[1e-4]]); Crp=Parameter('Cr',values=[[1e-4]]); Iw1p=Parameter('Iw1',values=[[1e-4]])
acl=[Constant('ay_center_'+str(i),values=ca[i]) for i in range(na)]
lc=ParamFun(lcl)
def lcg(idx):
    def f(ay):
        i=idx[0]
        return lc(ay,acl[i],Parameter('k1_ay_%d'%i,values=[[1e-5]]),Parameter('k2_ay_%d'%i,values=[[1e-5]]))
    return f
flv=Fuzzify(centers=cl,functions='Triangular')(vel.sw(cw))
fla=Fuzzify(centers=ca,functions='Triangular')(accy.sw(cw))
lmls=LocalModel(input_function=lcg,pass_indexes=True)((accy.sw(cw),),(fla,))
Fy=LocalModel(output_function=lambda:Fir(W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}))(lmls,flv)
mbo=ParamFun(amb,parameters_and_constants=[r1p,mp,Kdp,Cvp,Crp,Iw1p])(tpos+tneg,vel.last(),Fy,delta.last())
acc_out=Output('acceleration',mbo)

dyn=nnodely(visualizer='Standard',seed=42,workspace=P_PATH,save_history=False)
dyn.addModel('yaw_rate',[yr_out,accy_c]); dyn.addMinimize('yaw_rate_error',yr.next(),yr_out,loss_function='rmse')
dyn.addModel('acc',[acc_out]); dyn.addMinimize('acc_error',acc.next(),acc_out,loss_function='rmse')
dyn.neuralizeModel()
dyn.loadData(name='training_set',source=ds_train); dyn.loadData(name='validation_set',source=ds_val); dyn.loadData(name='test_set',source=ds_test)

def ev(lbl):
    nn=len(ds_val['yaw_rate'])-nsp-5
    s=dyn.getSamples('validation_set',index=0,window=nn); o=dyn(s,sampled=True)
    py=np.asarray(o['yaw_rate_']).flatten(); pa=np.asarray(o['acceleration']).flatten(); nu=len(py)
    # withax: acc.sw() used in lateral → use ds_val directly for ground truth
    yr=float(np.sqrt(np.mean((py-np.array(ds_val['yaw_rate'][nsp:nsp+nu]))**2)))
    ax=float(np.sqrt(np.mean((pa-np.array(ds_val['acc'][nsp:nsp+nu]))**2)))
    print(f"  [{lbl}] yr={yr:.4f}({yr/persist_yr:.2f}×p) ax={ax:.4f}({ax/persist_ax:.2f}×p)",flush=True)
    return yr,ax

for ep,pat,model,lr,name in [(1500,300,'yaw_rate',1e-3,'P1'),(3000,500,'acc',1e-3,'P2')]:
    print(f"\n{name} ({ep}ep p={pat})",flush=True)
    dyn.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',test_dataset='test_set',
        training_params={'num_of_epochs':ep,'val_batch_size':100,'train_batch_size':100,'lr':lr,'models':model},
        optimizer='Adam',shuffle_data=True,early_stopping=earlystopping.early_stop_patience,
        early_stopping_params={'patience':pat,'error':f'{model}_error'})
    dyn.saveModel(); dyn.neuralizeModel()
yr2,ax2=ev('P2-Val(same-day)'); shutil.copy(os.path.join(P_PATH,'net.json'),os.path.join(P2_BK,'net.json'))

# A1's actual target metric: CROSS-DAY (Day1 test_set) error -- this is
# what "wider margins help extrapolation" is supposed to improve. No Phase3
# here (kept as a single-variable test; A2 covers Phase3 separately).
def ev_test(lbl):
    nn=len(ds_test['yaw_rate'])-nsp-5
    s=dyn.getSamples('test_set',index=0,window=nn); o=dyn(s,sampled=True)
    py=np.asarray(o['yaw_rate_']).flatten(); pa=np.asarray(o['acceleration']).flatten(); nu=len(py)
    yr_rmse=float(np.sqrt(np.mean((py-np.array(ds_test['yaw_rate'][nsp:nsp+nu]))**2)))
    ax_rmse=float(np.sqrt(np.mean((pa-np.array(ds_test['acc'][nsp:nsp+nu]))**2)))
    print(f"  [{lbl}] yr={yr_rmse:.4f} ax={ax_rmse:.4f}  (n={nu})",flush=True)
    return yr_rmse,ax_rmse

yr2_crossday, ax2_crossday = ev_test('P2-Test(CROSS-DAY, Day1) -- the actual A1 target metric')

print(f"\n[FINAL A1] wide-margin withax:", flush=True)
print(f"  same-day (Val)  : yr={yr2:.4f}  (exp9 original same-day was 0.1158)", flush=True)
print(f"  cross-day (Day1): yr={yr2_crossday:.4f}  (exp9's own TECHNICAL_NOTES.md-documented cross-day was 0.1992 for exp6c noax; "
      f"exp9 withax's own cross-day number was not separately documented, so this is the first direct same-architecture comparison)", flush=True)
with open(OUT_JSON,'w') as f:
    json.dump({'exp':'a1_widemargin','model':'withax','margin_vx':1.0,'margin_delta':0.3,'margin_ax':1.0,
               'yr_sameday':yr2,'ax_sameday':ax2,'yr_crossday':yr2_crossday,'ax_crossday':ax2_crossday,
               'persist_yr':persist_yr,'persist_ax':persist_ax},f,indent=2)
print(f"Saved: {OUT_JSON}",flush=True)
