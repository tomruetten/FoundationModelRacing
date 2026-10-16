"""
C3: Day2+Day3 + Mirror-Augmentierung, noax, Phase3 lr=1e-4.
Mirror: δ→-δ, ψ̇→-ψ̇, accy→-accy (die _mirror.csv-Dateien existieren bereits).
Trainingsset: 163324 + 163324_mirror + 113922 + 113922_mirror ≈ 16906 Zeilen.
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
P_PATH   = os.path.join(WORK, 'trained_models', 'msnn_exp11_mirror_combined')
P2_BK    = os.path.join(WORK, 'trained_models', 'msnn_exp11_phase2_backup')
OUT_JSON = '/tmp/exp11_mirror_combined_results.json'
os.makedirs(P_PATH,exist_ok=True); os.makedirs(P2_BK,exist_ok=True)

def load_clean(name, skip=0):
    df=pd.read_csv(os.path.join(DATA_DIR,name),index_col=0)
    df=df.dropna(subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y'])
    return df.iloc[skip:].reset_index(drop=True)

df_d3=load_clean('session_20260728_163324_seg1_20ms.csv').iloc[:6500].reset_index(drop=True)
d3_tr=int(len(df_d3)*2/3)
df_val=df_d3.iloc[d3_tr:].reset_index(drop=True)
df_d3_train=df_d3.iloc[:d3_tr].reset_index(drop=True)
# mirrors (already cropped to same length inside the file)
df_d3_m=load_clean('session_20260728_163324_seg1_20ms_mirror.csv').iloc[:d3_tr].reset_index(drop=True)
df_d2=load_clean('session_20260722_113922_seg1_20ms.csv',skip=1000)
df_d2_m=load_clean('session_20260722_113922_seg1_20ms_mirror.csv',skip=1000)

df_train=pd.concat([df_d3_train,df_d3_m,df_d2,df_d2_m],ignore_index=True)
df_test=pd.read_csv(os.path.join(DATA_DIR,'session_20260721_20ms.csv'),index_col=0)

print(f"[EXP11] Mirror augmented: 163324({len(df_d3_train)})+mirror({len(df_d3_m)})+113922({len(df_d2)})+mirror({len(df_d2_m)}) = {len(df_train)} rows",flush=True)

def to_dict(df):
    return {k:df[c].values.astype(np.float64) for k,c in [('yaw_rate','IMU_angular_velocity_z'),
        ('vel','odom_speed'),('delta','steering_cmd'),('acc','IMU_linear_acceleration_x'),
        ('accy','IMU_linear_acceleration_y'),('crt','current_motor')]}

ds_train=to_dict(df_train); ds_val=to_dict(df_val); ds_test=to_dict(df_test)
all_vx=df_train['odom_speed'].values; all_delta=df_train['steering_cmd'].values
all_accy=df_train['IMU_linear_acceleration_y'].values
persist_yr=float(np.sqrt(np.mean(np.diff(ds_val['yaw_rate'])**2)))
persist_ax=float(np.sqrt(np.mean(np.diff(ds_val['acc'])**2)))
print(f"Persistence: yr={persist_yr:.4f} ax={persist_ax:.4f}",flush=True)

n=10; nsp=20; cw=15; nv=2; nd=5; na=4; nlv=2
mv,Mv=float(np.nanmin(all_vx)),float(np.nanmax(all_vx))
md,Md=float(np.nanmin(all_delta)),float(np.nanmax(all_delta))
may,May=float(np.nanmin(all_accy)),float(np.nanmax(all_accy))
cv=list(np.linspace(mv-0.3,Mv+0.3,nv)); cd=list(np.linspace(md-0.1,Md+0.1,nd))
ca=list(np.linspace(may,May,na)); cl=list(np.linspace(mv,Mv,nlv))

yr=Input('yaw_rate'); vel=Input('vel'); delta=Input('delta')
acc=Input('acc'); accy=Input('accy'); crt=Input('crt')
sc=[Constant('sc_'+str(i),values=cd[i]) for i in range(nd)]
vc=[Constant('vc_'+str(i),values=cv[i]) for i in range(nv)]
def ucl(inp,vx,i0,v0,k1,k2,k3): return vx*(k1+k2*(inp-i0)+k3*(vx-v0))
uc=ParamFun(ucl)
def ucg(idx):
    def f(inp,vx):
        s,sp=idx[1],idx[0]
        return uc(inp,vx,sc[s],vc[sp],Parameter('k1_%d_%d'%(sp,s),values=[[1e-5]]),
                  Parameter('k2_%d_%d'%(sp,s),values=[[1e-5]]),Parameter('k3_%d_%d'%(sp,s),values=[[1e-5]]))
    return f
f1=Fuzzify(centers=cv,functions='Triangular')(vel.sw(nsp)); f2=Fuzzify(centers=cd,functions='Triangular')(delta.sw(nsp))
lms=LocalModel(input_function=ucg,pass_indexes=True)((delta.sw(nsp),vel.sw(nsp)),(f1,f2))
ft=Fuzzify(centers=cv,functions='Triangular')(vel.sw(nsp))
lmt=LocalModel(output_function=lambda:Fir(W_init=init_exp,W_init_params={'size_index':0,'max_value':1,'lambda':5,'monotonicity':'increasing'}))(lms,ft)
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
flv=Fuzzify(centers=cl,functions='Triangular')(vel.sw(cw)); fla=Fuzzify(centers=ca,functions='Triangular')(accy.sw(cw))
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
    yr_=float(np.sqrt(np.mean((py-np.array(ds_val['yaw_rate'][nsp:nsp+nu]))**2)))
    ax_=float(np.sqrt(np.mean((pa-np.array(ds_val['acc'][nsp:nsp+nu]))**2)))
    print(f"  [{lbl}] yr={yr_:.4f}({yr_/persist_yr:.2f}×p) ax={ax_:.4f}({ax_/persist_ax:.2f}×p)",flush=True)
    return yr_,ax_

for ep,pat,model,lr,name in [(1500,300,'yaw_rate',1e-3,'P1'),(3000,500,'acc',1e-3,'P2')]:
    print(f"\n{name}",flush=True)
    dyn.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',test_dataset='test_set',
        training_params={'num_of_epochs':ep,'val_batch_size':100,'train_batch_size':100,'lr':lr,'models':model},
        optimizer='Adam',shuffle_data=True,early_stopping=earlystopping.early_stop_patience,
        early_stopping_params={'patience':pat,'error':f'{model}_error'})
    dyn.saveModel(); dyn.neuralizeModel()
yr2,ax2=ev('P2'); shutil.copy(os.path.join(P_PATH,'net.json'),os.path.join(P2_BK,'net.json'))

print(f"\nPhase3 lr=1e-4",flush=True)
dyn.addConnect(accy_c,accy); dyn.addClosedLoop(acc_out,accy); dyn.neuralizeModel()
dyn.trainAndAnalyze(train_dataset='training_set',validation_dataset='validation_set',test_dataset='test_set',
    training_params={'num_of_epochs':1000,'val_batch_size':100,'train_batch_size':100,'lr':1e-4,
                     'connect':{'accy':'accy_computed'},'closed_loop':{'acc':'acceleration'}},
    optimizer='Adam',shuffle_data=True,early_stopping=earlystopping.early_stop_patience,
    early_stopping_params={'patience':200,'error':'yaw_rate_error'})
dyn.saveModel(); dyn.neuralizeModel(); yr3,ax3=ev('P3')
print(f"\n[FINAL exp11] Mirror combined yr P2={yr2:.4f} P3={yr3:.4f} ({((yr3-yr2)/yr2*100):+.2f}%)",flush=True)
with open(OUT_JSON,'w') as f:
    json.dump({'exp':'11_mirror_combined','train_rows':len(df_train),'model':'noax','p3_lr':1e-4,
               'yr2':yr2,'ax2':ax2,'yr3':yr3,'ax3':ax3,'persist_yr':persist_yr,'persist_ax':persist_ax,
               'yr_change_pct':round((yr3-yr2)/yr2*100,2)},f,indent=2)
print(f"Saved: {OUT_JSON}",flush=True)
