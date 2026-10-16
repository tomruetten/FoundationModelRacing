# diag_eval.py — reproduziert die ev()-Funktion aus dem exp9-Training,
# um zu prüfen, ob net.json korrekt geladen wird.
#
# BUGFIX 2026-08-13 (gegenüber meiner ursprünglichen Version):
#   1. DATA_DIR auf einen tatsächlich existierenden Pfad geändert (der alte
#      Pfad src/ma_jonathanmohr-main/processed_data existiert nicht — den
#      echten Pfad siehe DATA_DIR in exp9_withax_combined.py).
#   2. Vor dyn(s, sampled=True) muss neuralizeModel() noch einmal aufgerufen
#      werden — im nnodely-Quellcode bestätigt: loadModel() setzt am Ende
#      bedingungslos self._neuralized=False, und __call__() prüft zu Beginn
#      check(self.neuralized, RuntimeError, "not neuralized"). Das Original-
#      skript würde hier direkt abstürzen, statt eine falsche, aber
#      lauffähige Zahl wie 0.408 zu liefern.
import os, sys
import numpy as np
import pandas as pd
from nnodely.basic.relation import NeuObj

WORK      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(WORK, 'Code von Felix', 'ma_jonathanmohr-main', 'processed_data')
MODEL_DIR = os.path.join(WORK, 'trained_models', 'msnn_exp9_phase2_backup')

NeuObj.names.clear(); NeuObj.names.append('SampleTime')

import torch
from nnodely import *
from nnodely.support import earlystopping
from nnodely.support.initializer import *

# ── Daten (identisch zu exp9) ─────────────────────────────────────
df_d3 = pd.read_csv(os.path.join(DATA_DIR,
    'session_20260728_163324_seg1_20ms.csv'), index_col=0)
df_d3 = df_d3.dropna(
    subset=['IMU_linear_acceleration_x','IMU_linear_acceleration_y']
).iloc[:6500].reset_index(drop=True)
d3_tr  = int(len(df_d3)*2/3)
df_val = df_d3.iloc[d3_tr:].reset_index(drop=True)

def to_dict(df):
    return {k: df[c].values.astype(np.float64) for k,c in [
        ('yaw_rate','IMU_angular_velocity_z'),
        ('vel','odom_speed'), ('delta','steering_cmd'),
        ('acc','IMU_linear_acceleration_x'),
        ('accy','IMU_linear_acceleration_y'),
        ('crt','current_motor')]}

ds_val    = to_dict(df_val)
persist_yr = float(np.sqrt(np.mean(np.diff(ds_val['yaw_rate'])**2)))
print(f"Val Zeilen: {len(df_val)}  persist_yr: {persist_yr:.4f}")

# ── Modell neu aufbauen (identisch zu exp9; loadModel() verwirft diesen
#    handgebauten Graph komplett wieder — er dient nur dazu, dass addModel/
#    addMinimize ein Objekt zum Aufrufen haben) ─────────────────────
nsp=20; cw=15; nv=2; nd=5; nax=2; na=4; nlv=2; n=10
all_vx    = df_d3.iloc[:d3_tr]['odom_speed'].values
all_delta = df_d3.iloc[:d3_tr]['steering_cmd'].values
all_accy  = df_d3.iloc[:d3_tr]['IMU_linear_acceleration_y'].values
all_ax    = df_d3.iloc[:d3_tr]['IMU_linear_acceleration_x'].values

mv,Mv   = float(np.nanmin(all_vx)),   float(np.nanmax(all_vx))
md,Md   = float(np.nanmin(all_delta)),float(np.nanmax(all_delta))
mxa,Mxa = float(np.nanmin(all_ax)),   float(np.nanmax(all_ax))
may,May = float(np.nanmin(all_accy)), float(np.nanmax(all_accy))
cv  = list(np.linspace(mv-0.3,  Mv+0.3,  nv))
cd  = list(np.linspace(md-0.1,  Md+0.1,  nd))
cxa = list(np.linspace(mxa-0.3, Mxa+0.3, nax))
ca  = list(np.linspace(may, May, na))
cl  = list(np.linspace(mv, Mv, nlv))

yr=Input('yaw_rate'); vel=Input('vel'); delta=Input('delta')
acc=Input('acc');     accy=Input('accy'); crt=Input('crt')
sc  = [Constant('steer_center_'+str(i), values=cd[i])  for i in range(nd)]
vc  = [Constant('vx_center_'+str(i),   values=cv[i])   for i in range(nv)]
axc = [Constant('ax_center_'+str(i),   values=cxa[i])  for i in range(nax)]

def ucl(inp,vx,ax,i0,v0,a0,k1,k2,k3,k4):
    return vx*(k1+k2*(inp-i0)+k3*(vx-v0)+k4*(ax-a0))
uc = ParamFun(ucl)
def ucg(idx):
    def f(inp,vx,ax):
        i_sp,i_st,i_ax = idx
        return uc(inp,vx,ax,sc[i_st],vc[i_sp],axc[i_ax],
            Parameter('lat_k1_%d_%d_%d'%(*idx,), values=[[1e-5]]),
            Parameter('lat_k2_%d_%d_%d'%(*idx,), values=[[1e-5]]),
            Parameter('lat_k3_%d_%d_%d'%(*idx,), values=[[1e-5]]),
            Parameter('lat_k4_%d_%d_%d'%(*idx,), values=[[1e-5]]))
    return f

f1=Fuzzify(centers=cv, functions='Triangular')(vel.sw(nsp))
f2=Fuzzify(centers=cd, functions='Triangular')(delta.sw(nsp))
f3=Fuzzify(centers=cxa,functions='Triangular')(acc.sw(nsp))
lms=LocalModel(input_function=ucg,pass_indexes=True)(
    (delta.sw(nsp),vel.sw(nsp),acc.sw(nsp)),(f1,f2,f3))
ft =Fuzzify(centers=cv, functions='Triangular')(vel.sw(nsp))
fat=Fuzzify(centers=cxa,functions='Triangular')(acc.sw(nsp))
lmt=LocalModel(output_function=lambda:Fir(
    W_init=init_negexp,
    W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}
))(lms,(ft,fat))
yr_out  = Output('yaw_rate_', lmt)
accy_c  = Output('accy_computed', lmt*vel.last())

def amb(Ty,v,Fy,d,r1,m,Kd,Cv,Cr,Iw):
    return ((1./m)*(2*(Ty/r1)-Kd*v**2-Cv*v-Fy*torch.sin(d))-Cr*9.81)/(1.+(2./m)*(2*(Iw/r1**2.)))
def ap(T): return torch.mul(T,torch.gt(T,0))
def an(T): return torch.mul(T,~torch.gt(T,0))
def lcl(ay,ay0,k1,k2): return k1+k2*(ay-ay0)
bv=0.0001
tpos=Linear(Tanh(Fir(output_dimension=n,b=True,
    b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},
    dropout=0.05)(ParamFun(ap)(crt.sw(cw)))))
tneg=Linear(Tanh(Fir(output_dimension=n,b=True,
    b_init=init_negexp,b_init_params={'size_index':0,'first_value':bv,'lambda':3},
    W_init=init_negexp,W_init_params={'size_index':0,'first_value':0.0001,'lambda':3},
    dropout=0.05)(ParamFun(an)(crt.sw(cw)))))
r1p=Parameter('r1',values=[[5e-3]]); mp=Parameter('mass',values=[[3.788]])
Kdp=Parameter('Kd',values=[[1e-4]]); Cvp=Parameter('Cv',values=[[1e-4]])
Crp=Parameter('Cr',values=[[1e-4]]); Iw1p=Parameter('Iw1',values=[[1e-4]])
acl=[Constant('ay_center_'+str(i),values=ca[i]) for i in range(na)]
lc=ParamFun(lcl)
def lcg(idx):
    def f(ay):
        i=idx[0]
        return lc(ay,acl[i],
            Parameter('k1_ay_%d'%i,values=[[1e-5]]),
            Parameter('k2_ay_%d'%i,values=[[1e-5]]))
    return f
flv=Fuzzify(centers=cl,functions='Triangular')(vel.sw(cw))
fla=Fuzzify(centers=ca,functions='Triangular')(accy.sw(cw))
lmls=LocalModel(input_function=lcg,pass_indexes=True)((accy.sw(cw),),(fla,))
Fy=LocalModel(output_function=lambda:Fir(
    W_init=init_negexp,
    W_init_params={'size_index':0,'first_value':0.0001,'lambda':5,'monotonicity':'increasing'}
))(lmls,flv)
mbo=ParamFun(amb,parameters_and_constants=[r1p,mp,Kdp,Cvp,Crp,Iw1p])(
    tpos+tneg,vel.last(),Fy,delta.last())
acc_out=Output('acceleration',mbo)

dyn=nnodely(visualizer='Standard', seed=42, workspace=MODEL_DIR, save_history=False)
dyn.addModel('yaw_rate',[yr_out,accy_c])
dyn.addMinimize('yaw_rate_error',yr.next(),yr_out,loss_function='rmse')
dyn.addModel('acc',[acc_out])
dyn.addMinimize('acc_error',acc.next(),acc_out,loss_function='rmse')

# Kritische Reihenfolge: erst neuralize, dann loadModel, danach nochmal
# neuralize (loadModel() setzt self._neuralized auf False; ohne erneutes
# neuralize vor dyn(...) wirft es RuntimeError: The network is not
# neuralized.)
dyn.neuralizeModel()
dyn.loadData(name='val_diag', source=ds_val)
# BUGFIX: loadModel(name, model_folder) -- das erste Positionsargument ist
# der Dateiname (Standard 'net'), nicht der Ordner. dyn.loadModel(MODEL_DIR)
# würde den kompletten absoluten Pfad als name behandeln und dann nach
# "<MODEL_DIR>.json" suchen (existiert nicht). workspace=MODEL_DIR ist
# bereits im Konstruktor gesetzt, ein nackter Aufruf loadModel() sucht also
# net.json unter workspace — genau die Datei, die wir wollen.
dyn.loadModel()             # lädt die trainierten Gewichte (verwirft den oben handgebauten Graph)
dyn.neuralizeModel()        # BUGFIX: baut den torch-Graph mit dem geladenen model_def neu auf

# ── Evaluation identisch zu exp9s ev() ────────────────────────────────
nn = len(ds_val['yaw_rate']) - nsp - 5
s  = dyn.getSamples('val_diag', index=0, window=nn)
o  = dyn(s, sampled=True)
py = np.asarray(o['yaw_rate_']).flatten()
pa = np.asarray(o['acceleration']).flatten()
nu = len(py)
yr_rmse = float(np.sqrt(np.mean((py - np.array(ds_val['yaw_rate'][nsp:nsp+nu]))**2)))
ax_rmse = float(np.sqrt(np.mean((pa - np.array(ds_val['acc'][nsp:nsp+nu]))**2)))
print(f"\n[diag] yr={yr_rmse:.4f} ({yr_rmse/persist_yr:.2f}×p)  ax={ax_rmse:.4f}")
print("→ Erwartet: yr≈0.1158 (1.05×p)")
print("→ Falls 0.1158 rauskommt: Alignment-Bug im step-by-step eval, Modell selbst ist OK")
print("→ Falls immer noch 0.408: Problem mit Datenpfad oder net.json-Laden")
