"""
run_deep_dynamics_car02_v8.py — DD v8: 6-Parameter + K=10 (kombiniertes Experiment)

Zwei unabhängige Verbesserungen gleichzeitig auf die DD-6-Parameter-Variante
(Cf=Cr=1.3, Ef=Er=0 fest) angewendet:
  1. K=10 Rollout (K=5→K=10 brachte bei v2 +19% Verbesserung)
  2. Weniger Parameter (10→6) + 3-Kanal-Ausgabe

DD v2  K=5  (10-param, 3ch): 0.7329
DD v2  K=10 (10-param, 3ch): 0.5950  ← Effekt von K=10
DD v4  K=5  (6-param,  5ch): 1.3295  ← 6-param aber 5ch und K=5
DD v7  K=5  (5-param,  3ch, fixed Iz): 0.8223 ← Iz=0.05 geschätzt, Abweichung

Dieses Experiment:
  6-Parameter (Bf,Df,Br,Dr,Frx + Iz lernbar, Bereich [0.03,0.08])
  K=10 Rollout
  3-Kanal-Ausgabe [v_x, v_y, ψ̇]
Erwartung: Iz konvergiert zu einem plausiblen Wert, K=10 bringt einen
Vorteil bei der Verteilungsanpassung, nahe an v2-K10 oder besser.
"""
import os, sys, pickle, yaml, copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

import types
_ws = types.ModuleType("wandb")
class _R:
    name="local_run"; summary={}
    def log(self,d): pass
    def finish(self): pass
_ws.run=_R(); _ws.init=lambda **kw: None
sys.modules["wandb"]=_ws

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_dynamics_model_car02 import DeepDynamicsCar02, PhysicsGuard, device

PROC       = "processed_data"
OUTPUT_DIR = "trained_models/deep_dynamics_car02_v8"
K          = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

V8_PARAMS = [
    {"name": "Bf",  "min": 0.1,  "max": 10.0},
    {"name": "Df",  "min": 0.1,  "max":  2.0},
    {"name": "Br",  "min": 0.1,  "max": 10.0},
    {"name": "Dr",  "min": 0.1,  "max":  2.0},
    {"name": "Frx", "min": -50.0,"max": 50.0},
    {"name": "Iz",  "min":  0.03, "max":  0.08},  # learned in physically valid range
]
CF_CR = 1.3; EF_ER = 0.0

TRAIN_FILES = [
    f"{PROC}/session_20260722_113922_seg1_20ms.csv",
    f"{PROC}/session_20260722_113922_seg2_20ms.csv",
    f"{PROC}/session_20260722_113922_seg3_20ms.csv",
    f"{PROC}/session_20260722_115055_seg1_20ms.csv",
    f"{PROC}/session_20260722_120608_seg1_20ms.csv",
    f"{PROC}/session_20260722_120608_seg2_20ms.csv",
    f"{PROC}/session_20260722_120608_seg3_20ms.csv",
    f"{PROC}/session_20260722_113922_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260722_113922_seg2_20ms_mirror.csv",
    f"{PROC}/session_20260722_113922_seg3_20ms_mirror.csv",
    f"{PROC}/session_20260722_115055_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260722_120608_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260722_120608_seg2_20ms_mirror.csv",
    f"{PROC}/session_20260722_120608_seg3_20ms_mirror.csv",
    f"{PROC}/session_20260728_122000_seg1_20ms.csv",
    f"{PROC}/session_20260728_122000_seg2_20ms.csv",
    f"{PROC}/session_20260728_122813_seg1_20ms.csv",
    f"{PROC}/session_20260728_122813_seg2_20ms.csv",
    f"{PROC}/session_20260728_123200_seg1_20ms.csv",
    f"{PROC}/session_20260728_123412_seg1_20ms.csv",
    f"{PROC}/session_20260728_123713_seg1_20ms.csv",
    f"{PROC}/session_20260728_123713_seg2_20ms.csv",
    f"{PROC}/session_20260728_123955_seg1_20ms.csv",
    f"{PROC}/session_20260728_124427_seg1_20ms.csv",
    f"{PROC}/session_20260728_163324_seg1_20ms.csv",
    f"{PROC}/session_20260728_170012_seg1_20ms.csv",
    f"{PROC}/session_20260728_170012_seg2_20ms.csv",
    f"{PROC}/session_20260728_122000_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_122000_seg2_20ms_mirror.csv",
    f"{PROC}/session_20260728_122813_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_122813_seg2_20ms_mirror.csv",
    f"{PROC}/session_20260728_123200_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_123412_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_123713_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_123713_seg2_20ms_mirror.csv",
    f"{PROC}/session_20260728_123955_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_124427_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_163324_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_170012_seg1_20ms_mirror.csv",
    f"{PROC}/session_20260728_170012_seg2_20ms_mirror.csv",
]
VAL_FILES = [f"{PROC}/session_20260722_115055_seg2_20ms.csv"]
TEST_FILE  =  f"{PROC}/session_20260721_20ms.csv"


class DeepDynamicsCar02_v8(DeepDynamicsCar02):
    def __init__(self, cfg):
        cfg2 = copy.deepcopy(cfg); cfg2["parameters"] = V8_PARAMS
        super().__init__(cfg2)

    def _bicycle_step(self, state_t, p):
        v_x=state_t[:,0]; v_y=state_t[:,1]; psi=state_t[:,2]; st=state_t[:,4]
        vxs = torch.abs(v_x) + self.v0
        af = st - torch.atan2(v_y + self.lf*psi, vxs)
        ar =      torch.atan2(-v_y + self.lr*psi, vxs)
        g=9.81
        Fzf = self.mass*g*self.lr/(self.lf+self.lr)
        Fzr = self.mass*g*self.lf/(self.lf+self.lr)
        Fyf = p["Df"]*Fzf*torch.sin(CF_CR*torch.atan(
            p["Bf"]*af - EF_ER*(p["Bf"]*af-torch.atan(p["Bf"]*af))))
        Fyr = p["Dr"]*Fzr*torch.sin(CF_CR*torch.atan(
            p["Br"]*ar - EF_ER*(p["Br"]*ar-torch.atan(p["Br"]*ar))))
        Frx=p["Frx"]; Iz=p["Iz"]
        dvx=(Frx-Fyf*torch.sin(st))/self.mass+v_y*psi
        dvy=(Fyr+Fyf*torch.cos(st))/self.mass-v_x*psi
        dps=(self.lf*Fyf*torch.cos(st)-self.lr*Fyr)/Iz
        return torch.stack([v_x+dvx*self.dt, v_y+dvy*self.dt, psi+dps*self.dt], 1)


class DDDatasetV8(torch.utils.data.Dataset):
    COLS_IN    = ["v_x_local","v_y_local","mocap_angular_velocity","speed_cmd","steering_cmd"]
    COLS_STATE = ["v_x_local","v_y_local","mocap_angular_velocity"]
    COLS_CMD   = ["speed_cmd","steering_cmd"]
    def __init__(self, files, H, K, scaler=None):
        xs,fs,fc=[],[],[]
        for f in files:
            if not os.path.exists(f): continue
            df=pd.read_csv(f,index_col=0)
            x=df[self.COLS_IN].values.astype(np.float32)
            s=df[self.COLS_STATE].values.astype(np.float32)
            c=df[self.COLS_CMD].values.astype(np.float32)
            m=np.abs(x[:,0])>0.3; x,s,c=x[m],s[m],c[m]
            for t in range(H,len(x)-K-1):
                xs.append(x[t-H:t]); fs.append(s[t:t+K]); fc.append(c[t:t+K])
        self.xr=np.stack(xs); self.fs=np.stack(fs); self.fc=np.stack(fc)
        flat=self.xr.reshape(-1,self.xr.shape[-1])
        if scaler is None:
            self.sc=StandardScaler(); nf=self.sc.fit_transform(flat)
        else:
            self.sc=scaler; nf=scaler.transform(flat)
        self.xn=nf.reshape(self.xr.shape).astype(np.float32)
    def __len__(self): return len(self.xr)
    def __getitem__(self,i):
        return (torch.from_numpy(self.xr[i]),torch.from_numpy(self.xn[i]),
                torch.from_numpy(self.fs[i]),torch.from_numpy(self.fc[i]))


def rollout(model,xr,xn,fs,fc,sm,ss,crit):
    tot=0.0; cr=xr.clone(); cn=xn.clone()
    for k in range(fs.shape[1]):
        p=model(cr,cn); tot+=crit(p,fs[:,k])
        nr=torch.cat([p,fc[:,k]],1)
        cr=torch.cat([cr[:,1:],nr.detach().unsqueeze(1)],1)
        cn=(cr-sm)/ss
    return tot/fs.shape[1]


def eval_cl(model,csv,sc,H):
    df=pd.read_csv(csv,index_col=0)
    COLS_IN=["v_x_local","v_y_local","mocap_angular_velocity","speed_cmd","steering_cmd"]
    COLS_EV=["v_x_local","v_y_local","mocap_angular_velocity"]
    xa=df[COLS_IN].values.astype(np.float32)
    ya=df[COLS_EV].values.astype(np.float32)
    xn=sc.transform(xa).astype(np.float32)
    cr=torch.from_numpy(xa[:H]).unsqueeze(0).to(device)
    cn=torch.from_numpy(xn[:H]).unsqueeze(0).to(device)
    preds=[]; model.eval()
    with torch.no_grad():
        for t in range(len(xa)-H):
            p=model(cr,cn); preds.append(p.squeeze(0).cpu().numpy())
            nr=torch.cat([p.squeeze(0),torch.from_numpy(xa[H+t,3:]).to(device)]).unsqueeze(0).unsqueeze(0)
            cr=torch.cat([cr[:,1:],nr],1)
            cn=(cr-torch.tensor(sc.mean_,device=device,dtype=torch.float32))/torch.tensor(sc.scale_,device=device,dtype=torch.float32)
    rmse=np.sqrt(np.mean((np.array(preds)-ya[H:])**2,axis=0))
    for n,r in zip(COLS_EV,rmse): print(f"    {n:<35} {r:.4f}")
    m=float(rmse.mean())
    print(f"\n  DD v8 CL-RMSE(3ch) = {m:.4f}")
    print(f"  DD v2-K10 (3ch)    = 0.5950  (reference)")
    print(f"  GRU Run30 (3ch)    = 0.4448  (GRU best)")
    return m


if __name__ == "__main__":
    cfg=yaml.safe_load(open(os.path.join(os.path.dirname(__file__),"deep_dynamics_car02.yaml")))
    H=cfg["model"]["horizon"]; BS=cfg["model"]["batch_size"]
    lr=cfg["model"]["lr"]; pat=cfg["model"]["patience"]
    print("="*65)
    print(f"  DD v8: 6-param (Bf,Df,Br,Dr,Frx,Iz∈[0.03,0.08]) + K={K}")
    print(f"  Iz learned (not fixed), Cf=Cr={CF_CR}, Ef=Er={EF_ER}")
    print(f"  DD v2-K10: 0.5950 | DD v2-K5: 0.7329"); print("="*65)

    tr_ds=DDDatasetV8(TRAIN_FILES,H,K)
    va_ds=DDDatasetV8(VAL_FILES,H,K,scaler=tr_ds.sc)
    tr_dl=torch.utils.data.DataLoader(tr_ds,BS,shuffle=True,drop_last=True,num_workers=2)
    va_dl=torch.utils.data.DataLoader(va_ds,BS,shuffle=False,num_workers=2)
    print(f"  Train:{len(tr_ds)}  Val:{len(va_ds)}")

    model=DeepDynamicsCar02_v8(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=10)
    crit=nn.HuberLoss()
    sm=torch.tensor(tr_ds.sc.mean_,device=device,dtype=torch.float32)
    ss=torch.tensor(tr_ds.sc.scale_,device=device,dtype=torch.float32)
    best=float("inf"); ni=0; best_p=os.path.join(OUTPUT_DIR,"best.pth")

    for ep in range(1,301):
        model.train(); tl=0
        for xr,xn,fs,fc in tr_dl:
            xr=xr.to(device); xn=xn.to(device); fs=fs.to(device); fc=fc.to(device)
            opt.zero_grad(); loss=rollout(model,xr,xn,fs,fc,sm,ss,crit)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
        tl/=len(tr_dl)
        model.eval(); vl=0
        with torch.no_grad():
            for xr,xn,fs,fc in va_dl:
                xr=xr.to(device); xn=xn.to(device); fs=fs.to(device); fc=fc.to(device)
                vl+=rollout(model,xr,xn,fs,fc,sm,ss,crit).item()
        vl/=len(va_dl)
        sch.step(vl)
        print(f"Epoch {ep:3d}/300  tr={tl:.6f}  va={vl:.6f}",end="")
        if vl<best: best=vl; ni=0; torch.save(model.state_dict(),best_p); print("  ✓")
        else:
            ni+=1; print()
            if ni>=pat: print(f"Early stop {ep}."); break

    pickle.dump(tr_ds.sc,open(os.path.join(OUTPUT_DIR,"scaler.pkl"),"wb"))
    model.load_state_dict(torch.load(best_p,map_location=device,weights_only=False))
    eval_cl(model,TEST_FILE,tr_ds.sc,H)
