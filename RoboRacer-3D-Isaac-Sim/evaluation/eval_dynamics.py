"""Vehicle-dynamics model evaluation: GRU family, Deep Dynamics, MS-NN.

Datasets (all 20 ms CSVs from processed_data):
  day3_val_sameday    same day/track as training, racing section (rows 4333:6500)
  day2_val            Day 2 validation segment
  day1_test_crossday  Day 1, never used for training/selection
  newtrack_20260827   new track recorded 2026-08-27, validation split

Metrics per model x dataset:
  one-step open loop      predict t+1 from true history; RMSE per output channel,
                          plus persistence baseline (y[t+1] = y[t])
  multi-horizon rollout   autoregressive feedback of predicted state, true commands;
                          RMSE at 5/10/25/50/100 steps (0.1 .. 2 s)
  full rollout            warm-up 10 true steps, then autoregressive over the whole
                          segment (60 s / 125 s tiers), 3-channel and 5-channel RMSE
  pose drift              integrate (v_x, v_y, yaw rate) of prediction vs truth from
                          the same start: position error after 2 s and 5 s
  error vs speed          one-step yaw-rate RMSE in speed bins

All predictions are aligned to the NEXT sample (pred from window ending at t vs
truth at t+1).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import DATA_LOCAL, REPO, Report, env, load_module, log, rmse

DT = 0.02
HORIZONS = [5, 10, 25, 50, 100]
STATE3 = ["v_x_local", "v_y_local", "mocap_angular_velocity"]

DATASETS = [
    ("day3_val_sameday", "session_20260728_163324_seg1_20ms.csv", (4333, 6500), "Day 3, racing section (same day/track as training)"),
    ("day2_val", "session_20260722_115055_seg2_20ms.csv", None, "Day 2 validation segment"),
    ("day1_test_crossday", "session_20260721_20ms.csv", None, "Day 1, held-out test day"),
    ("newtrack_20260827", "session_20260827_143212_val_20ms.csv", None, "New track 2026-08-27, validation split"),
]
# MS-NN is scored against the IMU yaw rate; the Day-2 GRU validation file has a biased IMU gyro
# column (see data-sanity table), so MS-NN uses another Day-2 segment it was not trained on.
MSNN_CSV_OVERRIDE = {"day2_val": "session_20260722_115055_seg1_20ms.csv"}
IMU_BIAS_FLAG = 0.3  # rad/s


def imu_sanity(df: pd.DataFrame) -> dict:
    a = df["IMU_angular_velocity_z"]; b = df["mocap_angular_velocity"]; ok = a.notna() & b.notna()
    still = (df["odom_speed"].abs() < 0.05) & ok
    return dict(bias=float((a - b)[ok].mean()), rmse=float(np.sqrt(((a - b)[ok] ** 2).mean())),
                standing=float(a[still].abs().mean()) if still.sum() > 20 else float("nan"), n_still=int(still.sum()))


# ----------------------------------------------------------------------------- data
def load_dataset(csv: Path, rows):
    df = pd.read_csv(csv, index_col=0)
    df = df.dropna(subset=["IMU_linear_acceleration_x", "IMU_linear_acceleration_y"]).reset_index(drop=True)
    if rows is not None:
        df = df.iloc[rows[0]:rows[1]].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- models
class GRUNet(torch.nn.Module):
    """RNN_jomoh_1 family: GRU (1-2 layers, hidden 100/200) + fc1, fc3, fc5, LeakyReLU(0.1).
    Hidden state starts at zero for every call (as in training): feed the full window."""

    def __init__(self, in_sz, out_sz, hidden, layers):
        super().__init__()
        self.gru1 = torch.nn.GRU(in_sz, hidden, num_layers=layers, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, hidden)
        self.fc5 = torch.nn.Linear(hidden, out_sz)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        out, _ = self.gru1(x)
        z = self.act(self.fc1(out[:, -1]))
        z = self.act(self.fc3(z))
        return self.fc5(z)


class StateModel:
    """Common interface: windows of `history` rows of `input_cols` -> next row of `output_cols`."""
    name: str
    input_cols: list[str]
    output_cols: list[str]
    history: int

    def predict(self, windows: np.ndarray) -> np.ndarray:  # (B,H,in) -> (B,out)
        raise NotImplementedError


class GRUModel(StateModel):
    def __init__(self, name, path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        cfg = ck.get("config", {})
        hidden = sd["gru1.weight_hh_l0"].shape[1]
        layers = 1 + max(int(k.split("_l")[-1]) for k in sd if k.startswith("gru1.weight_ih_l"))
        self.input_cols = list(cfg["input_columns"])
        self.output_cols = list(cfg["output_columns"])
        self.history = int(cfg.get("history", 10))
        self.name = name
        self.device = device
        self.model = GRUNet(len(self.input_cols), len(self.output_cols), hidden, layers).to(device).eval()
        self.model.load_state_dict(sd)
        self.desc = f"GRU h={hidden} L={layers}, in={len(self.input_cols)} out={len(self.output_cols)}"

    @torch.no_grad()
    def predict(self, windows):
        out = []
        for i in range(0, len(windows), 4096):
            x = torch.from_numpy(windows[i:i + 4096]).float().to(self.device)
            out.append(self.model(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.output_cols)), np.float32)


class DeepDynamicsModel(StateModel):
    """Deep Dynamics v8 (Pacejka bicycle model with NN-estimated parameters)."""
    input_cols = ["v_x_local", "v_y_local", "mocap_angular_velocity", "speed_cmd", "steering_cmd"]
    output_cols = STATE3

    def __init__(self, name, ckpt, scaler_pkl, cfg_yaml, device):
        import copy
        import yaml
        base = load_module(REPO / "dynamics/training/09_deep_dynamics_model.py", "deep_dynamics_model_car02")
        cfg = yaml.safe_load(open(cfg_yaml))
        v8_params = [
            {"name": "Bf", "min": 0.1, "max": 10.0}, {"name": "Df", "min": 0.1, "max": 2.0},
            {"name": "Br", "min": 0.1, "max": 10.0}, {"name": "Dr", "min": 0.1, "max": 2.0},
            {"name": "Frx", "min": -50.0, "max": 50.0}, {"name": "Iz", "min": 0.03, "max": 0.08},
        ]
        CF_CR, EF_ER = 1.3, 0.0

        class V8(base.DeepDynamicsCar02):
            def __init__(self, cfg):
                c2 = copy.deepcopy(cfg); c2["parameters"] = v8_params
                super().__init__(c2)

            def _bicycle_step(self, s, p):
                v_x, v_y, psi, st = s[:, 0], s[:, 1], s[:, 2], s[:, 4]
                vxs = torch.abs(v_x) + self.v0
                af = st - torch.atan2(v_y + self.lf * psi, vxs)
                ar = torch.atan2(-v_y + self.lr * psi, vxs)
                g = 9.81
                Fzf = self.mass * g * self.lr / (self.lf + self.lr)
                Fzr = self.mass * g * self.lf / (self.lf + self.lr)
                Fyf = p["Df"] * Fzf * torch.sin(CF_CR * torch.atan(p["Bf"] * af - EF_ER * (p["Bf"] * af - torch.atan(p["Bf"] * af))))
                Fyr = p["Dr"] * Fzr * torch.sin(CF_CR * torch.atan(p["Br"] * ar - EF_ER * (p["Br"] * ar - torch.atan(p["Br"] * ar))))
                Frx, Iz = p["Frx"], p["Iz"]
                dvx = (Frx - Fyf * torch.sin(st)) / self.mass + v_y * psi
                dvy = (Fyr + Fyf * torch.cos(st)) / self.mass - v_x * psi
                dps = (self.lf * Fyf * torch.cos(st) - self.lr * Fyr) / Iz
                return torch.stack([v_x + dvx * self.dt, v_y + dvy * self.dt, psi + dps * self.dt], 1)

        self.name = name
        self.device = device
        self.history = int(cfg["model"]["horizon"])
        self.model = V8(cfg).to(device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        self.scaler = pickle.load(open(scaler_pkl, "rb"))
        self.desc = f"Deep Dynamics v8 (6 Pacejka params, GRU h={cfg['model']['gru_hidden']} L={cfg['model']['gru_layers']})"

    @torch.no_grad()
    def predict(self, windows):
        out = []
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)
        for i in range(0, len(windows), 2048):
            xr = torch.from_numpy(windows[i:i + 2048]).float().to(self.device)
            xn = (xr - mean) / scale
            out.append(self.model(xr, xn).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


# ----------------------------------------------------------------------------- windows
def make_windows(X: np.ndarray, H: int):
    """(T,in) -> windows (T-H, H, in); window i covers rows i..i+H-1 and predicts row i+H."""
    idx = np.arange(H)[None, :] + np.arange(len(X) - H)[:, None]
    return X[idx]


def one_step(model: StateModel, df: pd.DataFrame):
    X = df[model.input_cols].fillna(0.0).to_numpy(np.float32)
    Y = df[model.output_cols].fillna(0.0).to_numpy(np.float32)
    H = model.history
    pred = model.predict(make_windows(X, H))          # predicts rows H..T-1
    true = Y[H:]
    persist = Y[H - 1:-1]                             # y[t] as prediction for t+1
    return pred, true, persist


def rollout(model: StateModel, df: pd.DataFrame, starts: np.ndarray, H_max: int):
    """Autoregressive rollout from several starts in parallel.
    Returns pred (S,H_max,out), true (S,H_max,out); state inputs that the model
    predicts are fed back, everything else (commands, IMU-only inputs) is taken from truth."""
    X = df[model.input_cols].fillna(0.0).to_numpy(np.float32)
    Y = df[model.output_cols].fillna(0.0).to_numpy(np.float32)
    H = model.history
    fb = [(model.input_cols.index(c), j) for j, c in enumerate(model.output_cols) if c in model.input_cols]
    starts = starts[(starts >= H) & (starts + H_max <= len(X))]
    if len(starts) == 0:
        return None, None, starts
    win = np.stack([X[s - H:s] for s in starts])       # (S,H,in)
    preds = np.zeros((len(starts), H_max, len(model.output_cols)), np.float32)
    trues = np.zeros_like(preds)
    for h in range(H_max):
        p = model.predict(win)
        preds[:, h] = p
        trues[:, h] = Y[starts + h]
        nxt = X[starts + h].copy()                     # true row for step h (commands etc.)
        for ii, jo in fb:
            nxt[:, ii] = p[:, jo]
        win = np.concatenate([win[:, 1:], nxt[:, None, :]], axis=1)
    return preds, trues, starts


def integrate_pose(v_x, v_y, yr, dt=DT):
    """Body velocities -> planar trajectory. Arrays (S,H)."""
    psi = np.cumsum(yr * dt, axis=1)
    psi = np.concatenate([np.zeros((psi.shape[0], 1)), psi[:, :-1]], axis=1)
    x = np.cumsum((v_x * np.cos(psi) - v_y * np.sin(psi)) * dt, axis=1)
    y = np.cumsum((v_x * np.sin(psi) + v_y * np.cos(psi)) * dt, axis=1)
    return x, y


# ----------------------------------------------------------------------------- MS-NN
class MSNNModel:
    """Wraps dynamics/msnn/src/inference_msnn.MSNNInference (nnodely, per-step)."""

    def __init__(self, name, model_dir):
        src = REPO / "dynamics/msnn/src"
        sys.path.insert(0, str(src))
        try:
            from nnodely.basic.relation import NeuObj
            NeuObj.names.clear(); NeuObj.names.append("SampleTime")
        except Exception:
            pass
        mod = load_module(src / "inference_msnn.py", f"inference_msnn_{abs(hash(name))}")
        self.infer = mod.MSNNInference(model_dir=str(model_dir))
        self.name = name
        self.desc = "MS-NN (nnodely, physics-structured, yaw rate + a_x from IMU/VESC inputs)"
        self.warmup = 25

    def run(self, df: pd.DataFrame):
        self.infer.reset()
        yr_p, ax_p = [], []
        d = df["steering_cmd"].to_numpy(float); v = df["odom_speed"].to_numpy(float)
        ax = df["IMU_linear_acceleration_x"].to_numpy(float); ay = df["IMU_linear_acceleration_y"].to_numpy(float)
        crt = df["current_motor"].fillna(0.0).to_numpy(float)
        for i in range(len(df)):
            y, a = self.infer.step(delta=d[i], vel=v[i], acc=ax[i], accy=ay[i], crt=crt[i])
            yr_p.append(y); ax_p.append(a)
        yr_p = np.array(yr_p); ax_p = np.array(ax_p)
        # prediction at i is for i+1
        yr_t = df["IMU_angular_velocity_z"].to_numpy(float)
        ax_t = df["IMU_linear_acceleration_x"].to_numpy(float)
        w = self.warmup
        return dict(yr_pred=yr_p[w:-1], yr_true=yr_t[w + 1:], yr_persist=yr_t[w:-1],
                    ax_pred=ax_p[w:-1], ax_true=ax_t[w + 1:], ax_persist=ax_t[w:-1])


# ----------------------------------------------------------------------------- main
def discover_models(device):
    models = []
    dep = REPO / "dynamics/deployed/trained_models"
    rt = REPO / "dynamics/retraining_20260815/checkpoints"
    cands = [
        ("GRU run30 (deployed)", dep / "gru_run30_best.pth"),
        ("GRU run38 sched.sampling", rt / "run38_schedsamp___20260815183802715201.pth"),
        ("GRU run39 h200", rt / "run39_h200___20260815185858062499.pth"),
        ("GRU run40 2-layer", rt / "run40_2layer___20260815191430922657.pth"),
        ("GRU 0731 IMU-ax/a_y_corr", REPO / "dynamics/msnn/local_run___20260731204259241445.pth"),
    ]
    for n, p in cands:
        if p.exists():
            try:
                models.append(GRUModel(n, p, device))
            except Exception as e:  # noqa: BLE001
                log(f"skip {n}: {e}")
    dd = DATA_LOCAL / "dynamics_trained_models/deep_dynamics_car02_v8"
    yml = DATA_LOCAL / "dynamics_trained_models/deep_dynamics_car02.yaml"
    if (dd / "best.pth").exists() and yml.exists():
        try:
            models.append(DeepDynamicsModel("Deep Dynamics v8", dd / "best.pth", dd / "scaler.pkl", yml, device))
        except Exception as e:  # noqa: BLE001
            log(f"skip Deep Dynamics: {e}")
    return models


def discover_msnn():
    dep = REPO / "dynamics/deployed/trained_models"
    rt = REPO / "dynamics/retraining_20260815/checkpoints"
    out = []
    for n, p in [("MS-NN exp9 phase2 (deployed)", dep / "msnn_exp9_phase2_backup"),
                 ("MS-NN a1 wide-margin phase2", rt / "msnn_a1_widemargin_phase2_backup_20260815"),
                 ("MS-NN a2 phase3 lower-lr", rt / "msnn_a2_phase3_lowerlr_20260815")]:
        if (p / "net.json").exists():
            out.append((n, p))
    return out


def run(report: Report, quick: bool = False, skip_msnn: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(env("DATA_DIR", DATA_LOCAL / "processed_data"))
    report.h(2, "Vehicle dynamics")
    report.p("Models predict the next 20 ms sample of the vehicle state from a history window of true "
             "(or, in rollouts, predicted) states and the commanded speed/steering. Errors are RMSE "
             "against MoCap ground truth (GRU, Deep Dynamics) or IMU/VESC ground truth (MS-NN). "
             "Persistence = repeating the last observed value.")

    datasets = []
    for key, fn, rows, desc in DATASETS:
        p = data_dir / fn
        if p.exists():
            df = load_dataset(p, rows)
            if quick:
                df = df.iloc[: min(len(df), 1500)].reset_index(drop=True)
            datasets.append((key, df, desc))
        else:
            report.note(f"dataset missing: {fn}")
    report.table(["dataset", "rows", "duration [s]", "mean speed [m/s]", "description"],
                 [[k, len(df), len(df) * DT, float(df["v_x_local"].abs().mean()), d] for k, df, d in datasets])

    # data sanity: IMU gyro vs MoCap yaw rate
    report.h(3, "Data sanity: IMU yaw rate vs MoCap yaw rate")
    san_rows = []
    for key, fn, rows, desc in DATASETS:
        for tag, f in ((key, fn), (key + " (MS-NN file)", MSNN_CSV_OVERRIDE.get(key))):
            if not f or not (data_dir / f).exists():
                continue
            d = load_dataset(data_dir / f, rows if f == fn else None)
            sn = imu_sanity(d)
            san_rows.append([tag, f, sn["bias"], sn["rmse"], sn["standing"], "BIASED" if abs(sn["bias"]) > IMU_BIAS_FLAG else "ok"])
            report.put("dynamics", "imu_sanity", f, value=sn)
    report.table(["dataset", "file", "mean(IMU - MoCap) [rad/s]", "RMSE [rad/s]", "|IMU| while standing", "flag"], san_rows)
    report.p("A non-zero gyro reading while the car stands still is a sensor/processing bias in that CSV, not a model "
             "error. Across all 29 processed CSVs, 7 segments show a constant bias of 0.7 to 2.0 rad/s "
             "(e.g. 20260722_115055 seg2, 20260722_113922 seg3, 20260728_164746 seg2); MoCap-based columns are unaffected.")

    models = discover_models(device)
    report.p("Models: " + "; ".join(f"**{m.name}** ({m.desc})" for m in models))

    # ---- one-step + rollouts ---------------------------------------------------------------
    one_rows, roll_curves, drift_rows, full_rows = [], {}, [], []
    speed_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0])
    speed_rows = []
    ts_store = {}
    for key, df, desc in datasets:
        starts = np.arange(0, len(df), 25)
        for m in models:
            try:
                pred, true, persist = one_step(m, df)
            except Exception as e:  # noqa: BLE001
                log(f"{m.name} on {key}: {e}"); continue
            r = rmse(pred, true); rp = rmse(persist, true)
            row = [m.name, key]
            for c in STATE3:
                if c in m.output_cols:
                    j = m.output_cols.index(c); row += [float(r[j]), float(rp[j])]
                else:
                    row += [float("nan"), float("nan")]
            one_rows.append(row)
            report.put("dynamics", "one_step", m.name, key, value={c: float(r[m.output_cols.index(c)]) for c in m.output_cols})
            if "mocap_angular_velocity" in m.output_cols:
                jy = m.output_cols.index("mocap_angular_velocity")
                bp = report.summary.setdefault("dynamics", {}).setdefault("beats_persistence", {})
                bp[m.name] = bp.get(m.name, 0) + int(r[jy] < rp[jy])
            # speed bins (yaw rate)
            if "mocap_angular_velocity" in m.output_cols:
                j = m.output_cols.index("mocap_angular_velocity")
                v = np.abs(df["v_x_local"].to_numpy())[m.history:]
                err = (pred[:, j] - true[:, j]) ** 2
                for b0, b1 in zip(speed_bins[:-1], speed_bins[1:]):
                    sel = (v >= b0) & (v < b1)
                    if sel.sum() > 50:
                        speed_rows.append([m.name, key, f"{b0:.1f}-{b1:.1f}", int(sel.sum()), float(np.sqrt(err[sel].mean()))])
            # rollouts
            H_max = max(HORIZONS)
            preds, trues, st = rollout(m, df, starts, H_max)
            if preds is not None:
                for c in ["mocap_angular_velocity", "v_x_local"]:
                    if c in m.output_cols:
                        j = m.output_cols.index(c)
                        curve = [float(np.sqrt(np.mean((preds[:, :h, j] - trues[:, :h, j]) ** 2))) for h in HORIZONS]
                        roll_curves[(key, m.name, c)] = curve
                        report.put("dynamics", "rollout_rmse_by_horizon", m.name, key, c, value=dict(zip([str(h) for h in HORIZONS], curve)))
                if all(c in m.output_cols for c in STATE3):
                    js = [m.output_cols.index(c) for c in STATE3]
                    x_p, y_p = integrate_pose(preds[:, :, js[0]], preds[:, :, js[1]], preds[:, :, js[2]])
                    x_t, y_t = integrate_pose(trues[:, :, js[0]], trues[:, :, js[1]], trues[:, :, js[2]])
                    d = np.hypot(x_p - x_t, y_p - y_t)
                    dist = np.cumsum(np.hypot(trues[:, :, js[0]], trues[:, :, js[1]]) * DT, axis=1)
                    drift_rows.append([m.name, key, float(np.median(d[:, 49])), float(np.median(d[:, 99])),
                                       float(np.median(d[:, 99] / np.maximum(dist[:, 99], 1e-3)) * 100)])
                    report.put("dynamics", "drift_m", m.name, key, value={"1s": float(np.median(d[:, 49])), "2s": float(np.median(d[:, 99]))})
            # full rollout (Kaidong-style tiers, single start after warm-up)
            for tier, n in [("60s", 3000), ("125s", 6250)]:
                Hn = min(n, len(df) - m.history - 1)
                if Hn < 500:
                    continue
                p1, t1, _ = rollout(m, df, np.array([m.history]), Hn)
                if p1 is None:
                    continue
                js3 = [m.output_cols.index(c) for c in STATE3 if c in m.output_cols]
                r3 = float(np.sqrt(np.mean((p1[0][:, js3] - t1[0][:, js3]) ** 2)))
                r5 = float(np.sqrt(np.mean((p1[0] - t1[0]) ** 2)))
                full_rows.append([m.name, key, tier, Hn * DT, r3, r5])
                report.put("dynamics", "full_rollout", m.name, key, tier, value={"rmse_3ch": r3, "rmse_5ch": r5})
            # store time series for plot
            if key in ("newtrack_20260827", "day1_test_crossday") and "mocap_angular_velocity" in m.output_cols:
                ts_store[(key, m.name)] = (pred[:, m.output_cols.index("mocap_angular_velocity")], true[:, m.output_cols.index("mocap_angular_velocity")])

    report.h(3, "One-step open-loop RMSE (model | persistence baseline)")
    report.table(["model", "dataset", "v_x [m/s]", "persist", "v_y [m/s]", "persist", "yaw rate [rad/s]", "persist"], one_rows)
    report.h(3, "Autoregressive rollout: yaw-rate RMSE vs horizon [rad/s]")
    rows = [[k[1], k[0]] + v for k, v in roll_curves.items() if k[2] == "mocap_angular_velocity"]
    report.table(["model", "dataset"] + [f"{h*DT:.1f}s" for h in HORIZONS], rows)
    report.h(3, "Autoregressive rollout: v_x RMSE vs horizon [m/s]")
    rows = [[k[1], k[0]] + v for k, v in roll_curves.items() if k[2] == "v_x_local"]
    report.table(["model", "dataset"] + [f"{h*DT:.1f}s" for h in HORIZONS], rows)
    report.h(3, "Full autoregressive rollout (warm-up 10 steps, then closed loop)")
    report.table(["model", "dataset", "tier", "length [s]", "RMSE 3ch (v_x,v_y,yaw)", "RMSE 5ch"], full_rows)
    report.h(3, "Pose drift after integrating predicted body velocities")
    report.table(["model", "dataset", "median drift @1 s [m]", "median drift @2 s [m]", "drift @2 s [% of path]"], drift_rows)
    report.h(3, "One-step yaw-rate RMSE by speed bin [rad/s]")
    report.table(["model", "dataset", "|v_x| bin [m/s]", "n", "RMSE"], speed_rows)

    # figures
    if roll_curves:
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (key, df, desc) in zip(axes, datasets):
            for (k, mn, c), curve in roll_curves.items():
                if k == key and c == "mocap_angular_velocity":
                    ax.plot([h * DT for h in HORIZONS], curve, marker="o", label=mn)
            ax.set_title(key, fontsize=9); ax.set_xlabel("horizon [s]"); ax.grid(alpha=.3)
        axes[0].set_ylabel("yaw-rate RMSE [rad/s]"); axes[-1].legend(fontsize=7)
        report.savefig(fig, "dyn_rollout_yawrate.png", "Yaw-rate error growth with rollout horizon, per dataset.")
    if one_rows:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        names = sorted({r[0] for r in one_rows}); keys = [d[0] for d in datasets]
        w = 0.8 / max(1, len(names))
        for i, n in enumerate(names):
            vals = [next((r[6] for r in one_rows if r[0] == n and r[1] == k), np.nan) for k in keys]
            ax.bar(np.arange(len(keys)) + i * w, vals, w, label=n)
        pers = [next((r[7] for r in one_rows if r[1] == k and not np.isnan(r[7])), np.nan) for k in keys]
        ax.plot(np.arange(len(keys)) + 0.4, pers, "k_", markersize=25, label="persistence")
        ax.set_xticks(np.arange(len(keys)) + 0.4); ax.set_xticklabels(keys, fontsize=8); ax.set_ylabel("one-step yaw-rate RMSE [rad/s]")
        ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=.3)
        report.savefig(fig, "dyn_onestep_yawrate.png", "One-step yaw-rate RMSE per model and dataset; black bars = persistence baseline.")
    if ts_store:
        keys_ts = sorted({k for k, _ in ts_store})
        fig, axes = plt.subplots(len(keys_ts), 1, figsize=(11, 3 * len(keys_ts)))
        for ax, key in zip(np.atleast_1d(axes), keys_ts):
            first = True
            for (k, mn), (p, t) in ts_store.items():
                if k != key: continue
                s = slice(500, 1500)
                tt = np.arange(len(t))[s] * DT
                if first: ax.plot(tt, t[s], "k", lw=1.5, label="MoCap truth"); first = False
                ax.plot(tt, p[s], lw=0.9, label=mn)
            ax.set_title(f"{key}: one-step yaw-rate prediction (20 s window)", fontsize=9); ax.set_ylabel("rad/s"); ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=4)
        axes_arr = np.atleast_1d(axes); axes_arr[-1].set_xlabel("t [s]")
        report.savefig(fig, "dyn_timeseries_yawrate.png", "Predicted vs measured yaw rate (one-step) over a 20 s window.")

    # ---- MS-NN --------------------------------------------------------------------------
    if not skip_msnn:
        report.h(3, "MS-NN (physics-structured) one-step open loop")
        report.p("Inputs: steering, odom speed, IMU a_x/a_y, motor current. Outputs: next yaw rate (IMU) and a_x. "
                 "Ratio = RMSE / persistence RMSE (below 1 beats repeating the last measurement).")
        msnn_rows = []
        for n, p in discover_msnn():
            try:
                mm = MSNNModel(n, p)
            except Exception as e:  # noqa: BLE001
                report.note(f"{n}: failed to load ({e})"); continue
            for key, df, desc in datasets:
                if key in MSNN_CSV_OVERRIDE and (data_dir / MSNN_CSV_OVERRIDE[key]).exists():
                    df = load_dataset(data_dir / MSNN_CSV_OVERRIDE[key], None)
                    key = key + "*"
                dfi = df.iloc[:600] if quick else df          # contiguous: MS-NN needs the 20 ms sequence
                try:
                    res = mm.run(dfi.reset_index(drop=True))
                except Exception as e:  # noqa: BLE001
                    report.note(f"{n} on {key}: {e}"); continue
                r_yr = float(rmse(res["yr_pred"], res["yr_true"])); p_yr = float(rmse(res["yr_persist"], res["yr_true"]))
                r_ax = float(rmse(res["ax_pred"], res["ax_true"])); p_ax = float(rmse(res["ax_persist"], res["ax_true"]))
                msnn_rows.append([n, key, r_yr, p_yr, r_yr / p_yr if p_yr else np.nan, r_ax, p_ax, r_ax / p_ax if p_ax else np.nan])
                report.put("dynamics", "msnn_one_step", n, key, value={"yaw_rate_rmse": r_yr, "yaw_rate_persist": p_yr, "ax_rmse": r_ax, "ax_persist": p_ax})
                log(f"{n} {key}: yr {r_yr:.4f} (persist {p_yr:.4f}) ax {r_ax:.4f}")
        report.table(["model", "dataset", "yaw RMSE", "persist", "ratio", "a_x RMSE", "persist", "ratio"], msnn_rows)
        report.p("\* Day-2 file replaced by session_20260722_115055_seg1 (clean IMU, not in MS-NN training).")

    report.h(3, "Reading the dynamics results")
    report.p("One-step errors near the persistence baseline mean the model mostly reproduces the last state; "
             "the rollout curves show how fast errors compound when the model drives itself, which is what matters "
             "for closed-loop simulation. Cross-day (Day 1) and new-track results measure generalisation beyond the "
             "training sessions.")
    return True
