"""
Deep Dynamics for F1TENTH Car02
Adapted from arxiv 2312.04374 (Chrosniak et al., IEEE RA-L 2024)

Architecture:
  Input  : [v_x, v_y, yaw_rate, speed_cmd, steering_cmd] × horizon steps
  GRU    : extracts temporal features
  FC     : maps to raw parameter estimates
  Guard  : sigmoid-clips to physically valid ranges
  Physics: bicycle model (Pacejka) computes next state

Output : [v_x_next, v_y_next, yaw_rate_next]

No dependency on the deep-dynamics package — fully self-contained.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Physics Guard ────────────────────────────────────────────────────────────

class PhysicsGuard(nn.Module):
    """Sigmoid-based output layer that clamps parameters to [min, max]."""
    def __init__(self, param_cfgs):
        super().__init__()
        mins   = torch.tensor([p["min"] for p in param_cfgs], dtype=torch.float32)
        ranges = torch.tensor([p["max"] - p["min"] for p in param_cfgs], dtype=torch.float32)
        self.register_buffer("mins",   mins)
        self.register_buffer("ranges", ranges)

    def forward(self, x):
        return torch.sigmoid(x) * self.ranges + self.mins


# ─── Model ───────────────────────────────────────────────────────────────────

class DeepDynamicsCar02(nn.Module):
    """
    Physics-constrained neural network for F1TENTH Car02 dynamics.

    The NN estimates per-timestep Pacejka tire coefficients (Bf,Cf,Df,Ef,
    Br,Cr,Dr,Er), net longitudinal force Frx, and moment of inertia Iz.
    A single-track bicycle model then integrates to predict the next state.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        v = cfg["vehicle"]
        self.lf          = v["lf"]
        self.lr          = v["lr"]
        self.mass        = v["mass"]
        self.steer_scale = v.get("steer_scale", 0.35)
        self.dt          = v.get("dt", 0.02)
        self.v0          = 0.5   # low-speed numerical stabiliser (same as bicycle_model.py)

        mc  = cfg["model"]
        n_input   = 5            # [v_x, v_y, yaw_rate, speed_cmd, steering_cmd]
        n_params  = len(cfg["parameters"])
        horizon   = mc["horizon"]
        gru_h     = mc["gru_hidden"]
        gru_l     = mc["gru_layers"]
        fc_h      = mc["fc_hidden"]
        fc_l      = mc["fc_layers"]

        self.horizon = horizon

        # GRU backbone
        self.gru = nn.GRU(
            input_size=n_input,
            hidden_size=gru_h,
            num_layers=gru_l,
            batch_first=True,
            dropout=0.1 if gru_l > 1 else 0.0,
        )

        # Fully-connected head
        fc_in = gru_h * horizon
        layers = []
        for _ in range(fc_l):
            layers += [nn.Linear(fc_in, fc_h), nn.Mish()]
            fc_in = fc_h
        self.fc = nn.Sequential(*layers)

        # Raw parameter projection + guard
        self.raw_out     = nn.Linear(fc_h, n_params)
        self.guard       = PhysicsGuard(cfg["parameters"])
        self.param_names = [p["name"] for p in cfg["parameters"]]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x_raw, x_norm):
        """
        x_raw  : (B, horizon, 5)  — unnormalised [v_x, v_y, ψ̇, spd_cmd, str_cmd]
        x_norm : (B, horizon, 5)  — StandardScaler-normalised (GRU input)
        Returns: (B, 3)           — [v_x_next, v_y_next, ψ̇_next]
        """
        gru_out, _ = self.gru(x_norm)                        # (B, H, gru_h)
        flat       = gru_out.reshape(gru_out.size(0), -1)    # (B, H*gru_h)
        fc_out     = self.fc(flat)                            # (B, fc_h)
        params_raw = self.raw_out(fc_out)                     # (B, n_params)
        params     = self.guard(params_raw)                   # (B, n_params) – physically bounded

        param_dict = {name: params[:, i] for i, name in enumerate(self.param_names)}
        state_t    = x_raw[:, -1, :]                          # last timestep (B, 5)
        return self._bicycle_step(state_t, param_dict)

    # ── bicycle model ─────────────────────────────────────────────────────────

    def _bicycle_step(self, state_t, p):
        """Euler integration of single-track model — all ops differentiable."""
        v_x       = state_t[:, 0]
        v_y       = state_t[:, 1]
        psi_dot   = state_t[:, 2]
        # state_t[:,3] = speed_cmd  (used indirectly via Frx learned by NN)
        steering  = state_t[:, 4]   # already in radians (servo calibration in extract_bag_for_felix.py)

        vx_safe = torch.abs(v_x) + self.v0   # numerical stability

        # Slip angles  (sign convention from bicycle_model.py)
        # steering_cmd is already in radians (converted by servo calibration in extract_bag_for_felix.py)
        alpha_f = steering - torch.atan2( v_y + self.lf * psi_dot, vx_safe)
        alpha_r =            torch.atan2(-v_y + self.lr * psi_dot, vx_safe)

        # Vertical loads (static, no load transfer for simplicity)
        g     = 9.81
        F_zf  = self.mass * g * self.lr / (self.lf + self.lr)
        F_zr  = self.mass * g * self.lf / (self.lf + self.lr)

        # Pacejka lateral forces
        Fyf = p["Df"] * F_zf * torch.sin(
            p["Cf"] * torch.atan(
                p["Bf"] * alpha_f - p["Ef"] * (p["Bf"] * alpha_f - torch.atan(p["Bf"] * alpha_f))
            )
        )
        Fyr = p["Dr"] * F_zr * torch.sin(
            p["Cr"] * torch.atan(
                p["Br"] * alpha_r - p["Er"] * (p["Br"] * alpha_r - torch.atan(p["Br"] * alpha_r))
            )
        )

        Frx = p["Frx"]   # net longitudinal force [N], estimated by NN
        Iz  = p["Iz"]

        # State derivatives (vehicle frame)
        dv_x   = (Frx - Fyf * torch.sin(steering)) / self.mass + v_y * psi_dot
        dv_y   = (Fyr + Fyf * torch.cos(steering)) / self.mass - v_x * psi_dot
        dpsi   = (self.lf * Fyf * torch.cos(steering) - self.lr * Fyr) / Iz

        # Euler step
        v_x_next   = v_x   + dv_x * self.dt
        v_y_next   = v_y   + dv_y * self.dt
        psi_next   = psi_dot + dpsi * self.dt

        return torch.stack([v_x_next, v_y_next, psi_next], dim=1)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DeepDynamicsDatasetCar02(torch.utils.data.Dataset):
    """
    Sliding-window dataset from pre-processed Car02 CSV files.

    Each sample:
      x_raw  : (horizon, 5)  — raw states + controls
      x_norm : (horizon, 5)  — StandardScaler-normalised
      label  : (3,)          — [v_x, v_y, ψ̇] at t+1
    """

    COLS_IN  = ["v_x_local", "v_y_local", "mocap_angular_velocity",
                "speed_cmd", "steering_cmd"]
    COLS_OUT = ["v_x_local", "v_y_local", "mocap_angular_velocity"]

    def __init__(self, csv_files, horizon, scaler=None, min_speed=0.3):
        import pandas as pd

        segs_x, segs_y = [], []

        for f in csv_files:
            df   = pd.read_csv(f, index_col=0)
            x_df = df[self.COLS_IN].values.astype(np.float32)
            y_df = df[self.COLS_OUT].values.astype(np.float32)

            # filter stationary rows
            speed_mask = np.abs(x_df[:, 0]) > min_speed
            x_df = x_df[speed_mask]
            y_df = y_df[speed_mask]

            # sliding window:  x[t-h:t] → y[t]
            for t in range(horizon, len(x_df)):
                segs_x.append(x_df[t - horizon: t])   # (horizon, 5)
                segs_y.append(y_df[t])                 # (3,)

        self.x_raw = np.stack(segs_x, axis=0)   # (N, horizon, 5)
        self.y     = np.stack(segs_y, axis=0)   # (N, 3)

        # Fit / apply normaliser on flattened input
        N, H, F = self.x_raw.shape
        flat = self.x_raw.reshape(-1, F)
        if scaler is None:
            self.scaler = StandardScaler()
            norm_flat   = self.scaler.fit_transform(flat)
        else:
            self.scaler = scaler
            norm_flat   = self.scaler.transform(flat)
        self.x_norm = norm_flat.reshape(N, H, F).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_raw[idx]),
            torch.from_numpy(self.x_norm[idx]),
            torch.from_numpy(self.y[idx]),
        )

    def split(self, train_ratio=0.85):
        n      = len(self)
        n_tr   = int(n * train_ratio)
        idx    = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        tr_idx = torch.utils.data.Subset(self, idx[:n_tr].tolist())
        va_idx = torch.utils.data.Subset(self, idx[n_tr:].tolist())
        return tr_idx, va_idx


# ─── Closed-loop evaluator ───────────────────────────────────────────────────

@torch.no_grad()
def closed_loop_rmse(model, csv_file, scaler, horizon, dt=0.02):
    """
    Auto-regressive rollout on a single CSV file.
    Predictions are fed back as inputs — same metric as Jonathan's CL-RMSE.
    Returns dict of per-state RMSE and mean CL-RMSE.
    """
    import pandas as pd

    COLS_IN  = DeepDynamicsDatasetCar02.COLS_IN
    COLS_OUT = DeepDynamicsDatasetCar02.COLS_OUT

    df    = pd.read_csv(csv_file, index_col=0)
    x_df  = df[COLS_IN].values.astype(np.float32)
    y_df  = df[COLS_OUT].values.astype(np.float32)

    model.eval()
    model.to(device)

    preds  = []
    truths = []

    # initialise buffer with ground-truth history
    buf = x_df[:horizon].copy()   # shape (horizon, 5)

    for t in range(horizon, len(x_df)):
        # normalise buffer
        buf_flat = scaler.transform(buf)                          # (horizon, 5)
        x_raw  = torch.from_numpy(buf[np.newaxis]).to(device)    # (1, H, 5)
        x_norm = torch.from_numpy(buf_flat[np.newaxis]).float().to(device)

        pred = model(x_raw, x_norm).cpu().numpy()[0]             # (3,)
        preds.append(pred)
        truths.append(y_df[t])

        # shift buffer: feed predicted v_x, v_y, ψ̇ back; keep GT controls
        new_row        = buf[-1].copy()
        new_row[0:3]   = pred
        new_row[3:5]   = x_df[t, 3:5]   # speed_cmd, steering_cmd from GT
        buf            = np.vstack([buf[1:], new_row])

    preds  = np.array(preds)    # (N, 3)
    truths = np.array(truths)   # (N, 3)

    rmse_per = np.sqrt(np.mean((preds - truths) ** 2, axis=0))
    labels   = ["v_x", "v_y", "ψ̇"]

    result = {l: float(r) for l, r in zip(labels, rmse_per)}
    result["mean_CL_RMSE"] = float(np.mean(rmse_per))
    return result, preds, truths
