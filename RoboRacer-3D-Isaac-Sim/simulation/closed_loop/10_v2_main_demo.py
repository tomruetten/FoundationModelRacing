"""
10_v2_main_demo.py — Isaac Sim V2: Closed-Loop Control with GRU Plant Model
=============================================================================

V1 replayed a recorded CSV.  V2 is a real closed loop:
  Reference path → Pure Pursuit controller → control commands
  → GRU dynamics model (plant) → predicted next state
  → State integrator → new (x, y, θ)
  → back to Pure Pursuit  (loop)

Optionally compares GRU ψ̇ with MS-NN ψ̇ side-by-side using the existing
MS-NN IPC server (05_msnn_dynamics_server.py).

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Reference path (x,y waypoints from CSV)            │
  │         ↓                                           │
  │  Pure Pursuit Controller                            │
  │         ↓  steering_cmd, speed_cmd                  │
  │  GRU Dynamics Model  ←──── (vx,vy,yr,ax,ay state)  │
  │         ↓  vx, vy, yr, ax, ay  (predicted)          │
  │  State Integrator (Euler 20ms)                      │
  │         ↓  x, y, θ   →  back to Pure Pursuit       │
  └─────────────────────────────────────────────────────┘
  Optional: MS-NN IPC server → ψ̇ comparison plot

Usage (minimal):
  python3 10_v2_main_demo.py \\
    --gru ../../dynamics/deployed/trained_models/gru_run30_best.pth \\
    --ref-csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv

Usage (with MS-NN ψ̇ comparison):
  # Terminal 1 — start MS-NN server first:
  conda run -n nnodely-env python3 05_msnn_dynamics_server.py \\
    --model ../../dynamics/deployed/trained_models/msnn_exp9_phase2_backup \\
    --queue-dir /tmp/isaac_msnn_v2

  # Terminal 2 — V2 demo:
  python3 10_v2_main_demo.py \\
    --gru ../../dynamics/deployed/trained_models/gru_run30_best.pth \\
    --ref-csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv \\
    --msnn-queue /tmp/isaac_msnn_v2

Output (in v2_output/):
  v2_trajectory.csv   ← full log: t, x, y, θ, vx, yr, steering_cmd, tracking_error, ...
  v2_trajectory.png   ← top-view path + tracking error + speed + ψ̇ comparison
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── GRU Model (identical to training, RNN_jomoh_1) ────────────────────────────

class RNN_jomoh_1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru1      = nn.GRU(input_size, 100, batch_first=True)
        self.fc1       = nn.Linear(100, 100)
        self.fc3       = nn.Linear(100, 100)
        self.fc5       = nn.Linear(100, output_size)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out, _ = self.gru1(x)
        x = self.LeakyReLU(self.fc1(out[:, -1, :]))
        x = self.LeakyReLU(self.fc3(x))
        return self.fc5(x)


# ── GRU Inference (in-process, no IPC needed) ─────────────────────────────────

_EXPECTED_INPUT_COLS = [
    'v_x_local', 'v_y_local', 'mocap_angular_velocity',
    'a_x_local', 'a_y_local', 'speed_cmd', 'steering_cmd'
]
_EXPECTED_OUTPUT_COLS = [
    'v_x_local', 'v_y_local', 'mocap_angular_velocity',
    'a_x_local', 'a_y_local'
]


_ONBOARD_INPUT_COLS = ['odom_speed', 'IMU_angular_velocity_z',
                       'IMU_linear_acceleration_x', 'IMU_linear_acceleration_y',
                       'steering_cmd', 'current_motor']
_ONBOARD_OUTPUT_COLS = ['odom_speed', 'IMU_angular_velocity_z',
                        'IMU_linear_acceleration_x', 'IMU_linear_acceleration_y']


class GRUInference:
    HISTORY = 10  # 10 timesteps × 20ms = 200ms lookback

    def __init__(self, checkpoint: str, device: str = 'cpu'):
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg  = ckpt.get('config', {})

        # P0-2: Verify checkpoint columns match Run30 specification.
        # Wrong columns (e.g. IMU_linear_acceleration_x, a_y_corr) cause
        # ~2.2 m/s² systematic lateral error in V2 closed-loop.
        got_in  = cfg.get('input_columns',  [])
        got_out = cfg.get('output_columns', [])
        in_sz  = cfg.get('input_size',  7)
        out_sz = cfg.get('output_size', 5)

        # ADAPT 2026-08-27: on-disk checkpoints (both gru_run30_best.pth and
        # local_run___20260731204259241445.pth) don't exactly match this
        # class's original hardcoded 7in/5out MoCap Run30 spec -- same root
        # cause 15_gru_inference.py's 2026-08-13 fix already documented
        # (checkpoint is really the onboard-only, no-MoCap Run37-equivalent
        # model). That fix adapted the SIMPLE, stateless inference wrapper
        # to the actual checkpoint; this class is harder to adapt cleanly
        # because V2's closed-loop state integrator needs a real v_y term
        # the onboard model never predicts (it has no lateral-velocity
        # output at all, only speed/yr/ax/ay) -- so onboard mode below
        # approximates v_y=0 (reasonable at the low, already-validated
        # closed-loop speeds ~0.3-0.8 m/s where side-slip is small, but a
        # real approximation, not the true model) and current_motor=0 for
        # the input the closed loop has no natural equivalent for -- same
        # precedent as this file's existing `crt=0` handling for MS-NN
        # (see the "motor current unavailable in Isaac Sim" comment further
        # down in this file). Flagged loudly at load time and in outputs so
        # this is never silently mistaken for the full 7-channel model.
        self._onboard_only = (got_in == _ONBOARD_INPUT_COLS or (in_sz == 6 and out_sz == 4))
        if self._onboard_only:
            print(f"[GRU] WARNING: onboard-only checkpoint (6in/4out, no MoCap v_y) -- "
                  f"NOT the documented 7in/5out Run30 spec. Approximating v_y=0 and "
                  f"current_motor=0 in the closed loop below. Treat results as reflecting "
                  f"this approximation, not the full MoCap model.", flush=True)
            self.model = RNN_jomoh_1(input_size=6, output_size=4).to(self.device)
        elif got_in and got_in != _EXPECTED_INPUT_COLS:
            raise RuntimeError(
                f"[GRU] WRONG checkpoint columns!\n"
                f"  Expected: {_EXPECTED_INPUT_COLS}\n"
                f"  Got:      {got_in}\n"
                f"  → Replace with gru_run30_best.pth (see README_V2.txt SCHRITT 0)"
            )
        elif in_sz != 7 or out_sz != 5:
            raise RuntimeError(
                f"[GRU] Wrong architecture: expected 7in/5out, got {in_sz}in/{out_sz}out\n"
                f"  → This checkpoint is likely Run37 (onboard-only). Use Run30."
            )
        else:
            self.model = RNN_jomoh_1(input_size=7, output_size=5).to(self.device)

        sd = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(sd)
        self.model.eval()

        val_loss = cfg.get('best_val_loss', '?')
        print(f"[GRU] Loaded: {checkpoint}")
        print(f"[GRU] input_cols: {got_in or '(not in config)'}")
        print(f"[GRU] val_loss  : {val_loss:.4f}" if isinstance(val_loss, float) else f"[GRU] val_loss: {val_loss}")
        self.reset()

    def reset(self):
        w = 6 if self._onboard_only else 7
        self._buf = deque([[0.0] * w for _ in range(self.HISTORY)], maxlen=self.HISTORY)

    def step(self, vx, vy, yr, ax, ay, speed_cmd, steering_cmd) -> dict:
        if self._onboard_only:
            self._buf.append([vx, yr, ax, ay, steering_cmd, 0.0])  # current_motor=0, see __init__ note
        else:
            self._buf.append([vx, vy, yr, ax, ay, speed_cmd, steering_cmd])
        x = torch.tensor(list(self._buf), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x).squeeze(0).cpu().numpy()
        if self._onboard_only:
            return {'vx': float(out[0]), 'vy': 0.0, 'yr': float(out[1]),
                    'ax': float(out[2]), 'ay': float(out[3])}
        return {'vx': float(out[0]), 'vy': float(out[1]), 'yr': float(out[2]),
                'ax': float(out[3]), 'ay': float(out[4])}


# ── MS-NN IPC Client (optional, for ψ̇ comparison) ────────────────────────────

class MSNNIPCClient:
    """Matches the IPC protocol of 05_msnn_dynamics_server.py."""

    def __init__(self, queue_dir: str):
        self._dir  = Path(queue_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Clean up stale files from previous runs to avoid wrong responses
        for f in self._dir.glob('req_*.json'): f.unlink(missing_ok=True)
        for f in self._dir.glob('res_*.json'): f.unlink(missing_ok=True)
        self._step = 0

    def reset(self):
        req = {'step': -1, 'reset': True, 'delta': 0.0,
               'vel': 0.0, 'acc': 0.0, 'accy': 0.0, 'crt': 0.0}
        self._step = 0
        self._send_recv(req)

    def step(self, delta, vel, acc, accy, crt=0.0):
        req = {'step': self._step, 'delta': float(delta), 'vel': float(vel),
               'acc': float(acc), 'accy': float(accy), 'crt': float(crt)}
        res = self._send_recv(req)
        self._step += 1
        return float(res.get('yr_pred', 0.0)), float(res.get('ax_pred', 0.0))

    def _send_recv(self, req):
        idx      = f"{req['step']:06d}" if req['step'] >= 0 else 'reset'
        req_path = self._dir / f'req_{idx}.json'
        res_path = self._dir / f'res_{idx}.json'
        req_path.write_text(json.dumps(req))
        for _ in range(100):        # wait up to 2s
            if res_path.exists():
                try:
                    res = json.loads(res_path.read_text())
                    res_path.unlink()
                    return res
                except Exception:
                    pass
            time.sleep(0.02)
        print(f"[MS-NN IPC] Timeout at step {req['step']}")
        return {'yr_pred': 0.0, 'ax_pred': 0.0}


# ── Pure Pursuit Controller ────────────────────────────────────────────────────

class PurePursuit:
    """
    Standard Pure Pursuit path-following controller.
    Ref: Coulter 1992, "Implementation of the Pure Pursuit Path Tracking Algorithm"
    """

    def __init__(self, wheelbase=0.3225, lookahead=0.5,
                 target_speed=1.5, max_steering=0.29,
                 train_speed_min=0.3, train_speed_max=2.5,
                 adaptive_lookahead=False, ld_min=0.3, ld_max=2.0, ld_curve_k=3.0,
                 path_v=None):
        self.L              = wheelbase
        self.ld             = lookahead      # base/configured lookahead
        self.v_target       = target_speed
        # P5-1 (2026-08-26): optional per-path-point target-speed profile
        # (e.g. a raceline's own ref_v optimal-speed array), same point
        # count/order as the path_xy passed into compute(). None (default)
        # preserves the old constant-target_speed behavior exactly --
        # existing callers (track1/track2 tests) are unaffected.
        self.path_v         = path_v
        self.d_max          = max_steering
        # P1-4: clamp speed to training distribution to avoid OOD GRU inputs
        self.spd_min        = train_speed_min
        self.spd_max        = train_speed_max
        # P3-1: optional curvature-adaptive lookahead (2026-08-15). See compute()
        # for the formula and the reasoning. Default OFF -- preserves the
        # original fixed-lookahead behavior exactly (regression-tested).
        self.adaptive_lookahead = adaptive_lookahead
        self.ld_min          = ld_min
        self.ld_max          = ld_max
        self.ld_curve_k      = ld_curve_k
        self.ld_effective    = lookahead     # diagnostic: ld actually used last compute() call
        # P1-2: local search window — avoids global nearest-point jumps on curved/loop paths
        self._last_closest  = 0
        self.closest_idx    = 0
        self.target_idx     = 0      # lookahead target point index (diagnostic)
        self.steering_clipped = False
        self.speed_clipped    = False
        # P2-9: cumulative arc-length cache for end-of-path slowdown (see compute()).
        # Keyed by id(path_xy) since the same array is reused every call in run_v2().
        self._cum_dist_cache_id  = None
        self._cum_dist           = None

    def compute(self, x, y, theta, path_xy):
        """Returns (steering_cmd [rad], speed_cmd [m/s])."""
        # P2-9: BUGFIX 2026-08-14 — build/reuse an arc-length lookup table for this
        # path. Found by running Z1 on session_20260728_163324_seg1_20ms.csv: the
        # path is a LOOP (start (0,0) and end (0.26,0.41) are only 0.48m apart in
        # straight-line distance, despite spanning 10294 points / ~206s). The old
        # "slow down near the end" logic measured straight-line distance to
        # path_xy[-1], which is ~0 for a loop track from step 0 onward -- the
        # controller thought it was "arriving" the entire time and crushed
        # speed_raw to the floor (observed: GRU vx collapsed to 0.14 m/s,
        # progress fraction 0.010 after 20s). Arc-length-remaining is loop-safe.
        if self._cum_dist_cache_id != id(path_xy):
            seg_lens = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1]))
            self._cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
            self._cum_dist_cache_id = id(path_xy)

        # P1-2: local window search (±200 from last closest) to prevent index jumps
        n = len(path_xy)
        lo = max(0, self._last_closest - 20)
        hi = min(n, self._last_closest + 200)
        window_xy = path_xy[lo:hi]
        dists_window = np.hypot(window_xy[:, 0] - x, window_xy[:, 1] - y)
        closest = lo + int(np.argmin(dists_window))
        self._last_closest = closest
        self.closest_idx   = closest

        # Curvature-adaptive speed AND (optionally) lookahead both need a local
        # curvature estimate, so compute it once here, before target search --
        # moved earlier than in the original code (used to run after the
        # steering block, speed-only) so adaptive lookahead can use it too.
        # BUGFIX 2026-08-15 (BUG-V2-14): estimate curvature over a fixed ARC
        # LENGTH ahead, not a fixed POINT COUNT ("10 points ahead"). Same bug
        # class as BUG-V2-12 (index/point-count vs arc-length). Diagnosed on
        # track2 (122000): its reference path has severe point-spacing
        # pathology -- 50.6% of consecutive points are <5mm apart (vs 7.5% on
        # track1), evidently from slow/near-stationary recording segments.
        # With a fixed "10 points ahead" window, landing near one of these
        # degenerate clusters collapses seg_len toward the 1e-6 floor and
        # curvature explodes to physically meaningless values (observed: real
        # driven-trajectory replay gave mean=27130, median=0.775, max=1.57M --
        # dominated by numerical spikes, not real geometry). This directly
        # explained why --adaptive-lookahead regressed badly on track2 (the
        # spiky signal yanks ld_eff toward ld_min at recording artifacts, not
        # real corners). track1's much more uniform spacing (CV=0.398 vs
        # track2's 1.095) made the point-count version accidentally fine
        # there. CURVE_LOOKAHEAD_M=0.4 chosen to match track1's own typical
        # "10 points" arc length (mean spacing 0.041m * 10 ~= 0.41m), so
        # track1's already-working behavior is preserved as closely as
        # possible while track2's pathology is fixed at the source.
        CURVE_LOOKAHEAD_M = 0.4
        target_cum = self._cum_dist[closest] + CURVE_LOOKAHEAD_M
        look_end = int(np.searchsorted(self._cum_dist, target_cum, side='left'))
        look_end = min(look_end, len(path_xy) - 1)
        if look_end > closest:
            dp       = path_xy[look_end] - path_xy[closest]
            seg_len  = np.hypot(dp[0], dp[1]) + 1e-6
            # BUGFIX 2026-08-15 (BUG-V2-13): theta is an unbounded raw integral
            # (StateIntegrator never wraps it to [-pi,pi]), so once |theta| drifts
            # past pi, the raw difference (atan2(dp) - theta) can itself exceed
            # 2*pi in magnitude. The old `min(head_chg, 2*pi-head_chg)` idiom only
            # unwraps a single +/-2*pi ambiguity -- past that it silently returns a
            # NEGATIVE head_chg, which flips curvature negative and lets
            # `1 - k_curve*curvature` exceed 1.0, commanding speed ABOVE v_target
            # (confirmed: this is the actual cause of the ~73% "speed_clipped" seen
            # in every V2 run so far -- the controller was hitting the spd_max=2.5
            # ceiling from ABOVE v_target, not the floor from below it). Use the
            # same robust wrap already used correctly elsewhere in this file
            # (see `alpha` below, and theta_delta/heading_error in run_v2())
            # instead of the ad-hoc min() form.
            head_chg = abs(((np.arctan2(dp[1], dp[0]) - theta) + np.pi) % (2 * np.pi) - np.pi)
            curvature = head_chg / seg_len                    # [rad/m]
        else:
            curvature = 0.0

        # BUGFIX 2026-08-18 (BUG-V2-15): `curvature` above is a HEADING-ERROR
        # quantity (car's current `theta` vs direction to a point ahead) --
        # correct for the speed-adaptation use it was originally written for
        # (deciding "how hard do I need to turn right now" legitimately
        # depends on where the car is currently facing), but wrong for
        # lookahead-adaptation, which needs the PATH's OWN curvature,
        # independent of the car's current heading. Verified on track2
        # (122000, see the V2 lab notes (not in this repo) Section 30 for the full Spearman-
        # correlation analysis): the two quantities are only weakly -- and
        # slightly NEGATIVELY -- correlated (rho=-0.233) whenever the car's
        # heading lags the path, which is exactly what happens on track2's
        # densely-packed corners (45 corners / 74.6m, 74% of gaps <1m) but
        # not on track1's gentler ones. That mismatch flips ld_effective's
        # response to real curvature backwards (rho=+0.256 vs true curvature,
        # should be strongly negative by the formula's own design intent).
        # Fixed by computing a genuine geometric curvature here -- the angle
        # between two PATH-direction vectors (the segment behind `closest`
        # vs the segment ahead of it), with no car state involved at all --
        # and using it ONLY for the lookahead-adaptive branch below. The
        # speed-adaptation `curvature` above is deliberately left untouched.
        HALF_WINDOW_M = CURVE_LOOKAHEAD_M / 2.0
        back_cum  = max(self._cum_dist[closest] - HALF_WINDOW_M, 0.0)
        look_back = int(np.searchsorted(self._cum_dist, back_cum, side='left'))
        look_back = max(look_back, 0)
        if look_end > closest and closest > look_back:
            dp_ahead   = path_xy[look_end] - path_xy[closest]
            dp_behind  = path_xy[closest] - path_xy[look_back]
            dir_ahead  = np.arctan2(dp_ahead[1], dp_ahead[0])
            dir_behind = np.arctan2(dp_behind[1], dp_behind[0])
            path_bend  = abs(((dir_ahead - dir_behind) + np.pi) % (2 * np.pi) - np.pi)
            span       = (self._cum_dist[look_end] - self._cum_dist[look_back]) + 1e-6
            path_curvature = path_bend / span              # [rad/m], geometric only
        else:
            path_curvature = 0.0

        # P3-1: optional curvature-adaptive lookahead (2026-08-15). Shrinks
        # lookahead in sharp curves (reduces corner-cutting, which is where it
        # hurts tracking most -- BUG-V2-03's own note: large lookahead cuts
        # corners more) and relaxes back toward the base value on straights
        # (low curvature -> ld_eff ~ self.ld). Opt-in via --adaptive-lookahead;
        # default False means ld_eff == self.ld always, i.e. byte-identical to
        # the original fixed-lookahead behavior (regression-tested, see
        # postfix_gru_batch tooling). Motivated by the 2026-08-15 lookahead
        # re-optimization finding: track1's tracking-optimal lookahead (~0.3)
        # is catastrophic on track2 (99.4% steer_clip) and vice versa -- a
        # single fixed constant can't be optimal everywhere, but reacting to
        # local curvature instead of a whole-track average might get closer
        # to each track's own local optimum without needing per-track configs.
        # Uses `path_curvature` (BUG-V2-15 fix above), not the heading-error
        # `curvature` used for speed-adaptation.
        if self.adaptive_lookahead:
            ld_eff = float(np.clip(self.ld / (1.0 + self.ld_curve_k * path_curvature),
                                   self.ld_min, self.ld_max))
        else:
            ld_eff = self.ld
        self.ld_effective = ld_eff

        dists = np.hypot(path_xy[closest:, 0] - x, path_xy[closest:, 1] - y)
        goal  = path_xy[-1]
        self.target_idx = len(path_xy) - 1
        for i, d in enumerate(dists):
            if d >= ld_eff:
                goal = path_xy[closest + i]
                self.target_idx = closest + i
                break

        # Angle from vehicle heading to goal
        alpha = np.arctan2(goal[1] - y, goal[0] - x) - theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Pure pursuit steering formula
        delta_raw = np.arctan2(2.0 * self.L * np.sin(alpha), ld_eff)
        delta = float(np.clip(delta_raw, -self.d_max, self.d_max))
        self.steering_clipped = bool(abs(float(delta_raw)) > self.d_max)

        # Reduce speed proportional to curvature (k_curve tuned for F1TENTH)
        # P5-1: base speed comes from the raceline's own ref_v at the current
        # closest-point index when a path_v profile was supplied, else the
        # constant self.v_target (old behavior). The curvature/end-of-path
        # damping below still applies on top either way -- it's a car-state
        # (heading-error) driven safety factor, not a duplicate of ref_v's
        # own curvature-awareness, so keeping it is not double-penalizing.
        base_speed = self.v_target if self.path_v is None else float(self.path_v[closest])
        k_curve = 1.5
        speed_raw = base_speed * max(0.35, 1.0 - k_curve * curvature)

        # Also slow down near path end. P2-9: arc-length remaining (loop-safe),
        # NOT straight-line distance to path_xy[-1] -- see cache comment above.
        remaining = float(self._cum_dist[-1] - self._cum_dist[closest])
        speed_raw = speed_raw * min(1.0, remaining / (self.ld * 3))
        # P1-4: clamp to training distribution
        speed = float(np.clip(speed_raw, self.spd_min, self.spd_max))
        self.speed_clipped = bool(speed_raw < self.spd_min or speed_raw > self.spd_max)

        return delta, speed


# ── State Integrator ──────────────────────────────────────────────────────────

class StateIntegrator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.reset()

    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta

    def update(self, vx, vy, yr):
        """Euler integration. Returns (x, y, theta)."""
        ct, st = np.cos(self.theta), np.sin(self.theta)
        self.x     += (vx * ct - vy * st) * self.dt
        self.y     += (vx * st + vy * ct) * self.dt
        self.theta += yr * self.dt
        return self.x, self.y, self.theta


# ── Live ψ̇ Filter (ADDED 2026-08-25) ──────────────────────────────────────────
#
# Investigated for Felix's "heading looks a bit off on straights" feedback
# (see report/Felix_new_task_list_20260825.txt item 8, not included in this repo). Root cause diagnosed
# there: GRU's instantaneous yaw-rate prediction carries small noise even
# while driving straight, so theta_integrated visibly wobbles a few degrees
# even though the car is not actually turning. The already-deployed
# --yr-filter (median_clip) does NOT address this -- it only smooths the
# REFERENCE PATH RECONSTRUCTION in extract_reference()/_filter_yr() above,
# which runs once, offline, before the closed loop even starts. It never
# touches the live yr signal that drives theta_integrated step-by-step.
#
# This filter is the first attempt at directly smoothing that live signal.
# It is applied to yr_for_integration -- the (possibly Hybrid-blended) yaw
# rate that StateIntegrator.update() actually uses each step -- and ONLY
# there: it does not touch the GRU's own recurrent state (gru_state, always
# fed pred['yr'] unfiltered, per the Hybrid-A scheme), theta_gru_only, or
# yr_diff (both computed from raw pred['yr']/msnn_yr). It is causal (only
# past/current samples), unlike the offline center=True rolling median in
# _filter_yr, since it must run online inside the simulation loop.
#
# Default is 'none', which reproduces the exact currently-deployed behavior
# (yr_for_integration passed straight to integrator.update(), unchanged) --
# this option is opt-in and does not alter any existing deployed run.
class LiveYrFilter:
    """Causal real-time filter for the per-step integration yaw rate.

    mode: 'none'   -- passthrough (default; no behavior change)
          'ema'     -- exponential moving average:
                       y[n] = alpha*x[n] + (1-alpha)*y[n-1]
                       Smaller alpha = more smoothing (more lag).
          'median'  -- causal rolling median over the last `window` raw
                       samples (including the current one).
    """
    def __init__(self, mode: str = 'none', window: int = 5, alpha: float = 0.3,
                 gate_thresh: float = 0.0):
        if mode not in ('none', 'ema', 'median'):
            raise ValueError(f"unknown --live-yr-filter mode: {mode}")
        self.mode   = mode
        self.window = max(1, int(window))
        self.alpha  = float(alpha)
        self.gate_thresh = float(gate_thresh)
        self._buf   = deque(maxlen=self.window)
        self._ema   = None

    def reset(self):
        self._buf.clear()
        self._ema = None

    def apply(self, yr_raw: float) -> float:
        # 2026-08-26: optional magnitude gate. Only smooth while the raw
        # signal is already small (near-straight driving -- exactly the
        # "heading looks a bit off on straights" complaint); pass real
        # cornering yr straight through, unfiltered, AND reset filter state
        # to the raw value so there's no lag re-entering a straight after a
        # turn. Rationale: the earlier ungated --live-yr-filter ema testing
        # (alpha=0.15) did cut arrow noise (5.29deg->3.21deg on track1) but
        # also nearly doubled theta_delta drift (8.6deg->19.2deg) because it
        # lagged the signal everywhere, including in corners where accuracy
        # matters most. Gating confines the lag to where it's harmless.
        if self.gate_thresh > 0.0 and abs(yr_raw) >= self.gate_thresh:
            self.reset()
            return yr_raw
        if self.mode == 'none':
            return yr_raw
        if self.mode == 'ema':
            self._ema = yr_raw if self._ema is None \
                        else self.alpha * yr_raw + (1.0 - self.alpha) * self._ema
            return self._ema
        # 'median'
        self._buf.append(yr_raw)
        return float(np.median(self._buf))


# ── Alpha Ramp Helper ─────────────────────────────────────────────────────────

def compute_alpha_effective(step: int, alpha_target: float,
                            warmup: int = 25, ramp: int = 25) -> float:
    """
    Smoothstep ramp for MS-NN blend weight after warmup.
    step < warmup:           alpha=0 (MS-NN buffer not yet valid)
    warmup <= step < warmup+ramp: smoothstep ramp 0→alpha_target
    step >= warmup+ramp:     alpha=alpha_target (full blend)

    Smoothstep avoids the control discontinuity of a step function at step=warmup.
    """
    if step < warmup:
        return 0.0
    if step < warmup + ramp:
        r = (step - warmup) / ramp
        r = 3 * r * r - 2 * r * r * r          # smoothstep
        return alpha_target * r
    return alpha_target


def compute_trust_factor(yr_diff_raw: float, cap: float = 0.3) -> float:
    """
    ADDED 2026-08-14 -- 'adaptive' Hybrid-A mode.

    Why: theta_delta = cumsum(alpha_effective * yr_diff * dt). Measured
    yr_diff has a persistent nonzero MEAN bias (~-0.2 to -0.3 rad/s across
    every V2 run tested), not just noise -- so for ANY constant alpha>0,
    drift grows roughly linearly with time regardless of how small alpha
    is; it's only a question of how many seconds until it saturates. This
    is why even alpha=0.1 already blew past the 0.5 rad fail threshold
    within 60s (see the V2 lab notes (not in this repo) fine-alpha-sweep section).

    A fixed linear blend can't fix this -- it applies the same untrustworthy
    correction every step regardless of whether MS-NN and GRU actually
    agree. This trust factor instead scales the blend DOWN when |yr_diff|
    is large (models disagree -> don't trust the correction) and lets it
    through closer to full strength only when |yr_diff| is small (models
    agree -> correction is plausible). linear ramp from trust=1 at
    yr_diff=0 to trust=0 at |yr_diff|>=cap.
    """
    return float(np.clip(1.0 - abs(yr_diff_raw) / cap, 0.0, 1.0))


# ── Reference Path Extraction ─────────────────────────────────────────────────

def _filter_yr(yr_series: np.ndarray, mode: str, clip_thresh: float = 3.0,
               median_window: int = 5) -> np.ndarray:
    """
    Filter yaw-rate series to remove spikes before reference path integration.

    Training data shows: mocap_angular_velocity p99=2.611, p99.9=5.194 rad/s.
    A 5.47 rad/s spike (max |IMU-MoCap| diff) would create a path fold if unfiltered.
    Clip at 3.0 rad/s covers >99% of real maneuvers (H2 recommendation).

    mode: 'none'         no filtering (raw values)
          'clip'         hard clip to [-clip_thresh, +clip_thresh]
          'median'       rolling median (window=median_window)
          'median_clip'  rolling median then clip (recommended for report)
    """
    out = yr_series.copy().astype(float)
    if mode in ('median', 'median_clip'):
        s = pd.Series(out)
        out = s.rolling(median_window, center=True, min_periods=1).median().to_numpy()
    if mode in ('clip', 'median_clip'):
        n_clipped = int(np.sum(np.abs(out) > clip_thresh))
        if n_clipped > 0:
            print(f"[ref] yr filter '{mode}': clipped {n_clipped} samples "
                  f"(max raw={np.abs(out).max():.2f} → clipped to ±{clip_thresh})")
        out = np.clip(out, -clip_thresh, clip_thresh)
    return out


def extract_reference(csv_path, dt=0.02, source='imu', yr_filter='none'):
    """
    Integrate CSV → (x, y) waypoints.

    source='imu'   (default): uses odom_speed + IMU_angular_velocity_z
                   Matches the standard reference path used in V1 evaluation.
    source='mocap': uses v_x_local/v_y_local + mocap_angular_velocity
                   More consistent with GRU training columns; recommended for warm start.

    yr_filter: 'none' | 'clip' | 'median' | 'median_clip'
               Filters yr before integration to remove sensor spikes.
               Training data: mocap yr p99=2.611 rad/s, max observed 5.47.
               'median_clip' (clip at 3.0 rad/s) recommended for report-quality reference.

    B2: IMU vs MoCap mean yr diff = -0.0077 rad/s (OK), but max|diff| = 5.47 (spikes).
        Use yr_filter='median_clip' with source='mocap' for warm-start evaluation.
    """
    df = pd.read_csv(csv_path)

    if source == 'mocap':
        yr_col = 'mocap_angular_velocity' if 'mocap_angular_velocity' in df.columns \
                 else 'IMU_angular_velocity_z'
        yr_raw = df.get(yr_col, pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy()
        yr = _filter_yr(yr_raw, yr_filter)
        vx_arr = df.get('v_x_local', df.get('odom_speed', pd.Series(np.zeros(len(df))))).fillna(0.0).to_numpy()
        vy_arr = df.get('v_y_local', pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy()
        x, y, theta = 0.0, 0.0, 0.0
        path = []
        for i in range(len(df)):
            path.append([x, y])
            ct, st = np.cos(theta), np.sin(theta)
            x     += (float(vx_arr[i]) * ct - float(vy_arr[i]) * st) * dt
            y     += (float(vx_arr[i]) * st + float(vy_arr[i]) * ct) * dt
            theta += float(yr[i]) * dt
    else:                                   # 'imu'
        yr_col = 'IMU_angular_velocity_z' if 'IMU_angular_velocity_z' in df.columns \
                 else 'mocap_angular_velocity'
        yr_raw = df.get(yr_col, pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy()
        yr = _filter_yr(yr_raw, yr_filter)
        vx_arr = df.get('odom_speed', df.get('v_x_local', pd.Series(np.zeros(len(df))))).fillna(0.0).to_numpy()
        x, y, theta = 0.0, 0.0, 0.0
        path = []
        for i in range(len(df)):
            path.append([x, y])
            x     += float(vx_arr[i]) * np.cos(theta) * dt
            y     += float(vx_arr[i]) * np.sin(theta) * dt
            theta += float(yr[i]) * dt

    return np.array(path)


# ── Main Simulation Loop ───────────────────────────────────────────────────────

def run_v2(gru: GRUInference, ref_path: np.ndarray,
           controller: PurePursuit, integrator: StateIntegrator,
           n_steps: int,
           msnn: 'MSNNIPCClient | None' = None,
           alpha: float = 0.0,
           warm_init: 'dict | None' = None,
           msnn_acc_source: str = 'gru',
           hybrid_mode: str = 'linear',
           yr_diff_cap: float = 0.3,
           live_yr_filter: 'LiveYrFilter | None' = None) -> pd.DataFrame:
    """
    alpha: MS-NN ψ̇ blend weight for Hybrid Plant mode (M1).
      0.0 = pure GRU      (GRU yr drives position integration)
      1.0 = pure MS-NN yr (MS-NN yr drives position integration)
      0.5 = equal blend
    GRU internal state always uses its own yr prediction (never MS-NN yr),
    to preserve GRU's internal consistency (Hybrid-A scheme).

    hybrid_mode: 'linear' | 'adaptive' (ADDED 2026-08-14)
      linear:   yr_for_integration = alpha*yr_msnn + (1-alpha)*yr_gru every
                step, regardless of how much MS-NN and GRU disagree. This is
                the ORIGINAL scheme -- confirmed by the fine alpha sweep to
                blow past the 0.5 rad drift threshold almost immediately for
                ANY alpha>0, because yr_diff has a persistent nonzero mean
                bias, not just noise, so the drift integral grows with time
                regardless of how small alpha is.
      adaptive: additionally scales alpha by compute_trust_factor(yr_diff),
                so the correction is only let through near full strength
                when MS-NN and GRU actually agree (|yr_diff| small), and is
                suppressed toward 0 when they disagree strongly. Targets the
                actual mechanism behind the drift, rather than just picking
                a smaller fixed alpha.
    yr_diff_cap: |yr_diff| (rad/s) at which adaptive trust reaches 0.

    warm_init: dict from build_warm_init(); pre-fills GRU history from CSV.
    msnn_acc_source: 'gru' | 'zero' | 'kinematic'
      gru:       acc=GRU pred ax, accy=GRU pred ay (default, V2 deployment)
      zero:      acc=0, accy=0 (ablation: removes acceleration input)
      kinematic: acc=d(vx)/dt, accy=prev_vx*prev_yr (physics approximation)
    live_yr_filter: optional LiveYrFilter (ADDED 2026-08-25). Applied to
      yr_for_integration (the value StateIntegrator.update() actually uses)
      after the Hybrid blend, every step. None / mode='none' = passthrough,
      byte-identical to pre-2026-08-25 behavior. Does not affect GRU's own
      recurrent state, theta_gru_only, or yr_diff -- all still computed from
      unfiltered pred['yr']/msnn_yr. See LiveYrFilter docstring above.
    """
    if alpha > 0.0 and msnn is None:
        print(f"[V2] WARNING: --alpha={alpha} requires --msnn-queue. Falling back to alpha=0.0")
        alpha = 0.0

    mode_str = {0.0: 'pure GRU'}.get(alpha,
               'pure MS-NN yr' if alpha == 1.0 else f'hybrid α={alpha}')
    print(f"\n[V2] Mode: {mode_str}  | msnn_acc_source={msnn_acc_source}")
    print(f"[V2] Simulating {n_steps} steps ({n_steps * 0.02:.1f}s) ...")
    t0 = time.time()

    # Apply warm start: pre-fill GRU history and set initial integrator position
    if warm_init is not None:
        for row in warm_init['history']:
            gru._buf.append(row)
        gru_state = dict(warm_init['gru_state'])
        integrator.reset(warm_init['x0'], warm_init['y0'], warm_init['theta0'])
        controller._last_closest = warm_init['path_offset']
        path_offset = warm_init['path_offset']
        print(f"[V2] Warm start: GRU history pre-filled from CSV rows 0-9")
        print(f"[V2]   Initial pos x={warm_init['x0']:.2f}  y={warm_init['y0']:.2f}"
              f"  θ={warm_init['theta0']:.3f}")
    else:
        gru_state  = {'vx': 0.0, 'vy': 0.0, 'yr': 0.0, 'ax': 0.0, 'ay': 0.0}
        path_offset = 0

    prev_vx = gru_state['vx']
    records  = []

    # Track theta_gru_only in parallel: integrates only yr_gru (never yr_msnn).
    # theta_delta = theta_integrated - theta_gru_only reveals Hybrid-A heading drift.
    # If alpha=0: theta_delta ≡ 0 always. If alpha>0: delta grows with yr_msnn bias.
    theta_gru_only = integrator.theta

    for step in range(min(n_steps, len(ref_path) - 1 - path_offset)):
        t = step * integrator.dt
        x, y, theta = integrator.x, integrator.y, integrator.theta

        # --- Pure Pursuit → control commands ---
        steering_cmd, speed_cmd = controller.compute(x, y, theta, ref_path)

        # --- GRU: predict next state ---
        pred = gru.step(
            vx=gru_state['vx'], vy=gru_state['vy'], yr=gru_state['yr'],
            ax=gru_state['ax'], ay=gru_state['ay'],
            speed_cmd=speed_cmd, steering_cmd=steering_cmd,
        )

        # --- MS-NN ψ̇ (used for Hybrid Plant or comparison) ---
        # msnn_acc_source ablation: choose what acceleration we feed MS-NN.
        # crt=0 always: motor current unavailable in Isaac Sim (aₓ output unusable).
        msnn_yr = None
        if msnn is not None:
            if msnn_acc_source == 'zero':
                acc_in, accy_in = 0.0, 0.0
            elif msnn_acc_source == 'kinematic':
                # Physics approximation: use GRU-predicted Δvx/dt and v×ω
                acc_in  = (pred['vx'] - prev_vx) / integrator.dt
                accy_in = gru_state['vx'] * gru_state['yr']
            else:                                          # 'gru' (default)
                acc_in  = gru_state['ax']
                accy_in = gru_state['ay']
            yr_msnn, _ = msnn.step(
                delta=steering_cmd,
                vel=gru_state['vx'],
                acc=acc_in, accy=accy_in, crt=0.0,
            )
            msnn_yr = yr_msnn

        # --- Hybrid ψ̇ with smoothstep ramp after MS-NN warmup ---
        # Q4-Q5: smoothstep ramp avoids step discontinuity in yr_final at warmup end.
        # BUG-V2-08: MS-NN buffer needs 25 steps warmup before predictions are valid.
        alpha_effective = compute_alpha_effective(step, alpha)
        yr_for_integration = pred['yr']
        trust_factor = 1.0  # only meaningful in 'adaptive' mode; logged either way
        if msnn_yr is not None and alpha_effective > 0.0:
            if hybrid_mode == 'adaptive':
                trust_factor = compute_trust_factor(msnn_yr - pred['yr'], cap=yr_diff_cap)
                alpha_effective = alpha_effective * trust_factor
            yr_for_integration = alpha_effective * msnn_yr + (1.0 - alpha_effective) * pred['yr']

        # --- Live ψ̇ filter (ADDED 2026-08-25, opt-in via --live-yr-filter) ---
        # Applied AFTER the Hybrid blend, directly on the value that will
        # drive theta_integrated this step. Does not touch gru_state,
        # theta_gru_only, or yr_diff (see run_v2() docstring).
        if live_yr_filter is not None:
            yr_for_integration = live_yr_filter.apply(yr_for_integration)

        # --- GRU internal state: always use GRU's own yr (Hybrid-A scheme) ---
        prev_vx    = gru_state['vx']
        gru_state  = pred

        # --- Integrate position using (possibly hybrid, possibly live-filtered) yr ---
        x_new, y_new, _ = integrator.update(pred['vx'], pred['vy'], yr_for_integration)

        # --- Track theta_gru_only in parallel (always uses yr_gru only) ---
        theta_gru_only += pred['yr'] * integrator.dt

        # --- Metrics ---
        dists_all = np.hypot(ref_path[:, 0] - x_new, ref_path[:, 1] - y_new)
        closest_i = int(np.argmin(dists_all))
        tracking_error = float(dists_all[closest_i])

        # Heading error: angle between vehicle heading and reference tangent
        if closest_i + 1 < len(ref_path):
            ref_tangent = ref_path[closest_i + 1] - ref_path[closest_i]
            ref_heading = np.arctan2(ref_tangent[1], ref_tangent[0])
        else:
            ref_heading = theta
        heading_error = float(abs((integrator.theta - ref_heading + np.pi) % (2*np.pi) - np.pi))

        # theta_delta: heading divergence between hybrid integrator and GRU-only integrator
        # Measures Hybrid-A internal state inconsistency. theta_delta ≡ 0 when alpha=0.
        theta_delta = float(((integrator.theta - theta_gru_only) + np.pi) % (2 * np.pi) - np.pi)

        yr_diff = float(msnn_yr - pred['yr']) if msnn_yr is not None else 0.0

        rec = dict(
            t=t, x=x_new, y=y_new,
            vx_pred=pred['vx'], vy_pred=pred['vy'],
            yr_gru=pred['yr'], yr_integrated=yr_for_integration,
            ax_pred=pred['ax'], ay_pred=pred['ay'],
            steering_cmd=steering_cmd, speed_cmd=speed_cmd,
            tracking_error=tracking_error,
            heading_error=heading_error,
            speed_error=speed_cmd - pred['vx'],
            alpha_effective=alpha_effective,
            trust_factor=trust_factor,
            closest_idx=controller.closest_idx,
            target_idx=controller.target_idx,
            ld_effective=controller.ld_effective,
            theta_integrated=integrator.theta,
            theta_gru_only=theta_gru_only,
            theta_delta=theta_delta,
            yr_diff=yr_diff,
            steering_clipped=int(controller.steering_clipped),
            speed_clipped=int(controller.speed_clipped),
        )
        if msnn_yr is not None:
            rec['yr_msnn'] = msnn_yr
        records.append(rec)

        if step % 250 == 0:
            yr_disp = f"{yr_for_integration:+6.3f}"
            print(f"  t={t:6.1f}s  x={x_new:+6.2f}  y={y_new:+6.2f}  "
                  f"vx={pred['vx']:+5.2f}  yr={yr_disp}  "
                  f"err={tracking_error:.3f}m  δ={steering_cmd:+.3f}")

    print(f"[V2] Done in {time.time()-t0:.1f}s")
    return pd.DataFrame(records)


# ── Output: CSV + Plot ────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, ref_path: np.ndarray, out_dir: str,
                 args_alpha: float = 0.0, skip_startup_s: float = 0.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / 'v2_trajectory.csv'
    df.to_csv(csv_path, index=False)
    print(f"[V2] CSV  → {csv_path}")

    has_msnn = 'yr_msnn' in df.columns
    is_hybrid = has_msnn and df['yr_integrated'].equals(df['yr_gru']) is False
    n_rows   = 3 if has_msnn else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    fig.suptitle('Vehicle Dynamics V2 — Closed-Loop Control (GRU Plant Model)', fontsize=13)

    # Row 0 left: top-view trajectory
    ax = axes[0, 0]
    ax.plot(ref_path[:, 0], ref_path[:, 1], 'k--', lw=1, alpha=0.5, label='Reference Path')
    ax.plot(df['x'], df['y'], 'b-', lw=2, label='GRU V2 Path')
    ax.scatter([0], [0], c='green', s=100, zorder=5, label='Start')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Top View: Reference vs V2 Path')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 0 right: tracking error
    ax = axes[0, 1]
    ax.plot(df['t'], df['tracking_error'], 'r-', lw=1.5)
    df_eval = df[df['t'] >= skip_startup_s] if skip_startup_s > 0 else df
    mean_err = df_eval['tracking_error'].mean()
    p95_err  = df_eval['tracking_error'].quantile(0.95)
    ax.axhline(mean_err, color='r', ls='--', alpha=0.6, label=f'Mean {mean_err:.3f} m')
    ax.axhline(p95_err,  color='orange', ls=':', alpha=0.6, label=f'P95 {p95_err:.3f} m')
    if skip_startup_s > 0:
        ax.axvline(skip_startup_s, color='gray', ls=':', lw=1.5,
                   label=f'skip {skip_startup_s:.0f}s startup')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Tracking Error [m]')
    ax.set_title('Path Tracking Error vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 left: speed
    ax = axes[1, 0]
    ax.plot(df['t'], df['vx_pred'],  label='vx_pred (GRU) [m/s]')
    ax.plot(df['t'], df['speed_cmd'], '--', alpha=0.7, label='speed_cmd (PP) [m/s]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[m/s]')
    ax.set_title('Speed: Commanded vs GRU Predicted')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 right: yaw rate
    ax = axes[1, 1]
    ax.plot(df['t'], df['yr_gru'],        label='ψ̇ GRU [rad/s]', color='green')
    ax.plot(df['t'], df['yr_integrated'], label='ψ̇ integrated [rad/s]',
            color='blue', lw=2, ls='--' if is_hybrid else '-')
    if has_msnn:
        ax.plot(df['t'], df['yr_msnn'], color='orange', alpha=0.8, lw=1,
                label='ψ̇ MS-NN [rad/s]')
    ax.plot(df['t'], df['steering_cmd'] * 4, ':', alpha=0.3, color='gray', label='δ×4')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('[rad/s]')
    mode_lbl = 'Hybrid' if is_hybrid else ('MS-NN' if has_msnn else 'GRU only')
    ax.set_title(f'Yaw Rate — {mode_lbl}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if has_msnn:
        diff_gru_msnn  = df['yr_gru']  - df['yr_msnn']
        diff_int_msnn  = df['yr_integrated'] - df['yr_msnn']
        rmse_gru  = float(np.sqrt((diff_gru_msnn ** 2).mean()))
        rmse_int  = float(np.sqrt((diff_int_msnn ** 2).mean()))

        ax = axes[2, 0]
        ax.plot(df['t'], diff_gru_msnn,  color='green',  lw=1, label=f'GRU−MS-NN (RMSE={rmse_gru:.4f})')
        if is_hybrid:
            ax.plot(df['t'], diff_int_msnn, color='blue', lw=1.5, ls='--',
                    label=f'Hybrid−MS-NN (RMSE={rmse_int:.4f})')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Δψ̇ [rad/s]')
        ax.set_title('ψ̇ Difference vs MS-NN')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        ax.hist(np.abs(diff_gru_msnn.values), bins=50, color='green',
                alpha=0.6, edgecolor='none', label='|GRU−MS-NN|')
        if is_hybrid:
            ax.hist(np.abs(diff_int_msnn.values), bins=50, color='blue',
                    alpha=0.6, edgecolor='none', label='|Hybrid−MS-NN|')
        ax.set_xlabel('|Δψ̇| [rad/s]')
        ax.set_title('ψ̇ Difference Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out / 'v2_trajectory.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[V2] Plot → {png_path}")

    # Console summary
    print(f"\n{'='*62}")
    print(f" V2 RESULTS SUMMARY")
    print(f"{'='*62}")
    path_len = float(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2).sum())
    ref_len  = float(np.sqrt(np.diff(ref_path[:, 0])**2 + np.diff(ref_path[:, 1])**2).sum())
    path_ratio = path_len / ref_len if ref_len > 0 else float('nan')
    print(f"  Duration       : {df['t'].max():.1f} s  ({len(df)} steps)")
    print(f"  Path driven    : {path_len:.1f} m  /  ref={ref_len:.1f} m"
          f"  ratio={path_ratio:.3f}"
          f"  {'⚠ CUT BEND (<0.85)' if path_ratio < 0.85 else ''}"
          f"  {'⚠ OVERSHOOT (>1.20)' if path_ratio > 1.20 else ''}")

    # Full metrics (includes startup transient)
    print(f"  Tracking error [FULL] :"
          f"  mean={df['tracking_error'].mean():.3f} m"
          f"  p95={df['tracking_error'].quantile(0.95):.3f} m"
          f"  max={df['tracking_error'].max():.3f} m")

    # Metrics excluding startup transient (more representative of steady-state)
    if skip_startup_s > 0 and len(df_eval) > 0:
        print(f"  Tracking error [t>{skip_startup_s:.0f}s]:"
              f"  mean={mean_err:.3f} m"
              f"  p95={p95_err:.3f} m"
              f"  max={df_eval['tracking_error'].max():.3f} m"
              f"  ← use this for comparison")

    print(f"  GRU vx         : mean={df['vx_pred'].mean():.2f} m/s"
          f"  max={df['vx_pred'].max():.2f} m/s")
    print(f"  GRU ψ̇ RMS      : {np.sqrt((df['yr_gru']**2).mean()):.4f} rad/s")
    print(f"  Heading error  : mean={df['heading_error'].mean():.3f} rad"
          f"  p95={df['heading_error'].quantile(0.95):.3f} rad")

    # Hybrid-A theta drift diagnostics (GPT A1: systematic bias is the key risk)
    if 'theta_delta' in df.columns:
        df_steady = df[df['t'] >= 1.0]  # skip first 1s (warmup+ramp period)
        td = df_steady['theta_delta'].abs() if len(df_steady) > 0 else df['theta_delta'].abs()
        print(f"  θ_delta (|drift|): mean={td.mean():.3f} rad"
              f"  max={td.max():.3f} rad = {np.degrees(td.max()):.1f}°"
              f"  {'⚠ LARGE DRIFT' if td.max() > 0.3 else 'OK'}")

    if 'trust_factor' in df.columns and has_msnn:
        tf = df['trust_factor']
        if tf.min() < 0.999:  # only show if adaptive mode actually did something
            print(f"  trust_factor    : mean={tf.mean():.3f}  min={tf.min():.3f}"
                  f"  (adaptive Hybrid-A: 1.0=full correction trusted, 0.0=suppressed)")

    if 'yr_diff' in df.columns and has_msnn:
        df_steady = df[df['t'] >= 1.0]
        yd = df_steady['yr_diff'] if len(df_steady) > 0 else df['yr_diff']
        print(f"  yr_diff (msnn-gru): mean={yd.mean():.4f} rad/s"
              f"  RMS={np.sqrt((yd**2).mean()):.4f} rad/s"
              f"  {'⚠ BIAS' if abs(yd.mean()) > 0.05 else 'OK'}")

    if 'steering_clipped' in df.columns:
        pct_s = 100 * df['steering_clipped'].mean()
        pct_v = 100 * df['speed_clipped'].mean()
        print(f"  Controller clip : steering {pct_s:.1f}%  speed {pct_v:.1f}%"
              f"  {'⚠ FREQUENT CLIP' if pct_s > 20 else ''}")

    steer_jerk = float(np.abs(np.diff(df['steering_cmd'].values)).mean())
    print(f"  Steering jerk  : mean|Δδ|={steer_jerk:.4f} rad/step"
          f"  {'⚠ HIGH JERK' if steer_jerk > 0.05 else 'OK'}")

    # Progress monotonicity metrics (GPT L2: catch divergence and PP backtracking)
    if 'closest_idx' in df.columns:
        idx_diff = np.diff(df['closest_idx'].values.astype(float))
        backsteps = int(np.sum(idx_diff < -2))    # ignore ±2 noise
        max_back  = int(abs(min(0, idx_diff.min()))) if len(idx_diff) > 0 else 0
        progress_frac = float(df['closest_idx'].iloc[-1]) / max(1, len(ref_path) - 1)
        if progress_frac >= 0.80:
            prog_status = 'OK'
        elif progress_frac >= 0.50:
            prog_status = 'WARN_PROGRESS'
        else:
            prog_status = '⚠ DIVERGED_PROGRESS'
        print(f"  Progress       : fraction={progress_frac:.3f}"
              f"  backsteps={backsteps}  max_back={max_back}"
              f"  → {prog_status}")

    if has_msnn:
        d_gru  = df['yr_gru']        - df['yr_msnn']
        d_int  = df['yr_integrated'] - df['yr_msnn']
        print(f"  GRU−MS-NN ψ̇    : RMSE={np.sqrt((d_gru**2).mean()):.4f} rad/s")
        if is_hybrid:
            print(f"  Hybrid−MS-NN ψ̇ : RMSE={np.sqrt((d_int**2).mean()):.4f} rad/s"
                  f"  (α={args_alpha:.1f})")
    print(f"{'='*62}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Isaac Sim V2 — Closed-Loop GRU Control Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--gru', required=True,
                        dest='gru_checkpoint',
                        help='GRU checkpoint (.pth)')
    parser.add_argument('--ref-csv', required=True,
                        help='CSV to extract reference path from '
                             '(use Day3 training data, NOT Day1)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Simulation duration [s] (default: 60)')
    parser.add_argument('--lookahead', type=float, default=0.5,
                        help='Pure Pursuit lookahead distance [m] (default: 0.5). '
                             'Used as-is unless --adaptive-lookahead is set, in which case '
                             'this is the BASE value the adaptive formula scales from.')
    parser.add_argument('--adaptive-lookahead', action='store_true',
                        help='ADDED 2026-08-15. Scale --lookahead down in sharp curves '
                             '(reduces corner-cutting) and let it relax back toward the '
                             'base value on straights, using the same local curvature '
                             'estimate already computed for speed adaptation. Default OFF '
                             '(backward-compatible, byte-identical to fixed lookahead). '
                             'Motivated by: a single fixed lookahead cannot be optimal on '
                             'every track (track1 wants ~0.3, track2 wants ~1.0-1.3; using '
                             "track1's own optimum on track2 gives ~99pct steering "
                             'saturation) -- reacting to LOCAL curvature instead of a '
                             'whole-track average may get closer to each section\'s own '
                             'optimum without per-track configs.')
    parser.add_argument('--ld-min', type=float, default=0.3,
                        help='--adaptive-lookahead only: floor for the adapted lookahead [m] '
                             '(default: 0.3)')
    parser.add_argument('--ld-max', type=float, default=2.0,
                        help='--adaptive-lookahead only: ceiling for the adapted lookahead [m] '
                             '(default: 2.0)')
    parser.add_argument('--ld-curve-k', type=float, default=3.0,
                        help='--adaptive-lookahead only: how aggressively lookahead shrinks '
                             'with curvature -- ld_eff = ld / (1 + k*curvature). '
                             'Higher = more aggressive shrinkage in curves. (default: 3.0)')
    parser.add_argument('--target-speed', type=float, default=1.5,
                        help='Target driving speed [m/s] (default: 1.5)')
    parser.add_argument('--msnn-queue', default=None,
                        help='MS-NN IPC queue dir (optional); '
                             'start 05_msnn_dynamics_server.py first')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Hybrid Plant blend weight (0=pure GRU, 1=pure MS-NN yr).'
                             ' Requires --msnn-queue. (default: 0.0)')
    parser.add_argument('--output-dir', default='v2_output',
                        help='Output directory (default: v2_output)')
    parser.add_argument('--init-mode', choices=['cold', 'warm'], default='cold',
                        help='cold: start from rest (x=y=theta=vx=0, GRU history zeros). '
                             'warm: pre-fill GRU history from first 10 CSV rows and '
                             'initialize position/state to avoid startup transient. '
                             '(default: cold)')
    parser.add_argument('--warm-start-skip-s', type=float, default=0.0,
                        help='--init-mode warm only: additional seconds into the CSV to skip '
                             'before starting, on top of the fixed 10-step/200ms GRU history '
                             'window (history is always the 10 rows immediately before the '
                             'start index). Use to skip a near-stationary opening segment in '
                             'the source recording (e.g. track2/122000 sits near-still for the '
                             'first ~20s, which otherwise lands the car inside a dense/'
                             'overlapping cluster of reference points and causes spin-in-place). '
                             '(default: 0.0)')
    parser.add_argument('--skip-startup-s', type=float, default=2.0,
                        help='Seconds of startup transient to exclude from reported metrics. '
                             'Does not affect simulation, only summary output. (default: 2.0)')
    parser.add_argument('--msnn-acc-source', choices=['gru', 'zero', 'kinematic'],
                        default='gru',
                        help='Acceleration input to MS-NN for ablation study: '
                             'gru=GRU predicted ax/ay (default, V2 deployment); '
                             'zero=acc=0 (isolate ψ̇ dependence on acc); '
                             'kinematic=d(vx)/dt and vx*yr approximation. '
                             'Only affects ψ̇ quality, not GRU plant behavior.')
    parser.add_argument('--hybrid-mode', choices=['linear', 'adaptive'], default='linear',
                        help='ADDED 2026-08-14. linear=original fixed-alpha blend every step '
                             '(default, backward-compatible; confirmed to drift past the 0.5rad '
                             'fail threshold almost immediately for any alpha>0). '
                             'adaptive=scale alpha down when MS-NN and GRU disagree strongly '
                             '(|yr_diff| large), only trust the correction near full strength '
                             'when they actually agree. See compute_trust_factor().')
    parser.add_argument('--yr-diff-cap', type=float, default=0.3,
                        help='--hybrid-mode adaptive only: |yr_diff| (rad/s) at which trust '
                             'reaches 0. (default: 0.3)')
    parser.add_argument('--ref-source', choices=['imu', 'mocap'], default='imu',
                        help='Column source for reference path integration: '
                             'imu=odom_speed+IMU_angular_velocity_z (default, matches V1); '
                             'mocap=v_x_local/v_y_local+mocap_angular_velocity '
                             '(matches GRU training frame; recommended with --init-mode warm '
                             'to avoid coordinate-frame mismatch, see B2).')
    parser.add_argument('--yr-filter', choices=['none', 'clip', 'median', 'median_clip'],
                        default='none',
                        help='Yaw-rate filter applied before reference path integration. '
                             'Removes sensor spikes that fold the reference path. '
                             'Training data: mocap yr p99=2.611 rad/s, max spike=5.47 rad/s. '
                             'none=raw (default, backward-compatible); '
                             'clip=hard clip at 3.0 rad/s; '
                             'median=rolling median window=5; '
                             'median_clip=median then clip (recommended for report baseline). '
                             'Only affects reference path geometry, not GRU predictions.')
    parser.add_argument('--live-yr-filter', choices=['none', 'ema', 'median'],
                        default='none',
                        help='ADDED 2026-08-25. Causal filter applied to the LIVE '
                             'per-step yaw rate that actually drives theta_integrated '
                             '(after the Hybrid blend, inside the closed loop) -- '
                             'unlike --yr-filter above, which only smooths the offline '
                             'reference-path reconstruction. Investigated for Felix\'s '
                             '"heading looks a bit off on straights" feedback (report/'
                             'Felix_new_task_list_20260825.txt item 8, not in this repo). '
                             'none=passthrough (default, backward-compatible, does NOT '
                             'change any currently-deployed run); '
                             'ema=exponential moving average (see --live-yr-filter-alpha); '
                             'median=causal rolling median (see --live-yr-filter-window).')
    parser.add_argument('--live-yr-filter-window', type=int, default=5,
                        help='--live-yr-filter median only: rolling window size in steps '
                             '(causal, i.e. current + past window-1 samples). (default: 5)')
    parser.add_argument('--live-yr-filter-alpha', type=float, default=0.3,
                        help='--live-yr-filter ema only: smoothing factor in (0, 1]. '
                             'Smaller = smoother/more lag, larger = closer to passthrough. '
                             '(default: 0.3)')
    parser.add_argument('--live-yr-filter-gate', type=float, default=0.0,
                        help='2026-08-26: only apply --live-yr-filter while |raw yr| is '
                             'below this (rad/s) -- i.e. only smooth near-straight driving, '
                             'pass real cornering yr through unfiltered and reset filter '
                             'state (no re-entry lag). 0.0 (default) = ungated, matches '
                             'original --live-yr-filter behavior. Motivation: ungated ema '
                             '(alpha=0.15) cut arrow noise on straights but nearly doubled '
                             'theta_delta drift by lagging the signal in corners too.')
    args = parser.parse_args()

    print('=' * 62)
    print(' Isaac Sim V2 — Closed-Loop Control Demo')
    print('=' * 62)
    print(f'  GRU checkpoint : {args.gru_checkpoint}')
    print(f'  Reference CSV  : {args.ref_csv}')
    print(f'  Duration       : {args.duration} s')
    print(f'  Lookahead      : {args.lookahead} m'
          + (f'  (adaptive: ld_min={args.ld_min} ld_max={args.ld_max} k={args.ld_curve_k})'
             if args.adaptive_lookahead else '  (fixed)'))
    print(f'  Target speed   : {args.target_speed} m/s')
    print(f'  MS-NN IPC      : {args.msnn_queue or "disabled"}')
    print(f'  Hybrid alpha   : {args.alpha}  (0=GRU, 1=MS-NN yr)')
    print(f'  Hybrid mode    : {args.hybrid_mode}'
          + (f'  (yr_diff_cap={args.yr_diff_cap})' if args.hybrid_mode == 'adaptive' else ''))
    print(f'  Init mode      : {args.init_mode}')
    print(f'  Skip startup   : {args.skip_startup_s} s')
    print(f'  MS-NN acc src  : {args.msnn_acc_source}')

    # P2-5: Guard against Felix's data constraints being violated at runtime
    ref_csv_str = str(args.ref_csv)
    if '20260721' in ref_csv_str:
        raise SystemExit(
            "[ERROR] --ref-csv contains Day1 data (session_20260721).\n"
            "  Day1 is TEST ONLY — never use as reference path (Felix constraint).\n"
            "  Use: session_20260728_163324_seg1_20ms.csv  (Day3)"
        )
    if '20260728_124427' in ref_csv_str:
        raise SystemExit(
            "[ERROR] --ref-csv contains excluded file (session_20260728_124427).\n"
            "  This file has 100% NaN IMU columns and is excluded by Felix.\n"
            "  Use: session_20260728_163324_seg1_20ms.csv  (Day3)"
        )

    print(f'  Ref source     : {args.ref_source}  (imu=IMU+odom, mocap=GRU-frame columns)')
    print(f'  yr filter      : {args.yr_filter}  (clip@3.0 rad/s; training p99=2.611)')
    live_filter_desc = args.live_yr_filter
    if args.live_yr_filter == 'median':
        live_filter_desc += f' (window={args.live_yr_filter_window})'
    elif args.live_yr_filter == 'ema':
        live_filter_desc += f' (alpha={args.live_yr_filter_alpha})'
    print(f'  live yr filter : {live_filter_desc}  (applied to closed-loop integration yr; '
          f'default none = no behavior change)')

    gru        = GRUInference(args.gru_checkpoint)
    df_ref_csv = pd.read_csv(args.ref_csv)
    ref_path   = extract_reference(args.ref_csv, source=args.ref_source,
                                   yr_filter=args.yr_filter)
    controller = PurePursuit(lookahead=args.lookahead,
                             target_speed=args.target_speed,
                             adaptive_lookahead=args.adaptive_lookahead,
                             ld_min=args.ld_min, ld_max=args.ld_max,
                             ld_curve_k=args.ld_curve_k)
    integrator = StateIntegrator(dt=0.02)

    # Build warm_init dict if requested (Q15: pre-fill GRU history from CSV)
    warm_init = None
    if args.init_mode == 'warm':
        skip_steps = int(round(args.warm_start_skip_s / 0.02))
        HIST = GRUInference.HISTORY     # 10 steps = 200ms lookback
        N = HIST + skip_steps           # start index; history = the HIST rows right before it
        needed_cols = _EXPECTED_INPUT_COLS[:5]  # v_x_local...a_y_local
        cmd_cols    = ['speed_cmd', 'steering_cmd']
        missing = [c for c in needed_cols + cmd_cols if c not in df_ref_csv.columns]
        if missing:
            print(f"[V2] WARNING: warm start skipped — CSV missing columns: {missing}")
        elif len(df_ref_csv) <= N:
            print(f"[V2] WARNING: warm start skipped — CSV too short ({len(df_ref_csv)} rows)")
        else:
            if skip_steps > 0:
                print(f'[V2] Warm start skip: {args.warm_start_skip_s:.1f}s '
                      f'({skip_steps} steps) — starting at CSV row {N}')
            history_rows = []
            for i in range(N - HIST, N):
                row = df_ref_csv.iloc[i]
                history_rows.append([
                    float(row['v_x_local']),  float(row['v_y_local']),
                    float(row['mocap_angular_velocity']),
                    float(row['a_x_local']),  float(row['a_y_local']),
                    float(row['speed_cmd']),  float(row['steering_cmd']),
                ])
            init_row = df_ref_csv.iloc[N]
            # Integrator starts at the integrated position corresponding to CSV row N
            x0, y0 = float(ref_path[N, 0]), float(ref_path[N, 1])
            if N > 0:
                dp     = ref_path[N] - ref_path[N - 1]
                theta0 = float(np.arctan2(dp[1], dp[0]))
            else:
                theta0 = 0.0
            warm_init = {
                'history':     history_rows,
                'gru_state': {
                    'vx': float(init_row['v_x_local']),
                    'vy': float(init_row['v_y_local']),
                    'yr': float(init_row['mocap_angular_velocity']),
                    'ax': float(init_row['a_x_local']),
                    'ay': float(init_row['a_y_local']),
                },
                'x0':          x0,
                'y0':          y0,
                'theta0':      theta0,
                'path_offset': N,
            }

    msnn = None
    if args.msnn_queue:
        try:
            msnn = MSNNIPCClient(args.msnn_queue)
            msnn.reset()
            print(f'[V2] MS-NN IPC connected: {args.msnn_queue}')
        except Exception as e:
            print(f'[V2] MS-NN IPC unavailable ({e}) — continuing without')

    live_yr_filter = LiveYrFilter(mode=args.live_yr_filter,
                                  window=args.live_yr_filter_window,
                                  alpha=args.live_yr_filter_alpha,
                                  gate_thresh=args.live_yr_filter_gate)

    n_steps = int(args.duration / 0.02)
    df = run_v2(gru, ref_path, controller, integrator, n_steps, msnn,
                alpha=args.alpha, warm_init=warm_init,
                msnn_acc_source=args.msnn_acc_source,
                hybrid_mode=args.hybrid_mode, yr_diff_cap=args.yr_diff_cap,
                live_yr_filter=live_yr_filter)
    save_outputs(df, ref_path, args.output_dir, args_alpha=args.alpha,
                 skip_startup_s=args.skip_startup_s)


if __name__ == '__main__':
    main()
