"""
12_v2_msnn_plant_demo.py — Isaac Sim V2: MS-NN Alone as the Plant (EXPERIMENTAL)
=================================================================================

10_v2_main_demo.py uses GRU as the plant because GRU outputs the full state
[vx, vy, yr, ax, ay] needed for position integration. MS-NN's IPC protocol
only returns (yr_pred, ax_pred) -- no vx, no vy -- so it was never designed
to drive a closed loop by itself.

This script (added 2026-08-14, per explicit user request "let's actually
try MS-NN alone as a plant") gives MS-NN a minimal vx estimate so it CAN
drive the loop, to see what that looks like:
  - vx is integrated locally: vx[t+1] = clip(vx[t] + ax_msnn*dt, vx_min, vx_max)
    using MS-NN's OWN predicted ax (self-consistent -- same principle
    10_v2_main_demo.py uses when GRU feeds back its own state).
  - vy is assumed 0 (low-slip kinematic-bicycle approximation -- the SAME
    assumption 09_v2_reference_extract.py already uses for this exact
    vehicle/track, not a new invented shortcut).

Caveat (read before trusting the numbers): MS-NN was trained/validated as
a psi-dot corrector fed REAL IMU-measured vel/acc (see V1, dynamics/msnn).
It has never seen a self-integrated, drifting vx as its own "vel" input.
This script tells us what THAT specific combination looks like -- it is
an honest experimental probe, not a claim that this is how MS-NN should be
deployed.

Does not modify 10_v2_main_demo.py or the GRU pipeline at all. Reuses
PurePursuit / StateIntegrator / MSNNIPCClient / extract_reference from it
unchanged (imported via importlib, not copy-pasted, so fixes there
propagate here too).

Usage:
  Terminal 1 (nnodely-env, start first):
    conda run -n nnodely-env python3 05_msnn_dynamics_server.py \\
      --model ../../../dynamics/deployed \\
      --queue-dir /tmp/isaac_msnn_v2
    (NOTE: use the ORIGINAL server at ../../simulation/isaac_sim/05_msnn_dynamics_server.py,
     not the copy bundled in this V2 folder -- see the V2 lab notes (not in this repo).)

  Terminal 2:
    conda activate roboracer
    python3 12_v2_msnn_plant_demo.py \\
      --ref-csv <Day2/Day3 CSV> --msnn-queue /tmp/isaac_msnn_v2 \\
      --duration 60 --target-speed 1.5 --lookahead 0.9

Output (in --output-dir): msnn_plant_trajectory.csv + msnn_plant_plot.png
"""

import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Reuse the tested building blocks from 10_v2_main_demo.py ──────────────────
# (module name starts with a digit, so plain `import` syntax can't be used;
# load it directly from its file path instead)
_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("v2_main_demo", _HERE / "10_v2_main_demo.py")
_v2main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2main)

MSNNIPCClient    = _v2main.MSNNIPCClient
PurePursuit      = _v2main.PurePursuit
StateIntegrator  = _v2main.StateIntegrator
extract_reference = _v2main.extract_reference

DT = 0.02
MSNN_WARMUP_STEPS = 25   # matches compute_alpha_effective's warmup in 10_v2_main_demo.py


# ── MS-NN wrapped as a standalone "plant" (EXPERIMENTAL, see module docstring) ─

class MSNNPlant:
    """
    Turns MS-NN's (yr_pred, ax_pred) IPC output into a full-state plant
    step by locally integrating vx from MS-NN's own ax and assuming vy=0.
    """

    def __init__(self, msnn_client: MSNNIPCClient, vx_init: float = 0.0,
                 vx_min: float = 0.0, vx_max: float = 4.5):
        self.msnn = msnn_client
        self.vx_min, self.vx_max = vx_min, vx_max
        self.vx = float(np.clip(vx_init, vx_min, vx_max))
        self.prev_ax = 0.0

    def reset(self, vx_init: float = 0.0):
        self.vx = float(np.clip(vx_init, self.vx_min, self.vx_max))
        self.prev_ax = 0.0
        self.msnn.reset()

    def step(self, steering_cmd: float, speed_cmd: float) -> dict:
        # Feed MS-NN its own previous state (vel=vx going into this step,
        # acc=ax it predicted last step) -- self-consistent, same principle
        # as GRU feeding back its own state in 10_v2_main_demo.py's run_v2().
        # accy is left at 0.0: vy is assumed 0 throughout (see module docstring),
        # so there is no lateral-accel signal to feed back.
        yr_pred, ax_pred = self.msnn.step(
            delta=steering_cmd, vel=self.vx, acc=self.prev_ax, accy=0.0, crt=0.0)
        self.prev_ax = ax_pred
        self.vx = float(np.clip(self.vx + ax_pred * DT, self.vx_min, self.vx_max))
        return {'vx': self.vx, 'vy': 0.0, 'yr': float(yr_pred), 'ax': float(ax_pred), 'ay': 0.0}


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_msnn_plant(plant: MSNNPlant, ref_path: np.ndarray,
                    controller: PurePursuit, integrator: StateIntegrator,
                    n_steps: int) -> pd.DataFrame:
    print(f"\n[MSNN-Plant] Simulating {n_steps} steps ({n_steps * DT:.1f}s) ...")
    print(f"[MSNN-Plant] EXPERIMENTAL: MS-NN driving the loop alone via a "
          f"self-integrated vx (vx_min={plant.vx_min}, vx_max={plant.vx_max}), vy≡0.")
    t0 = time.time()
    records = []

    for step in range(min(n_steps, len(ref_path) - 1)):
        t = step * integrator.dt
        x, y, theta = integrator.x, integrator.y, integrator.theta

        steering_cmd, speed_cmd = controller.compute(x, y, theta, ref_path)
        pred = plant.step(steering_cmd, speed_cmd)

        x_new, y_new, _ = integrator.update(pred['vx'], pred['vy'], pred['yr'])

        dists_all = np.hypot(ref_path[:, 0] - x_new, ref_path[:, 1] - y_new)
        closest_i = int(np.argmin(dists_all))
        tracking_error = float(dists_all[closest_i])

        if closest_i + 1 < len(ref_path):
            ref_tangent = ref_path[closest_i + 1] - ref_path[closest_i]
            ref_heading = np.arctan2(ref_tangent[1], ref_tangent[0])
        else:
            ref_heading = theta
        heading_error = float(abs((integrator.theta - ref_heading + np.pi) % (2 * np.pi) - np.pi))

        records.append(dict(
            t=t, x=x_new, y=y_new,
            vx_pred=pred['vx'], yr_pred=pred['yr'], ax_pred=pred['ax'],
            steering_cmd=steering_cmd, speed_cmd=speed_cmd,
            tracking_error=tracking_error, heading_error=heading_error,
            speed_error=speed_cmd - pred['vx'],
            closest_idx=controller.closest_idx, target_idx=controller.target_idx,
            steering_clipped=int(controller.steering_clipped),
            speed_clipped=int(controller.speed_clipped),
        ))

        if step % 250 == 0:
            print(f"  t={t:6.1f}s  x={x_new:+6.2f}  y={y_new:+6.2f}  "
                  f"vx={pred['vx']:+5.2f}  yr={pred['yr']:+6.3f}  "
                  f"err={tracking_error:.3f}m  δ={steering_cmd:+.3f}")

    print(f"[MSNN-Plant] Done in {time.time()-t0:.1f}s")
    return pd.DataFrame(records)


def save_outputs(df: pd.DataFrame, ref_path: np.ndarray, out_dir: str, skip_startup_s: float = 2.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / 'msnn_plant_trajectory.csv'
    df.to_csv(csv_path, index=False)
    print(f"[MSNN-Plant] CSV  → {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vehicle Dynamics V2 — MS-NN ALONE as Plant (EXPERIMENTAL, self-integrated vx)',
                 fontsize=12)

    ax = axes[0, 0]
    ax.plot(ref_path[:, 0], ref_path[:, 1], 'k--', lw=1, alpha=0.5, label='Reference Path')
    ax.plot(df['x'], df['y'], 'r-', lw=1.2, label='MS-NN-plant driven')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('Top-view trajectory')
    ax.legend(fontsize=8); ax.axis('equal'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df['t'], df['vx_pred'], label='vx (MS-NN, self-integrated)')
    ax.plot(df['t'], df['speed_cmd'], '--', alpha=0.6, label='speed_cmd')
    ax.set_xlabel('t [s]'); ax.set_ylabel('vx [m/s]'); ax.set_title('Speed tracking')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df['t'], df['yr_pred'], label='yr (MS-NN)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('yaw rate [rad/s]'); ax.set_title('MS-NN yaw rate')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df['t'], df['tracking_error'])
    ax.set_xlabel('t [s]'); ax.set_ylabel('tracking error [m]'); ax.set_title('Tracking error over time')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png_path = out / 'msnn_plant_plot.png'
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"[MSNN-Plant] Plot → {png_path}")

    df_steady = df[df['t'] >= skip_startup_s]
    ref_len = float(np.sum(np.hypot(np.diff(ref_path[:, 0]), np.diff(ref_path[:, 1]))))
    driven_len = float(np.sum(np.hypot(np.diff(df['x']), np.diff(df['y']))))

    print("\n" + "=" * 62)
    print(" MS-NN-alone Plant — Summary")
    print("=" * 62)
    print(f"  Duration            : {df['t'].iloc[-1]:.1f} s")
    print(f"  Path driven / ref   : {driven_len:.1f} / {ref_len:.1f} m "
          f"(ratio={driven_len/ref_len:.3f})")
    print(f"  Tracking error [all]        : mean={df['tracking_error'].mean():.4f} m  "
          f"p95={df['tracking_error'].quantile(0.95):.4f} m")
    print(f"  Tracking error [t>{skip_startup_s:.1f}s]: mean={df_steady['tracking_error'].mean():.4f} m  "
          f"p95={df_steady['tracking_error'].quantile(0.95):.4f} m")
    print(f"  vx (self-integrated): mean={df_steady['vx_pred'].mean():.3f}  "
          f"min={df_steady['vx_pred'].min():.3f}  max={df_steady['vx_pred'].max():.3f} m/s")
    print(f"  yr RMS              : {np.sqrt((df_steady['yr_pred']**2).mean()):.4f} rad/s")
    print(f"  Controller clip     : steering={100*df['steering_clipped'].mean():.1f}%  "
          f"speed={100*df['speed_clipped'].mean():.1f}%")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(
        description='Isaac Sim V2 — MS-NN ALONE as Plant (EXPERIMENTAL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ref-csv', required=True,
                        help='CSV to extract reference path from (Day2/Day3 only, NOT Day1)')
    parser.add_argument('--msnn-queue', required=True,
                        help='MS-NN IPC queue dir; start 05_msnn_dynamics_server.py first '
                             '(use the ORIGINAL at ../../simulation/isaac_sim/05_msnn_dynamics_server.py)')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--lookahead', type=float, default=0.9,
                        help='default matches the value 10_v2_main_demo.py confirmed near-optimal')
    parser.add_argument('--target-speed', type=float, default=1.5)
    parser.add_argument('--vx-init', type=float, default=0.0)
    parser.add_argument('--vx-max', type=float, default=4.5,
                        help='training data p95=4.151 m/s; headroom above that (default: 4.5)')
    parser.add_argument('--output-dir', default='msnn_plant_output')
    parser.add_argument('--skip-startup-s', type=float, default=2.0,
                        help='seconds excluded from summary stats (MS-NN needs '
                             f'{MSNN_WARMUP_STEPS} steps = {MSNN_WARMUP_STEPS*DT:.2f}s warmup)')
    parser.add_argument('--ref-source', choices=['imu', 'mocap'], default='imu')
    parser.add_argument('--yr-filter', choices=['none', 'clip', 'median', 'median_clip'],
                        default='median_clip')
    args = parser.parse_args()

    print('=' * 62)
    print(' Isaac Sim V2 — MS-NN ALONE as Plant (EXPERIMENTAL)')
    print('=' * 62)
    print(f'  Reference CSV  : {args.ref_csv}')
    print(f'  MS-NN IPC      : {args.msnn_queue}')
    print(f'  Duration       : {args.duration} s')
    print(f'  Lookahead      : {args.lookahead} m')
    print(f'  Target speed   : {args.target_speed} m/s')
    print(f'  vx init/max    : {args.vx_init} / {args.vx_max} m/s')

    # Same Felix data-constraint guard as 10_v2_main_demo.py -- do not relax here.
    ref_csv_str = str(args.ref_csv)
    if '20260721' in ref_csv_str:
        raise SystemExit(
            "[ERROR] --ref-csv contains Day1 data (session_20260721).\n"
            "  Day1 is TEST ONLY — never use as reference path (Felix constraint).")
    if '20260728_124427' in ref_csv_str:
        raise SystemExit(
            "[ERROR] --ref-csv contains excluded file (session_20260728_124427).\n"
            "  This file has 100% NaN IMU columns and is excluded by Felix.")

    ref_path = extract_reference(args.ref_csv, source=args.ref_source, yr_filter=args.yr_filter)
    controller = PurePursuit(lookahead=args.lookahead, target_speed=args.target_speed)
    integrator = StateIntegrator(dt=DT)

    msnn = MSNNIPCClient(args.msnn_queue)
    plant = MSNNPlant(msnn, vx_init=args.vx_init, vx_max=args.vx_max)
    plant.reset(vx_init=args.vx_init)

    n_steps = int(args.duration / DT)
    df = run_msnn_plant(plant, ref_path, controller, integrator, n_steps)
    save_outputs(df, ref_path, args.output_dir, skip_startup_s=args.skip_startup_s)


if __name__ == '__main__':
    main()
