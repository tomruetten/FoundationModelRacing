"""
09_v2_reference_extract.py — Reference Trajectory Extraction
=============================================================
Integrates ground-truth (odom_speed, IMU_angular_velocity_z) from a
processed CSV to build a (x, y, θ) reference path for V2 Pure Pursuit.

Assumes vy ≈ 0 (kinematic bicycle model, valid for low-slip F1TENTH).

Usage:
  python3 09_v2_reference_extract.py \
    --csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv \
    --output reference_path_day3.csv

Output:
  reference_path_day3.csv        ← waypoints (x, y, θ, vx, yr, speed_cmd, steering_cmd)
  reference_path_day3_plot.png   ← top-view + speed/yr plot
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def extract_reference(csv_path: str, output_path: str = None, dt: float = 0.02):
    df = pd.read_csv(csv_path)
    print(f"[ref] Loaded {len(df)} rows from {csv_path}")

    x, y, theta = 0.0, 0.0, 0.0
    rows = []

    for i, row in df.iterrows():
        vx = float(row['odom_speed'])
        yr = float(row['IMU_angular_velocity_z'])
        t  = i * dt

        rows.append({
            't': t, 'x': x, 'y': y, 'theta': theta,
            'vx': vx, 'yr': yr,
            'speed_cmd':   float(row.get('speed_cmd',   1.5)),
            'steering_cmd': float(row.get('steering_cmd', 0.0)),
        })

        # Euler integration (vy = 0)
        x     += vx * np.cos(theta) * dt
        y     += vx * np.sin(theta) * dt
        theta += yr * dt

    path_df = pd.DataFrame(rows)

    if output_path is None:
        output_path = Path(csv_path).stem + '_reference_path.csv'

    path_df.to_csv(output_path, index=False)

    print(f"[ref] Saved: {output_path}  ({len(path_df)} waypoints, {path_df['t'].max():.1f}s)")
    print(f"      x ∈ [{path_df['x'].min():.2f}, {path_df['x'].max():.2f}] m")
    print(f"      y ∈ [{path_df['y'].min():.2f}, {path_df['y'].max():.2f}] m")
    print(f"      vx ∈ [{path_df['vx'].min():.2f}, {path_df['vx'].max():.2f}] m/s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Reference Trajectory — {Path(csv_path).name}')

    ax = axes[0]
    ax.plot(path_df['x'], path_df['y'], 'b-', linewidth=1.5)
    ax.scatter([0], [0], c='green', s=120, zorder=5, label='Start')
    ax.scatter([path_df['x'].iloc[-1]], [path_df['y'].iloc[-1]],
               c='red', s=120, zorder=5, label='End')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Top View')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(path_df['t'], path_df['vx'],          label='odom_speed [m/s]')
    ax.plot(path_df['t'], path_df['yr'],           label='IMU yr [rad/s]', alpha=0.7)
    ax.plot(path_df['t'], path_df['steering_cmd'], label='steering_cmd [rad]', alpha=0.6)
    ax.set_xlabel('t [s]')
    ax.set_title('Speed / Yaw Rate / Steering')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = str(Path(output_path).with_suffix('')) + '_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ref] Plot: {plot_path}")

    return path_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    required=True,  help='processed_data/*.csv')
    parser.add_argument('--output', default=None,   help='Output CSV path')
    args = parser.parse_args()
    extract_reference(args.csv, args.output)
