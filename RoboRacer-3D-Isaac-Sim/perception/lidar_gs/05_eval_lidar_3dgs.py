"""
eval_raydrop_v7.py — Evaluate v7 model's learned ray-drop predictions.

Compares:
  1. Dense output (no filtering)
  2. Learned ray-drop filtering (sigmoid(colors[:,1]) > threshold)
  3. Alpha-based filtering (same as v6 eval)
  4. Combined: learned ray-drop + alpha
  5. Density-matched to real Livox

Usage:
  conda run -n nerfstudio-py310 python3 scripts/eval_raydrop_v7.py
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from train_lidar_3dgs import (
    PROBE_ROOT, ROOT, BAG, CSYNC,
    T_BASE_MOCAP_t, T_BASE_LIDAR_t, MAX_RANGE_M,
    VCAM_W, VCAM_H, VCAM_FOV_DEG, CUBEMAP_FACES,
    load_mocap, parse_pointcloud2, make_virtual_K, make_viewmat,
    render_depth,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(exp_name):
    exp_dir = PROBE_ROOT / 'experiments' / exp_name
    ckpt_path = exp_dir / 'best.pt'
    if not ckpt_path.exists():
        ckpt_path = exp_dir / 'final.pt'
    print(f"Loading model: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    return {k: torch.nn.Parameter(v.to(DEVICE)) for k, v in ckpt.items()}


def get_eval_frames(n_samples=50):
    ts_f, pos_f, slerp = load_mocap()
    conn = sqlite3.connect(str(BAG))
    cur = conn.cursor()
    cur.execute("""
        SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/livox/lidar' ORDER BY m.timestamp
    """)
    lidar_rows = cur.fetchall()
    conn.close()

    cam_sync = {}
    with open(str(CSYNC)) as f:
        for row in csv.DictReader(f):
            lt = int(row['lidar_timestamp_ns'])
            cam_sync[lt] = row['matched'].strip().lower() == 'true'

    all_data = []
    for ts, data in lidar_rows:
        lidar_ts = float(ts)
        if not cam_sync.get(ts, False):
            continue
        if lidar_ts < ts_f[0] or lidar_ts > ts_f[-1]:
            continue

        i = max(0, min(int(np.searchsorted(ts_f, lidar_ts)) - 1, len(ts_f) - 2))
        alpha_t = (lidar_ts - ts_f[i]) / max(float(ts_f[i+1] - ts_f[i]), 1.0)
        pos_m = (1.0 - alpha_t) * pos_f[i] + alpha_t * pos_f[i+1]
        R_wm = slerp(lidar_ts).as_matrix()

        t_wb = R_wm @ (-T_BASE_MOCAP_t) + pos_m
        R_wb = R_wm
        t_wl = R_wb @ T_BASE_LIDAR_t + t_wb

        xyz_lidar = parse_pointcloud2(bytes(data))
        if len(xyz_lidar) == 0:
            continue
        ranges = np.linalg.norm(xyz_lidar, axis=1)
        xyz_lidar = xyz_lidar[ranges < MAX_RANGE_M]
        xyz_world = (R_wb @ xyz_lidar.T).T + t_wl

        all_data.append({
            'R_wl': R_wb, 't_wl': t_wl,
            'real_pts_world': xyz_world.astype(np.float32),
            'n_real': len(xyz_world),
        })

    indices = np.linspace(0, len(all_data) - 1, n_samples, dtype=int)
    frames = [all_data[i] for i in indices]
    print(f"  Selected {len(frames)} eval frames "
          f"(avg {np.mean([f['n_real'] for f in frames]):.0f} real pts/frame)")
    return frames


def depth_to_world(depth_np, viewmat, K, mask=None):
    if mask is None:
        mask = depth_np > 0.1
    v, u = np.where(mask)
    if len(v) == 0:
        return np.empty((0, 3), dtype=np.float32)
    d = depth_np[v, u]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    cam_pts = np.stack([x, y, d], axis=1)
    R_cw = viewmat[:3, :3]
    t_cw = viewmat[:3, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    return ((R_wc @ cam_pts.T).T + t_wc).astype(np.float32)


def render_scan(splats, R_wl, t_wl, K,
                raydrop_thresh=None, alpha_thresh=None, max_pts=None):
    """Render one scan with optional ray-drop and alpha filtering."""
    all_pts = []
    with torch.no_grad():
        for face_name, R_face in CUBEMAP_FACES.items():
            viewmat = make_viewmat(R_wl, t_wl, R_face)
            depth, colors_img, alpha, _ = render_depth(
                splats, viewmat, K, VCAM_W, VCAM_H, DEVICE)

            depth_np = depth.cpu().numpy()
            alpha_np = alpha.cpu().numpy()

            mask = depth_np > 0.1

            if raydrop_thresh is not None:
                raydrop_prob = torch.sigmoid(colors_img[:, :, 1]).cpu().numpy()
                mask = mask & (raydrop_prob >= raydrop_thresh)

            if alpha_thresh is not None:
                mask = mask & (alpha_np >= alpha_thresh)

            pts = depth_to_world(depth_np, viewmat, K, mask)
            if len(pts) > 0:
                all_pts.append(pts)

    if not all_pts:
        return np.empty((0, 3), dtype=np.float32)

    pts = np.concatenate(all_pts)
    if max_pts and len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


def geometric_eval(synth_pts, real_pts):
    if len(synth_pts) < 10 or len(real_pts) < 10:
        return {}
    tree_real = cKDTree(real_pts)
    tree_synth = cKDTree(synth_pts)
    d_s2r, _ = tree_real.query(synth_pts, k=1)
    d_r2s, _ = tree_synth.query(real_pts, k=1)
    chamfer = (d_s2r.mean() + d_r2s.mean()) / 2
    results = {'chamfer': chamfer, 'n_synth': len(synth_pts), 'n_real': len(real_pts)}
    for t in [0.05, 0.10, 0.20]:
        p = (d_s2r < t).mean()
        r = (d_r2s < t).mean()
        f = 2 * p * r / (p + r + 1e-8)
        results[f'f@{t:.2f}'] = f
    return results


def eval_config(name, splats, frames, K, avg_real_pts,
                raydrop_thresh=None, alpha_thresh=None, density_match=False):
    synth_accum = []
    real_accum = []
    n_pts_list = []
    target = int(avg_real_pts) if density_match else None

    for frame in frames:
        pts = render_scan(splats, frame['R_wl'], frame['t_wl'], K,
                          raydrop_thresh=raydrop_thresh,
                          alpha_thresh=alpha_thresh,
                          max_pts=target)
        synth_accum.append(pts)
        real_accum.append(frame['real_pts_world'])
        n_pts_list.append(len(pts))

    synth_all = np.concatenate(synth_accum)
    real_all = np.concatenate(real_accum)
    metrics = geometric_eval(synth_all, real_all)
    avg_pts = np.mean(n_pts_list)
    ratio = avg_pts / avg_real_pts

    print(f"  {name:<40s} pts={avg_pts:>8.0f} ({ratio:>5.1f}x)  "
          f"Chamfer={metrics.get('chamfer',0):.4f}m  "
          f"F@5={metrics.get('f@0.05',0):.3f}  "
          f"F@10={metrics.get('f@0.10',0):.3f}  "
          f"F@20={metrics.get('f@0.20',0):.3f}")
    return metrics, avg_pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='lidar_3dgs_v7_raydrop')
    parser.add_argument('--n-poses', type=int, default=50)
    args = parser.parse_args()

    out_dir = PROBE_ROOT / 'experiments' / args.exp / 'raydrop_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    splats = load_model(args.exp)
    K = make_virtual_K(VCAM_W, VCAM_H, VCAM_FOV_DEG)

    print("\nGetting eval frames...")
    frames = get_eval_frames(n_samples=args.n_poses)
    avg_real = np.mean([f['n_real'] for f in frames])

    print(f"\n=== Filtering Comparison (v7 model, {len(frames)} frames) ===\n")

    # No filtering
    eval_config("Dense (no filter)", splats, frames, K, avg_real)

    # Alpha-only (same as v6 approach)
    for t in [0.5, 0.9, 0.99]:
        eval_config(f"Alpha >= {t}", splats, frames, K, avg_real, alpha_thresh=t)

    # Learned ray-drop only
    for t in [0.3, 0.5, 0.7, 0.9]:
        eval_config(f"RayDrop >= {t}", splats, frames, K, avg_real, raydrop_thresh=t)

    # Combined
    eval_config("RayDrop>=0.5 + Alpha>=0.9",
                splats, frames, K, avg_real, raydrop_thresh=0.5, alpha_thresh=0.9)
    eval_config("RayDrop>=0.7 + Alpha>=0.9",
                splats, frames, K, avg_real, raydrop_thresh=0.7, alpha_thresh=0.9)

    # Density-matched
    print()
    eval_config(f"Density-matched (~{avg_real:.0f} pts, no filter)",
                splats, frames, K, avg_real, density_match=True)
    eval_config(f"Density-matched + RayDrop>=0.5",
                splats, frames, K, avg_real, raydrop_thresh=0.5, density_match=True)
    eval_config(f"Density-matched + Alpha>=0.9",
                splats, frames, K, avg_real, alpha_thresh=0.9, density_match=True)
    eval_config(f"Density-matched + RD>=0.5 + A>=0.9",
                splats, frames, K, avg_real,
                raydrop_thresh=0.5, alpha_thresh=0.9, density_match=True)

    # Save example PLYs
    print(f"\n=== Saving example PLYs ===")
    ex_dir = out_dir / 'examples'
    ex_dir.mkdir(exist_ok=True)
    idx = len(frames) // 2
    frame = frames[idx]
    real = frame['real_pts_world']

    configs = [
        ('real', None, [1.0, 0.3, 0.2]),
        ('dense', {}, [0.5, 0.5, 0.5]),
        ('raydrop_05', {'raydrop_thresh': 0.5}, [0.2, 0.6, 1.0]),
        ('raydrop_07', {'raydrop_thresh': 0.7}, [0.2, 0.8, 1.0]),
        ('alpha_09', {'alpha_thresh': 0.9}, [1.0, 1.0, 0.2]),
        ('combined', {'raydrop_thresh': 0.5, 'alpha_thresh': 0.9}, [0.2, 1.0, 0.3]),
        ('matched', {'raydrop_thresh': 0.5, 'max_pts': len(real)}, [0.0, 1.0, 0.5]),
    ]

    for name, kwargs, color in configs:
        if name == 'real':
            pts = real
        else:
            pts = render_scan(splats, frame['R_wl'], frame['t_wl'], K, **kwargs)

        print(f"  {name}: {len(pts)} pts")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        o3d.io.write_point_cloud(str(ex_dir / f'{name}.ply'), pcd)

    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
