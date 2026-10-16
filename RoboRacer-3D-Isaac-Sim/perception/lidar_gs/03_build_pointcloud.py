"""
build_lidar_pointcloud.py — Accumulate LiDAR scans using MoCap poses into a
global world-frame point cloud, then save as COLMAP points3D.txt so nerfstudio's
splatfacto can use it for initialization instead of random point cloud.

Transform chain (per lidar scan at timestamp t):
  T_world_lidar = T_world_mocap(t) @ inv(T_BASE_MOCAP) @ T_BASE_LIDAR

  T_BASE_MOCAP: t=[0.10, 0.00, 0.20], R=I
  T_BASE_LIDAR: t=[0.27, 0.00, 0.11], R=I

Output: adds colmap/points3D.txt to the target dataset directory.
        nerfstudio reads this as initial 3D point cloud.

Usage:
  conda run -n nerfstudio-py310 python3 scripts/build_lidar_pointcloud.py [--exp EXPERIMENT_NAME]
"""
import argparse
import csv
import os
import struct
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

# ── Config ─────────────────────────────────────────────────────────────────────
# Raw data isn't included in this repo (see root README) -- point these at
# your own copies, e.g. export PROBE_ROOT=/path/to/Probe
PROBE_ROOT = Path(os.environ.get('PROBE_ROOT', './data/Probe'))
ROOT = Path(os.environ.get('LIDAR3DGS_ROOT', './data/Newmocap/layout_02_slow_run_02'))
BAG  = ROOT / 'run_20260327_105224_0.db3'
CSYNC = ROOT / 'camera_sync.csv'

T_BASE_MOCAP_t = np.array([0.10, 0.00, 0.20])
T_BASE_LIDAR_t = np.array([0.27, 0.00, 0.11])
POSE_OFFSET    = 28

SG_WIN, SG_POLY    = 21, 3
OUTLIER_POS_JUMP_M = 0.50

# Voxel downsampling: keep one point per voxel of this size (meters)
VOXEL_SIZE = 0.05   # 5cm voxels → reasonable density for 6.5x4.6m scene

# Maximum range filter (points beyond this are typically noisy)
MAX_RANGE_M = 8.0

def load_mocap():
    conn = sqlite3.connect(str(BAG))
    cur  = conn.cursor()
    cur.execute("""
        SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/vrpn_mocap/Car2/pose' ORDER BY m.timestamp
    """)
    rows = cur.fetchall()
    conn.close()

    n = len(rows)
    ts_ns = np.empty(n, dtype=np.float64)
    pos   = np.empty((n, 3))
    quat  = np.empty((n, 4))
    for i, (ts, data) in enumerate(rows):
        b = bytes(data)
        px, py, pz, qx, qy, qz, qw = struct.unpack_from('<7d', b, POSE_OFFSET)
        ts_ns[i] = float(ts)
        pos[i]   = [px, py, pz]
        quat[i]  = [qx, qy, qz, qw]

    # VRPN outlier filter
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    ok = np.ones(n, dtype=bool)
    ok[1:] &= (d < OUTLIER_POS_JUMP_M)
    ok[:-1] &= (d < OUTLIER_POS_JUMP_M)
    ts_ns, pos, quat = ts_ns[ok], pos[ok], quat[ok]

    _, uniq = np.unique(ts_ns, return_index=True)
    ts_ns, pos, quat = ts_ns[uniq], pos[uniq], quat[uniq]

    for ax in range(3): pos[:, ax]  = savgol_filter(pos[:, ax],  SG_WIN, SG_POLY)
    for ax in range(4): quat[:, ax] = savgol_filter(quat[:, ax], SG_WIN, SG_POLY)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    slerp = Slerp(ts_ns, Rotation.from_quat(quat))
    print(f"  Mocap: {len(ts_ns)} messages after filtering")
    return ts_ns, pos, slerp


def load_lidar_scans():
    """Read all PointCloud2 messages from bag. Returns list of (timestamp_ns, Nx3 float32 array)."""
    import struct as struct

    conn = sqlite3.connect(str(BAG))
    cur  = conn.cursor()
    cur.execute("""
        SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/livox/lidar' ORDER BY m.timestamp
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  LiDAR: {len(rows)} scans")
    return rows


def _align(idx: int, n: int) -> int:
    if idx % n != 0: idx += n - (idx % n)
    return idx


def parse_pointcloud2(data: bytes):
    """Parse ROS2 sensor_msgs/PointCloud2 (CDR little-endian). Returns Nx3 float32 xyz."""
    b = bytes(data)
    idx = 4  # skip CDR encapsulation header

    # Header.stamp
    idx += 8  # sec(4) + nanosec(4)

    # Header.frame_id (string: uint32 length + chars including null + alignment)
    frame_id_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += frame_id_len
    idx = _align(idx, 4)

    # height, width
    height = struct.unpack_from('<I', b, idx)[0]; idx += 4
    width  = struct.unpack_from('<I', b, idx)[0]; idx += 4

    # fields sequence
    n_fields = struct.unpack_from('<I', b, idx)[0]; idx += 4
    fields = {}
    for _ in range(n_fields):
        idx = _align(idx, 4)
        name_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
        name = b[idx:idx+name_len].rstrip(b'\x00').decode(); idx += name_len
        idx = _align(idx, 4)
        f_offset = struct.unpack_from('<I', b, idx)[0]; idx += 4
        f_dtype  = b[idx]; idx += 1
        idx = _align(idx, 4)
        f_count  = struct.unpack_from('<I', b, idx)[0]; idx += 4
        fields[name] = (f_offset, f_dtype)

    # is_bigendian (bool=1 byte) + alignment
    idx += 1
    idx = _align(idx, 4)

    # point_step, row_step, data
    point_step = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += 4   # row_step
    data_len   = struct.unpack_from('<I', b, idx)[0]; idx += 4
    data_start = idx

    if 'x' not in fields or 'y' not in fields or 'z' not in fields:
        return np.empty((0, 3), dtype=np.float32)

    x_off, _ = fields['x']
    y_off, _ = fields['y']
    z_off, _ = fields['z']

    pts = np.frombuffer(b[data_start:data_start+data_len], dtype=np.uint8).reshape(-1, point_step)
    xs = np.frombuffer(pts[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    ys = np.frombuffer(pts[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    zs = np.frombuffer(pts[:, z_off:z_off+4].tobytes(), dtype=np.float32)

    xyz = np.stack([xs, ys, zs], axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.1)
    return xyz[valid]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    vox_idx = np.floor(pts / voxel).astype(np.int32)
    _, uniq = np.unique(vox_idx, axis=0, return_index=True)
    return pts[uniq]


def write_ply(pts: np.ndarray, ply_path: Path):
    """Write point cloud as PLY file (open3d) for nerfstudio to load."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(pts), 3), 0.5))
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  Wrote {len(pts)} points to {ply_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='layout02_slow02_baseline',
                        help='Experiment name under PROBE_ROOT/experiments/')
    parser.add_argument('--voxel', type=float, default=VOXEL_SIZE,
                        help='Voxel size for downsampling (meters)')
    args = parser.parse_args()

    exp_dir = PROBE_ROOT / 'experiments' / args.exp / 'dataset'
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} does not exist")
        sys.exit(1)

    print(f"Building LiDAR point cloud for {args.exp}")

    # Load mocap
    print("Loading mocap...")
    ts_f, pos_f, slerp = load_mocap()

    # Load lidar
    print("Loading LiDAR scans...")
    lidar_rows = load_lidar_scans()

    # Load camera_sync to map lidar_ts → whether a camera frame exists
    cam_sync = {}
    with open(str(CSYNC)) as f:
        for row in csv.DictReader(f):
            lt = int(row['lidar_timestamp_ns'])
            cam_sync[lt] = row['matched'].strip().lower() == 'true'

    # Accumulate global point cloud
    all_pts = []
    n_scans = 0

    for ts, data in lidar_rows:
        lidar_ts = float(ts)

        # Only use lidar scans matched to a camera frame
        if not cam_sync.get(ts, False):
            continue

        # Skip if outside mocap time range
        if lidar_ts < ts_f[0] or lidar_ts > ts_f[-1]:
            continue

        # Mocap pose at lidar timestamp
        i = max(0, min(int(np.searchsorted(ts_f, lidar_ts)) - 1, len(ts_f) - 2))
        alpha = (lidar_ts - ts_f[i]) / max(float(ts_f[i+1] - ts_f[i]), 1.0)
        pos_m = (1.0 - alpha) * pos_f[i] + alpha * pos_f[i+1]
        R_wm  = slerp(lidar_ts).as_matrix()

        # T_world_base
        t_wb = R_wm @ (-T_BASE_MOCAP_t) + pos_m
        R_wb = R_wm

        # T_world_lidar
        t_wl = R_wb @ T_BASE_LIDAR_t + t_wb
        R_wl = R_wb  # T_BASE_LIDAR rotation = identity

        # Parse scan
        xyz_lidar = parse_pointcloud2(bytes(data))
        if len(xyz_lidar) == 0:
            continue

        # Filter by range
        ranges = np.linalg.norm(xyz_lidar, axis=1)
        xyz_lidar = xyz_lidar[ranges < MAX_RANGE_M]

        # Transform to world frame
        xyz_world = (R_wl @ xyz_lidar.T).T + t_wl

        all_pts.append(xyz_world.astype(np.float32))
        n_scans += 1

    if not all_pts:
        print("ERROR: no scans processed")
        sys.exit(1)

    pts = np.concatenate(all_pts, axis=0)
    print(f"  Accumulated {len(pts)} points from {n_scans} scans")

    # Voxel downsample
    pts = voxel_downsample(pts, args.voxel)
    print(f"  After voxel downsampling ({args.voxel}m): {len(pts)} points")

    # Write as PLY (nerfstudio reads via open3d from ply_file_path in transforms.json)
    ply_path = exp_dir / 'lidar_pointcloud.ply'
    write_ply(pts, ply_path)

    # Update transforms.json to include ply_file_path
    import json
    tf_path = exp_dir / 'transforms.json'
    tf = json.load(open(tf_path))
    tf['ply_file_path'] = 'lidar_pointcloud.ply'
    json.dump(tf, open(tf_path, 'w'), indent=2)
    print(f"  Updated transforms.json with ply_file_path")
    print(f"  Done! PLY: {ply_path}")
    print(f"  Run: ns-train splatfacto --data {exp_dir} ...")


if __name__ == '__main__':
    main()
