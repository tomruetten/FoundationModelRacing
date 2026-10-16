"""
lidar_scan_lookup.py — nearest-real-scan lookup table for the MoCap+LiDAR
hybrid renderer (2026-07-03, "problem 1: try mixing real captured scans with
the synthetic 3DGS model instead of always synthesizing").

Loads every real Livox scan in the training bag together with its
MoCap-derived world pose (same convention as train_lidar_3dgs.py's validated
Identity-rotation baseline -- Chamfer 4.3cm), and builds a KD-tree over the
sensor origins so a query pose can find its nearest real capture.

Usage:
    from lidar_scan_lookup import ScanLookup
    lut = ScanLookup()
    idx, dist_m = lut.nearest(t_wl_query)
    xyz_sensor, intensity = lut.scan(idx)   # RAW real points, sensor frame
"""
import os
import struct
import sqlite3
import csv
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree

PROBE_ROOT = Path(os.environ.get('LIDAR3DGS_PROBE_ROOT', str(Path(__file__).resolve().parent)))
ROOT = Path(os.environ.get('LIDAR3DGS_ROOT', str(PROBE_ROOT / 'raw_data/layout02_2026-07-02')))
BAG = ROOT / os.environ.get('LIDAR3DGS_BAG', 'run_20260327_105224_0.db3')
CSYNC = ROOT / 'camera_sync.csv'

T_BASE_MOCAP_t = np.array([0.10, 0.00, 0.20])
T_BASE_LIDAR_t = np.array([0.27, 0.00, 0.11])
POSE_OFFSET = 28
SG_WIN, SG_POLY = 21, 3
OUTLIER_POS_JUMP_M = 0.50
MAX_RANGE_M = 8.0


def _align(idx, n):
    if idx % n != 0: idx += n - (idx % n)
    return idx


def _parse_pointcloud2(data):
    b = bytes(data)
    idx = 4
    idx += 8
    frame_id_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += frame_id_len
    idx = _align(idx, 4)
    idx += 4  # height
    idx += 4  # width
    n_fields = struct.unpack_from('<I', b, idx)[0]; idx += 4
    fields = {}
    for _ in range(n_fields):
        idx = _align(idx, 4)
        name_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
        name = b[idx:idx+name_len].rstrip(b'\x00').decode(); idx += name_len
        idx = _align(idx, 4)
        f_offset = struct.unpack_from('<I', b, idx)[0]; idx += 4
        idx += 1  # dtype
        idx = _align(idx, 4)
        idx += 4  # count
        fields[name] = f_offset
    idx += 1
    idx = _align(idx, 4)
    point_step = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += 4
    data_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
    data_start = idx
    if 'x' not in fields:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    x_off, y_off, z_off = fields['x'], fields['y'], fields['z']
    pts = np.frombuffer(b[data_start:data_start+data_len], dtype=np.uint8).reshape(-1, point_step)
    xs = np.frombuffer(pts[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    ys = np.frombuffer(pts[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    zs = np.frombuffer(pts[:, z_off:z_off+4].tobytes(), dtype=np.float32)
    xyz = np.stack([xs, ys, zs], axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.1)
    intensity = np.zeros(valid.sum(), dtype=np.float32)
    if 'intensity' in fields:
        i_off = fields['intensity']
        intensity = np.frombuffer(pts[:, i_off:i_off+4].tobytes(), dtype=np.float32)[valid]
    return xyz[valid], intensity


def _load_mocap():
    conn = sqlite3.connect(str(BAG))
    cur = conn.cursor()
    cur.execute("""SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/vrpn_mocap/Car2/pose' ORDER BY m.timestamp""")
    rows = cur.fetchall()
    conn.close()
    n = len(rows)
    ts_ns = np.empty(n, dtype=np.float64)
    pos = np.empty((n, 3))
    quat = np.empty((n, 4))
    for i, (ts, data) in enumerate(rows):
        b = bytes(data)
        px, py, pz, qx, qy, qz, qw = struct.unpack_from('<7d', b, POSE_OFFSET)
        ts_ns[i] = float(ts)
        pos[i] = [px, py, pz]
        quat[i] = [qx, qy, qz, qw]
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    ok = np.ones(n, dtype=bool)
    ok[1:] &= (d < OUTLIER_POS_JUMP_M)
    ok[:-1] &= (d < OUTLIER_POS_JUMP_M)
    ts_ns, pos, quat = ts_ns[ok], pos[ok], quat[ok]
    _, uniq = np.unique(ts_ns, return_index=True)
    ts_ns, pos, quat = ts_ns[uniq], pos[uniq], quat[uniq]
    for ax in range(3): pos[:, ax] = savgol_filter(pos[:, ax], SG_WIN, SG_POLY)
    for ax in range(4): quat[:, ax] = savgol_filter(quat[:, ax], SG_WIN, SG_POLY)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    slerp = Slerp(ts_ns, Rotation.from_quat(quat))
    return ts_ns, pos, slerp


class ScanLookup:
    def __init__(self, max_scans=None):
        ts_f, pos_f, slerp = _load_mocap()

        conn = sqlite3.connect(str(BAG))
        cur = conn.cursor()
        cur.execute("""SELECT m.timestamp, m.data FROM messages m
            JOIN topics t ON m.topic_id=t.id
            WHERE t.name='/livox/lidar' ORDER BY m.timestamp""")
        lidar_rows = cur.fetchall()
        conn.close()

        cam_sync = {}
        with open(str(CSYNC)) as f:
            for row in csv.DictReader(f):
                lt = int(row['lidar_timestamp_ns'])
                cam_sync[lt] = row['matched'].strip().lower() == 'true'

        self.R_wl_list = []
        self.t_wl_list = []
        self.xyz_list = []
        self.intensity_list = []
        self.ts_list = []

        for ts, data in lidar_rows:
            lidar_ts = float(ts)
            if not cam_sync.get(ts, False):
                continue
            if lidar_ts < ts_f[0] or lidar_ts > ts_f[-1]:
                continue
            i = max(0, min(int(np.searchsorted(ts_f, lidar_ts)) - 1, len(ts_f) - 2))
            alpha = (lidar_ts - ts_f[i]) / max(float(ts_f[i+1] - ts_f[i]), 1.0)
            pos_m = (1.0 - alpha) * pos_f[i] + alpha * pos_f[i+1]
            R_wm = slerp(lidar_ts).as_matrix()

            t_wb = R_wm @ (-T_BASE_MOCAP_t) + pos_m
            R_wb = R_wm
            t_wl = R_wb @ T_BASE_LIDAR_t + t_wb
            R_wl = R_wb  # validated Identity-rotation baseline (v7_raydrop)

            xyz_lidar, intensity = _parse_pointcloud2(bytes(data))
            if len(xyz_lidar) == 0:
                continue
            ranges = np.linalg.norm(xyz_lidar, axis=1)
            mask = ranges < MAX_RANGE_M

            self.R_wl_list.append(R_wl)
            self.t_wl_list.append(t_wl)
            self.xyz_list.append(xyz_lidar[mask])
            self.intensity_list.append(intensity[mask])
            self.ts_list.append(int(ts))

            if max_scans and len(self.t_wl_list) >= max_scans:
                break

        self.t_wl_arr = np.array(self.t_wl_list)
        self.tree = cKDTree(self.t_wl_arr)
        print(f"[ScanLookup] {len(self.t_wl_list)} real scans indexed "
              f"(from {BAG.name})")

        # Exact index: camera_frame_idx -> our scan list index, for frames that
        # are literal replays of the recorded trajectory (see camera_sync.csv:
        # 'frame_{camera_frame_idx:06d}.png' in transforms.json ties an
        # nerfstudio dataset frame directly back to the LiDAR scan captured at
        # the *same instant* -- no COLMAP/T_cl round-trip needed, so no
        # accumulated pose error).
        self.by_camera_frame_idx = {}
        ts_to_scanidx = {ts: i for i, ts in enumerate(self.ts_list)}

        with open(str(CSYNC)) as f:
            for row in csv.DictReader(f):
                if row['matched'].strip().lower() != 'true':
                    continue
                lt = int(row['lidar_timestamp_ns'])
                fidx = int(row['camera_frame_idx'])
                if lt in ts_to_scanidx:
                    self.by_camera_frame_idx[fidx] = ts_to_scanidx[lt]

    def exact(self, camera_frame_idx: int):
        """Exact real scan for a replay frame (by nerfstudio frame_XXXXXX.png index).
        Returns scan list index, or None if not found."""
        return self.by_camera_frame_idx.get(camera_frame_idx)

    def nearest(self, t_wl_query: np.ndarray):
        """Returns (idx, dist_m) of the nearest real scan to a query position."""
        dist, idx = self.tree.query(t_wl_query)
        return int(idx), float(dist)

    def angle_deg(self, idx: int, R_wl_query: np.ndarray) -> float:
        """Orientation difference (deg) between scan `idx`'s real pose and a query rotation."""
        R_rel = self.R_wl_list[idx].T @ R_wl_query
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    def scan(self, idx: int):
        """Raw real captured points (sensor frame) + intensity for scan `idx`."""
        return self.xyz_list[idx], self.intensity_list[idx]
