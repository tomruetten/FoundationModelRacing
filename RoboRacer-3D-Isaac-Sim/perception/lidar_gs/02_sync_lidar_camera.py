"""
Generate camera_sync.csv for new_recording_20260714 LiDAR 3DGS training.

Maps each LiDAR scan timestamp → nearest camera frame index.
Output format matches what train_lidar_3dgs.py load_colmap_poses() expects:
  lidar_timestamp_ns, camera_frame_idx, matched
"""
import os
import sqlite3
from pathlib import Path

import numpy as np

# Raw data isn't included in this repo (see root README) -- point these at
# your own copies, e.g. export NEW_DATA_ROOT=/path/to/New_probe
NEW_DATA_ROOT = Path(os.environ.get('NEW_DATA_ROOT', './data/New_probe'))
PROBE_ROOT = Path(os.environ.get('PROBE_ROOT', './data/Probe'))
BAG = NEW_DATA_ROOT / 'newmp4/lidar_20260714_161149/lidar_20260714_161149_0.db3'
TS_FILE = NEW_DATA_ROOT / 'newmp4/camera_20260714_161152.mp4_rgb_timestamps.txt'
OUT_CSV = PROBE_ROOT / 'experiments/new_recording_20260714/camera_sync.csv'

MATCH_THRESH_S = 0.05  # 50ms


def main():
    # Load camera timestamps: frame_idx → unix_sec
    cam_ts = {}
    with open(TS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cam_ts[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
    cam_indices  = np.array(sorted(cam_ts.keys()), dtype=np.int64)
    cam_times_ns = np.array([cam_ts[i] * 1e9 for i in cam_indices], dtype=np.int64)
    print(f"Loaded {len(cam_ts)} camera frames")

    # Load LiDAR timestamps from bag
    print(f"Reading LiDAR timestamps from {BAG.name}...")
    conn = sqlite3.connect(str(BAG))
    cur  = conn.cursor()
    cur.execute("""
        SELECT m.timestamp FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/livox/lidar' ORDER BY m.timestamp
    """)
    lidar_timestamps = [int(row[0]) for row in cur.fetchall()]
    conn.close()
    print(f"  Found {len(lidar_timestamps)} LiDAR scans")

    # Match each LiDAR scan to nearest camera frame
    thresh_ns = int(MATCH_THRESH_S * 1e9)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    n_matched = 0
    with open(OUT_CSV, 'w') as f:
        f.write('lidar_timestamp_ns,camera_frame_idx,matched\n')
        for lt_ns in lidar_timestamps:
            diff = np.abs(cam_times_ns - lt_ns)
            best_i = np.argmin(diff)
            matched = diff[best_i] < thresh_ns
            cam_idx = int(cam_indices[best_i])
            f.write(f'{lt_ns},{cam_idx},{str(matched)}\n')
            if matched:
                n_matched += 1

    print(f"Wrote {OUT_CSV}: {n_matched}/{len(lidar_timestamps)} matched")


if __name__ == '__main__':
    main()
