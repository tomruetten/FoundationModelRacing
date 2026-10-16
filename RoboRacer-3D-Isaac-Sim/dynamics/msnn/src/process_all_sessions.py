"""
process_all_sessions.py — Full pipeline for 3 new sessions.

Steps:
  1. Extract each .db3 bag → 4 CSVs in data/<name>/
  2. Split CSVs at MoCap gaps → sub-session directories
  3. Run DataPreparation.py on each sub-session
  4. Merge all processed CSVs (incl. original session_155646) → merged_all_20ms.csv

Run from ma_jonathanmohr-main/:
  conda run -n nerfstudio-py310 python3 process_all_sessions.py
"""

import os
import sys
import shutil
import time

import numpy as np
import pandas as pd

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_DIR)

# ---------- Config ----------

# Raw ROS bags aren't included in this repo (see root README) -- point this
# at your own copy, e.g. export DYNAMICS_DATA_ROOT=/path/to/Dynamik
DYNAMICS_DATA_ROOT = os.environ.get('DYNAMICS_DATA_ROOT', './data/Dynamik')

BAGS = [
    {
        'bag': f'{DYNAMICS_DATA_ROOT}/session_20260722_113922/all_data/all_data_0.db3',
        'name': 'session_20260722_113922',
    },
    {
        'bag': f'{DYNAMICS_DATA_ROOT}/session_20260722_115055/all_data/all_data_0.db3',
        'name': 'session_20260722_115055',
    },
    {
        'bag': f'{DYNAMICS_DATA_ROOT}/session_20260722_120608/all_data/all_data_0.db3',
        'name': 'session_20260722_120608',
    },
]

# Original session already processed
ORIGINAL_CSV = os.path.join(WORK_DIR, 'processed_data', 'session_20260721_20ms.csv')

GAP_THRESHOLD = 2.0   # MoCap gap threshold (seconds)
MIN_SEG_DURATION = 15.0  # minimum segment to keep (seconds)
TARGET_PERIOD = 20    # 50 Hz

MERGED_OUTPUT = os.path.join(WORK_DIR, 'processed_data', 'merged_all_20ms.csv')


# ---------- Step 1: Extract bags ----------

def extract_bag(bag_path, session_name):
    """Run extract_bag_for_felix.py on a single .db3 file."""
    out_dir = os.path.join(WORK_DIR, 'data', session_name)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= 4:
        print(f"  [SKIP] {session_name} already extracted.")
        return out_dir

    print(f"  Extracting {session_name} ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(WORK_DIR, 'extract_bag_for_felix.py'),
         '--bag', bag_path, '--name', session_name],
        cwd=WORK_DIR, capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Extraction failed for {session_name}")
    return out_dir


# ---------- Step 2: Find MoCap segments ----------

def find_segments(pose_csv):
    """Return list of (t_start, t_end) for continuous MoCap segments >= MIN_SEG_DURATION."""
    df = pd.read_csv(pose_csv)
    ts = df['timestamp'].values

    if len(ts) < 10:
        return []

    gaps = np.diff(ts)
    gap_idx = np.where(gaps > GAP_THRESHOLD)[0]

    boundaries = [0] + list(gap_idx + 1) + [len(ts)]
    segments = []
    for i in range(len(boundaries) - 1):
        t_start = ts[boundaries[i]]
        t_end = ts[boundaries[i + 1] - 1]
        if t_end - t_start >= MIN_SEG_DURATION:
            segments.append((t_start, t_end))

    return segments


# ---------- Step 3: Split CSVs ----------

def split_session(data_dir, session_name, segments):
    """Create per-segment sub-directories with sliced CSVs."""
    csv_keys = ['pose', 'controls', 'odom', 'imu']
    dfs = {}
    for key in csv_keys:
        path = os.path.join(data_dir, f'{session_name}_{key}.csv')
        dfs[key] = pd.read_csv(path)

    seg_dirs = []
    for i, (t_start, t_end) in enumerate(segments):
        seg_name = f'{session_name}_seg{i + 1}'
        seg_dir = os.path.join(WORK_DIR, 'data', seg_name)

        if os.path.isdir(seg_dir) and len(os.listdir(seg_dir)) >= 4:
            print(f"    [SKIP] {seg_name} already split.")
            seg_dirs.append((seg_name, seg_dir, t_end - t_start))
            continue

        os.makedirs(seg_dir, exist_ok=True)
        for key, df in dfs.items():
            mask = (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)
            seg_df = df[mask].copy()
            seg_df.to_csv(os.path.join(seg_dir, f'{seg_name}_{key}.csv'), index=False)

        dur = t_end - t_start
        print(f"    {seg_name}: {dur:.0f}s → {seg_dir}")
        seg_dirs.append((seg_name, seg_dir, dur))

    return seg_dirs


# ---------- Step 4: Run DataPreparation.py ----------

def run_dataprep(seg_name, seg_dir):
    """Run process_data() on a sub-session directory."""
    out_csv = os.path.join(WORK_DIR, 'processed_data', f'{seg_name}_{TARGET_PERIOD}ms.csv')
    if os.path.isfile(out_csv):
        print(f"    [SKIP] {seg_name} already processed.")
        return out_csv

    print(f"    Running DataPreparation on {seg_name} ...")
    from DataPreparation import process_data
    try:
        process_data(
            source_dir=seg_dir,
            target_dir=os.path.join(WORK_DIR, 'processed_data'),
            target_period=TARGET_PERIOD,
            plotting=False,
        )
        if os.path.isfile(out_csv):
            df = pd.read_csv(out_csv, index_col=0)
            print(f"    → {len(df)} rows, {len(df)*0.02:.1f}s @ 50Hz")
            return out_csv
        else:
            print(f"    WARNING: expected output not found: {out_csv}")
            return None
    except Exception as e:
        print(f"    ERROR in DataPreparation for {seg_name}: {e}")
        return None


# ---------- Step 5: Merge all CSVs ----------

def merge_all(csv_paths):
    """Concatenate all processed CSVs, reset index, return merged DataFrame."""
    dfs = []
    for path in csv_paths:
        if path and os.path.isfile(path):
            df = pd.read_csv(path, index_col=0)
            # Drop columns not needed for training boundary tracking
            dfs.append(df)
            print(f"  Loaded {os.path.basename(path)}: {len(df)} rows")
        else:
            print(f"  SKIP (not found): {path}")

    if not dfs:
        raise RuntimeError("No valid CSVs to merge!")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged: {len(merged)} rows total = {len(merged)*0.02:.1f}s @ 50Hz")
    return merged


# ---------- Main ----------

def main():
    t_start_total = time.time()
    os.makedirs(os.path.join(WORK_DIR, 'processed_data'), exist_ok=True)

    all_processed = []

    # Include original session_155646 if it exists
    if os.path.isfile(ORIGINAL_CSV):
        print(f"Original session: {os.path.basename(ORIGINAL_CSV)}")
        df_orig = pd.read_csv(ORIGINAL_CSV, index_col=0)
        print(f"  {len(df_orig)} rows ({len(df_orig)*0.02:.1f}s)")
        all_processed.append(ORIGINAL_CSV)
    else:
        print(f"WARNING: Original session CSV not found: {ORIGINAL_CSV}")

    for bag_info in BAGS:
        bag_path = bag_info['bag']
        session_name = bag_info['name']
        print(f"\n{'='*60}")
        print(f"SESSION: {session_name}")
        print(f"{'='*60}")

        # Step 1: Extract
        print("Step 1: Extracting bag ...")
        data_dir = extract_bag(bag_path, session_name)

        # Step 2: Find segments
        print("Step 2: Finding MoCap segments ...")
        pose_csv = os.path.join(data_dir, f'{session_name}_pose.csv')
        segments = find_segments(pose_csv)
        print(f"  Found {len(segments)} segments:")
        for i, (s, e) in enumerate(segments):
            print(f"    seg{i+1}: {e-s:.0f}s  ({s:.1f}s → {e:.1f}s)")

        # Step 3: Split
        print("Step 3: Splitting CSVs ...")
        seg_dirs = split_session(data_dir, session_name, segments)

        # Step 4: DataPreparation per segment
        print("Step 4: Running DataPreparation ...")
        for seg_name, seg_dir, dur in seg_dirs:
            out_csv = run_dataprep(seg_name, seg_dir)
            if out_csv:
                all_processed.append(out_csv)

    # Step 5: Merge
    print(f"\n{'='*60}")
    print("Step 5: Merging all processed sessions ...")
    print(f"{'='*60}")
    merged = merge_all(all_processed)

    # Check for GRU input columns
    gru_inputs = ['v_x_local', 'v_y_local', 'mocap_angular_velocity',
                  'a_x_local', 'a_y_local', 'speed_cmd', 'steering_cmd']
    missing = [c for c in gru_inputs if c not in merged.columns]
    if missing:
        print(f"WARNING: missing GRU input columns: {missing}")
    else:
        print(f"All 7 GRU input columns present ✓")

    # NaN check
    nan_counts = merged[gru_inputs].isna().sum()
    if nan_counts.any():
        print(f"NaN counts in GRU inputs:\n{nan_counts[nan_counts>0]}")
        n_before = len(merged)
        merged.dropna(subset=gru_inputs, inplace=True)
        print(f"Dropped {n_before - len(merged)} rows with NaN → {len(merged)} rows remaining")
    else:
        print(f"No NaN in GRU inputs ✓")

    # Save
    merged.to_csv(MERGED_OUTPUT, index=True)
    print(f"\nSaved: {MERGED_OUTPUT}")
    print(f"Final dataset: {len(merged)} rows = {len(merged)*0.02:.1f}s @ 50Hz")

    elapsed = time.time() - t_start_total
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("\nReady for training! Run:")
    print("  conda run -n nerfstudio-py310 python3 run_training.py")


if __name__ == '__main__':
    main()
