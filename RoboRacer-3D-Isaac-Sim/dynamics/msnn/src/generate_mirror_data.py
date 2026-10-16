"""
generate_mirror_data.py — Datenaugmentierung: Spiegelung

Erzeugt für alle Trainings-CSVs eine links-rechts gespiegelte Kopie:
  v_y, ψ̇, a_y, steering_cmd bekommen gleichzeitig ein negatives Vorzeichen
  → simuliert Fahrt in Gegenrichtung, verdoppelt die Datenmenge
"""

import pandas as pd
import os

PROC = 'processed_data'

# Spalten, deren Vorzeichen gespiegelt werden muss (links-rechts-symmetrische Größen)
MIRROR_COLS = [
    'v_y_local', 'v_y_global',
    'mocap_angular_velocity', 'odom_angular_velocity',
    'a_y_local', 'a_y_global',
    'steering_cmd',
    'IMU_linear_acceleration_y',  # lateral accel — flips with mirror
    'IMU_angular_velocity_z',     # yaw rate — flips with mirror
]

TRAIN_FILES = [
    f'{PROC}/session_20260722_113922_seg1_20ms.csv',
    f'{PROC}/session_20260722_113922_seg2_20ms.csv',
    f'{PROC}/session_20260722_113922_seg3_20ms.csv',
    f'{PROC}/session_20260722_115055_seg1_20ms.csv',
    f'{PROC}/session_20260722_120608_seg1_20ms.csv',
    f'{PROC}/session_20260722_120608_seg2_20ms.csv',
    f'{PROC}/session_20260722_120608_seg3_20ms.csv',
    # session_20260721_train_20ms.csv removed — Day1 training split violates
    # test-data isolation (Issue 3.5). Do NOT add it back.
]

generated = []
for path in TRAIN_FILES:
    df = pd.read_csv(path, index_col=0)
    mirrored = df.copy()
    flipped = []
    for col in MIRROR_COLS:
        if col in mirrored.columns:
            mirrored[col] = -mirrored[col]
            flipped.append(col)
    out = path.replace('.csv', '_mirror.csv')
    mirrored.to_csv(out)
    generated.append(out)
    print(f'  {os.path.basename(path)} → {os.path.basename(out)}  (flipped: {flipped})')

print(f'\nDone. {len(generated)} mirror files generated.')
print('\nAdd these to TRAIN_FILES in run_training.py:')
for f in generated:
    print(f"    '{f}',")
