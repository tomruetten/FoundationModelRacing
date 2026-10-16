"""
Evaluate trained MS-NN-full model on Day1 test set (session_20260721_20ms.csv).
Loads model from trained_models/msnn_full_car02/ and reports Ω and ax RMSE.

Usage:
  cd ma_jonathanmohr-main
  conda run -n nerfstudio-py310 python3 eval_msnn_car02.py
"""

import os, sys, glob, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

DATA_DIR   = os.path.join(WORK, 'processed_data')
MODEL_PATH = os.path.join(WORK, 'trained_models', 'msnn_full_car02')
TEST_CSV   = os.path.join(DATA_DIR, 'session_20260721_20ms.csv')
OUT_JSON   = '/tmp/eval_msnn_results.json'
OUT_PNG    = '/tmp/eval_msnn_plots.png'

from nnodely import *
from nnodely.support.initializer import *

NUM_SAMPLES_PAST = 20
CURRENT_WINDOW   = 15

def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Model directory not found: {MODEL_PATH}")
        print("  Run train_msnn_full_car02.py first.")
        sys.exit(1)

    df_test = pd.read_csv(TEST_CSV, index_col=0)
    print(f"Test set: {len(df_test)} samples = {len(df_test)*0.02:.1f}s")

    # Check required columns
    required = ['odom_speed', 'steering_cmd', 'IMU_linear_acceleration_x',
                'IMU_angular_velocity_z', 'current_motor', 'IMU_linear_acceleration_y']
    for col in required:
        if col not in df_test.columns:
            print(f"ERROR: column '{col}' missing from test CSV")
            sys.exit(1)

    ds_test = {
        'steer':    df_test['steering_cmd'].values.astype(np.float64),
        'vx':       df_test['odom_speed'].values.astype(np.float64),
        'yaw_rate': df_test['IMU_angular_velocity_z'].values.astype(np.float64),
        'ax':       df_test['IMU_linear_acceleration_x'].values.astype(np.float64),
        'vel':      df_test['odom_speed'].values.astype(np.float64),
        'crt':      df_test['current_motor'].values.astype(np.float64),
        'acc':      df_test['IMU_linear_acceleration_x'].values.astype(np.float64),
        'delta':    df_test['steering_cmd'].values.astype(np.float64),
        'accy':     df_test['IMU_linear_acceleration_y'].values.astype(np.float64),
    }

    # Load trained model
    dynamics = nnodely(visualizer='Standard', seed=12, workspace=MODEL_PATH)
    dynamics.loadModel()
    dynamics.neuralizeModel()

    dynamics.loadData(name='test_set', source=ds_test)

    num_use  = len(df_test) - NUM_SAMPLES_PAST - 5
    samples  = dynamics.getSamples('test_set', index=0, window=num_use)
    outputs  = dynamics(samples, sampled=True)

    # Yaw rate predictions
    yaw_pred = np.asarray(outputs['yaw_rate_']).flatten()
    yaw_true = np.array([samples['yaw_rate'][i] for i in range(len(yaw_pred))]).flatten()

    # Longitudinal acceleration predictions
    acc_pred = np.asarray(outputs['acceleration']).flatten()
    acc_true = np.array([samples['acc'][i][CURRENT_WINDOW] for i in range(len(acc_pred))]).flatten()

    rmse_yaw = float(np.sqrt(np.mean((yaw_pred - yaw_true)**2)))
    mae_yaw  = float(np.mean(np.abs(yaw_pred - yaw_true)))
    rmse_acc = float(np.sqrt(np.mean((acc_pred - acc_true)**2)))
    mae_acc  = float(np.mean(np.abs(acc_pred - acc_true)))

    print("\n" + "="*60)
    print("MS-NN-full — Test Set Results (Day1)")
    print("="*60)
    print(f"  Ω  (yaw rate)  RMSE: {rmse_yaw:.4f} rad/s  MAE: {mae_yaw:.4f}")
    print(f"  ax (long accel) RMSE: {rmse_acc:.4f} m/s²   MAE: {mae_acc:.4f}")
    print("="*60)

    # Load GRU comparison if available
    gru_results = None
    if os.path.exists('/tmp/eval_results.json'):
        with open('/tmp/eval_results.json') as f:
            gru_data = json.load(f)
        ch = gru_data.get('per_channel', {})
        gru_yaw = ch.get('mocap_angular_velocity', {}).get('cl_rmse')
        print(f"\nGRU Run30 vs MS-NN-lat (yaw rate):")
        if gru_yaw:
            diff = rmse_yaw - gru_yaw
            print(f"  GRU Run30 ψ̇ CL-RMSE: {gru_yaw:.4f} rad/s")
            print(f"  MS-NN-lat  Ω  RMSE:   {rmse_yaw:.4f} rad/s  ({diff:+.4f})")

    # Save JSON
    result = {
        'model':          'MS-NN-full',
        'model_path':     MODEL_PATH,
        'eval_timestamp': datetime.now().isoformat(),
        'dataset':        os.path.basename(TEST_CSV),
        'n_rows':         int(len(df_test)),
        'duration_s':     float(len(df_test) * 0.02),
        'yaw_rate_rmse':  rmse_yaw,
        'yaw_rate_mae':   mae_yaw,
        'acc_rmse':       rmse_acc,
        'acc_mae':        mae_acc,
        'note': 'open-loop one-step RMSE on Day1 test set',
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    # Plots
    t = np.arange(len(yaw_pred)) * 0.02
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('MS-NN-full — Test Set (session_20260721, Day1)', fontsize=13, fontweight='bold')

    ax1.plot(t, yaw_true,  lw=1.2, color='#C0392B', label='Ground Truth', alpha=0.9)
    ax1.plot(t, yaw_pred, lw=1.0, color='#2980B9', label='MS-NN-lat', alpha=0.85, ls='--')
    ax1.set_ylabel('Ω [rad/s]')
    ax1.set_title(f'Yaw Rate  RMSE={rmse_yaw:.4f} rad/s', fontsize=10, loc='left')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(t, acc_true,  lw=1.2, color='#C0392B', label='Ground Truth', alpha=0.9)
    ax2.plot(t, acc_pred, lw=1.0, color='#2980B9', label='MS-NN-long', alpha=0.85, ls='--')
    ax2.set_ylabel('ax [m/s²]')
    ax2.set_title(f'Longitudinal Acceleration  RMSE={rmse_acc:.4f} m/s²', fontsize=10, loc='left')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == '__main__':
    main()
