"""
post_training_eval.py — Run after training completes.

Finds the newest .pth in trained_models/, loads the RNN_jomoh_3 model,
evaluates on the processed CSV, and writes:
  /tmp/eval_results.json   — RMSE/MAE per channel + auto-regressive RMSE
  /tmp/eval_plots.png      — predictions vs ground truth (5 channels)
  /tmp/eval_report.html    — final HTML report for the artifact

Usage (from ma_jonathanmohr-main/):
  conda run -n nerfstudio-py310 python3 post_training_eval.py
"""

import sys, os, glob, json, struct
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- paths ----------
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(WORK_DIR, 'processed_data/session_20260721_20ms.csv')  # Run12 test set: full Day1
MODEL_DIR  = os.path.join(WORK_DIR, 'trained_models')
OUT_JSON   = '/tmp/eval_results.json'
OUT_PNG    = '/tmp/eval_plots.png'
OUT_HTML   = '/tmp/eval_report.html'

INPUTS  = ["v_x_local","v_y_local","mocap_angular_velocity","a_x_local","a_y_local","speed_cmd","steering_cmd"]
OUTPUTS = ["v_x_local","v_y_local","mocap_angular_velocity","a_x_local","a_y_local"]
UNITS   = ["m/s","m/s","rad/s","m/s²","m/s²"]
HISTORY = 10

# Felix thesis Table 8.1 baseline (RNN_jomoh_3, no control inputs, original dataset)
# Values from thesis Chapter 8 — used for comparison only
FELIX_BASELINE = {
    "v_x_local":             0.073,
    "v_y_local":             0.028,
    "mocap_angular_velocity": 0.251,
    "a_x_local":             0.587,
    "a_y_local":             0.751,
}


def find_newest_model():
    import re
    pths = glob.glob(os.path.join(MODEL_DIR, '*.pth'))
    if not pths:
        return None
    # Use timestamp embedded in filename — mtime changes on file copy/sync (Issue 8)
    def _ts(p):
        m = re.search(r'(\d{15,})\.pth$', os.path.basename(p))
        return m.group(1) if m else '0'
    return max(pths, key=_ts)


def load_model(pth_path):
    device = torch.device('cpu')  # force CPU: CUDA available but kernel incompatible on this machine
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    sys.path.insert(0, WORK_DIR)
    import RNN_architectures
    cls = getattr(RNN_architectures, cfg['model_class'])
    model = cls(input_size=cfg['input_size'], output_size=cfg['output_size']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded: {os.path.basename(pth_path)}")
    print(f"  model={cfg['model_class']}  input={cfg['input_size']}  output={cfg['output_size']}")
    print(f"  best_val_loss={cfg['best_val_loss']:.6f}")
    return model, cfg, device


def closed_loop_rmse(model, df, device):
    """Closed-loop rollout RMSE (replicates training.py evaluate_model).
    Sequences of length HISTORY+1, sliding window."""
    data = df[INPUTS].to_numpy(dtype=np.float32)
    gt   = df[OUTPUTS].to_numpy(dtype=np.float32)
    out_idx = [INPUTS.index(c) for c in OUTPUTS]

    T = len(data)
    H = HISTORY
    preds_all = []
    truth_all = []

    with torch.no_grad():
        window = torch.from_numpy(data[:H]).unsqueeze(0).to(device)  # [1, H, F]
        for t in range(H, T):
            pred = model(window)          # [1, Dy]
            p = pred.squeeze(0).cpu().numpy()
            preds_all.append(p)
            truth_all.append(gt[t])

            # slide window
            next_row = data[t].copy()
            next_row[out_idx] = p
            next_t = torch.from_numpy(next_row).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,F]
            window = torch.cat([window[:, 1:, :], next_t], dim=1)

    preds = np.array(preds_all)   # [T-H, 5]
    truth = np.array(truth_all)   # [T-H, 5]
    rmse = np.sqrt(np.mean((preds - truth)**2, axis=0))
    mae  = np.mean(np.abs(preds - truth), axis=0)
    return rmse, mae, preds, truth


def teacher_forced_rmse(model, df, device):
    """Teacher-forced RMSE (model sees GT at every step)."""
    data = df[INPUTS].to_numpy(dtype=np.float32)
    gt   = df[OUTPUTS].to_numpy(dtype=np.float32)
    H = HISTORY
    T = len(data)

    preds = []
    with torch.no_grad():
        for t in range(H, T):
            window = torch.from_numpy(data[t-H:t]).unsqueeze(0).to(device)
            pred = model(window).squeeze(0).cpu().numpy()
            preds.append(pred)

    preds = np.array(preds)
    truth = gt[H:]
    rmse = np.sqrt(np.mean((preds - truth)**2, axis=0))
    mae  = np.mean(np.abs(preds - truth), axis=0)
    return rmse, mae


def make_plots(preds, truth, rmse_cl, out_path):
    t = np.arange(len(preds)) * 0.02  # 20ms steps

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('GRU RNN_jomoh_3 — Auto-Regressive Predictions vs Ground Truth\nsession_20260721  (126s @ 50Hz)',
                 fontsize=13, fontweight='bold')

    colors_gt   = '#C0392B'
    colors_pred = '#2980B9'

    for i, (ax, col, unit) in enumerate(zip(axes, OUTPUTS, UNITS)):
        ax.plot(t, truth[:, i],  lw=1.2, color=colors_gt,   label='Ground Truth',   alpha=0.9)
        ax.plot(t, preds[:, i],  lw=1.0, color=colors_pred,  label='GRU Prediction', alpha=0.85, ls='--')
        ax.set_ylabel(f'{col}\n({unit})', fontsize=9)
        rmse_felix = FELIX_BASELINE.get(col, None)
        title = f'RMSE={rmse_cl[i]:.4f} {unit}'
        if rmse_felix:
            diff = rmse_cl[i] - rmse_felix
            sign = '+' if diff >= 0 else ''
            title += f'  |  Felix baseline={rmse_felix:.3f}  ({sign}{diff:.4f})'
        ax.set_title(title, fontsize=9, loc='left')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plots: {out_path}")


def main():
    pth = find_newest_model()
    if pth is None:
        print("ERROR: No .pth found in trained_models/. Training may not have finished.")
        sys.exit(1)

    model, cfg, device = load_model(pth)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Data: {len(df)} rows, {len(df)*0.02:.1f}s")

    print("\nRunning teacher-forced evaluation...")
    rmse_tf, mae_tf = teacher_forced_rmse(model, df, device)

    print("Running closed-loop (auto-regressive) evaluation...")
    rmse_cl, mae_cl, preds, truth = closed_loop_rmse(model, df, device)

    # --- Results table ---
    print("\n" + "="*70)
    print(f"{'Channel':<28} {'TF-RMSE':>8} {'CL-RMSE':>8} {'Felix':>8} {'Delta':>8}  Unit")
    print("-"*70)
    results = {}
    for i, col in enumerate(OUTPUTS):
        felix = FELIX_BASELINE.get(col, float('nan'))
        delta = rmse_cl[i] - felix
        sign  = '+' if delta >= 0 else ''
        print(f"  {col:<26} {rmse_tf[i]:>8.4f} {rmse_cl[i]:>8.4f} {felix:>8.3f} {sign}{delta:.4f}  {UNITS[i]}")
        results[col] = {
            'tf_rmse':   float(rmse_tf[i]),
            'cl_rmse':   float(rmse_cl[i]),
            'tf_mae':    float(mae_tf[i]),
            'cl_mae':    float(mae_cl[i]),
            'felix_baseline': float(felix),
            'delta_vs_felix': float(delta),
            'unit': UNITS[i],
        }
    print("-"*70)
    mean_cl = float(np.mean(rmse_cl))
    mean_felix = float(np.mean(list(FELIX_BASELINE.values())))
    print(f"  {'Mean':<26} {np.mean(rmse_tf):>8.4f} {mean_cl:>8.4f} {mean_felix:>8.3f} {mean_cl-mean_felix:+.4f}")
    print("="*70)

    summary = {
        'model_file':      os.path.basename(pth),
        'model_class':     cfg['model_class'],
        'best_val_loss':   float(cfg['best_val_loss']),
        'eval_timestamp':  datetime.now().isoformat(),
        'dataset':         os.path.basename(CSV_PATH),
        'n_rows':          int(len(df)),
        'duration_s':      float(len(df) * 0.02),
        'mean_cl_rmse':    mean_cl,
        'mean_felix_rmse': mean_felix,
        'per_channel':     results,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results: {OUT_JSON}")

    make_plots(preds, truth, rmse_cl, OUT_PNG)
    print(f"All done. Run generate_report.py to build the HTML report.")
    return summary


if __name__ == '__main__':
    main()
