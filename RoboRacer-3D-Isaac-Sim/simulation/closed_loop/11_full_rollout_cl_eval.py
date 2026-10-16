"""
Full Rollout Closed-Loop (CL) Evaluator — Standard Protocol
============================================================

Evaluates GRU checkpoint quality using full autoregressive rollout,
matching the protocol agreed with GPT (Rounds 4-5).

Protocol:
  - validation CSV: Day2/Day3 development sessions ONLY (never Day1)
  - warmup: first HISTORY=10 steps use ground-truth state (not autoregressive)
  - rollout: from step=10 onward, [vx, vy, yr, ax, ay] come from model predictions
  - controls: [speed_cmd, steering_cmd] always from CSV ground truth
  - no reset: unlike windowed CL (K=10), this never resets GRU history
  - metrics: 3ch RMSE (vx, vy, yr) and 5ch RMSE (vx, vy, yr, ax, ay)
  - duration: report both 60s and 125s tiers (do not loop short CSVs to reach 125s)

Usage:
  conda activate f1tenth_dynamics
  python3 11_full_rollout_cl_eval.py --checkpoint gru_run30_best.pth \
      --csv <path-to>/ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv
  # (ma_jonathanmohr-main/ raw sessions are not included in this repo --
  #  available on request, see main README.md)

  # Evaluate on multiple CSVs (weighted mean):
  python3 11_full_rollout_cl_eval.py --checkpoint gru_run30_best.pth --csv-dir <dir>

CRITICAL CONSTRAINTS (Felix):
  session_20260721_* (Day1) = TEST ONLY — NEVER use for checkpoint selection
  session_20260728_124427_* = EXCLUDED (100% NaN IMU columns)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Expected columns ──────────────────────────────────────────────────────────

_INPUT_COLS  = ['v_x_local', 'v_y_local', 'mocap_angular_velocity',
                'a_x_local', 'a_y_local', 'speed_cmd', 'steering_cmd']
_OUTPUT_COLS = ['v_x_local', 'v_y_local', 'mocap_angular_velocity',
                'a_x_local', 'a_y_local']
_3CH         = ['v_x_local', 'v_y_local', 'mocap_angular_velocity']  # vx, vy, yr
HISTORY      = 10
DT           = 0.02  # 20ms


# ── Model loader ──────────────────────────────────────────────────────────────

class _RNN_jomoh_1(torch.nn.Module):
    """Exact replica of RNN_jomoh_1 from training code.

    IMPORTANT: forward() always starts from h=0 (hidden state NOT preserved between calls).
    At inference, always feed a full HISTORY=10 step window — exactly like GRUInference.step().
    Do NOT feed single steps with accumulated h; that changes the computation.
    """
    def __init__(self, input_size=7, output_size=5, hidden_size=100):
        super().__init__()
        self.gru1      = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1       = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3       = torch.nn.Linear(hidden_size, hidden_size)
        self.fc5       = torch.nn.Linear(hidden_size, output_size)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out, _ = self.gru1(x)   # h=0 each call, _ discarded — matches training exactly
        z = self.LeakyReLU(self.fc1(out[:, -1, :]))
        z = self.LeakyReLU(self.fc3(z))
        return self.fc5(z)


def load_gru(ckpt_path: str, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get('config', {})
    model_state = ckpt.get('model_state_dict', ckpt)

    in_sz  = cfg.get('input_size',  7)
    out_sz = cfg.get('output_size', 5)

    if in_sz != 7 or out_sz != 5:
        raise RuntimeError(f"[CL-eval] Wrong arch: in={in_sz} out={out_sz}, expected 7/5")

    got = cfg.get('input_columns', [])
    if got and got != _INPUT_COLS:
        raise RuntimeError(f"[CL-eval] Wrong input columns!\n  got:      {got}\n  expected: {_INPUT_COLS}")

    model = _RNN_jomoh_1(input_size=in_sz, output_size=out_sz)
    model.load_state_dict(model_state)
    model.to(device).eval()

    # normed=False in config → no external scaler; model operates on raw values
    return model, cfg


def scale(arr, sc):
    return arr


def unscale(arr, sc):
    return arr


# ── Full rollout evaluation on one CSV ────────────────────────────────────────

def eval_csv(csv_path: str, model, cfg: dict, device='cpu',
             max_steps_60: int = 3000, max_steps_125: int = 6250) -> dict:
    """
    Run full autoregressive rollout on one CSV.
    Returns dict with 3ch/5ch RMSE for 60s and 125s tiers, plus per-channel RMSE.
    normed=False in config means no external scaler — model operates on raw sensor values.
    """
    df = pd.read_csv(csv_path)
    name = Path(csv_path).name

    # Safety guard: Day1 must never be used
    if '20260721' in name:
        raise SystemExit(
            f"[CL-eval] BLOCKED: {name} is Day1 — TEST ONLY. Use Day2/Day3 for dev eval.")
    if '20260728_124427' in name:
        raise SystemExit(
            f"[CL-eval] BLOCKED: {name} has 100% NaN IMU columns — excluded by Felix.")

    missing = [c for c in _INPUT_COLS + _OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[CL-eval] Missing columns in {name}: {missing}")

    n_rows = len(df)
    if n_rows < HISTORY + 5:
        return {'name': name, 'error': f'too short ({n_rows} rows < {HISTORY + 5})'}

    X_raw = df[_INPUT_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    Y_raw = df[_OUTPUT_COLS].fillna(0.0).to_numpy(dtype=np.float32)

    results = {}

    for label, max_steps in [('60s', max_steps_60), ('125s', max_steps_125)]:
        total = min(n_rows - HISTORY, max_steps)
        if total < 10:
            results[label] = {'error': f'not enough rows for {label}'}
            continue

        # Rolling buffer of HISTORY rows — exactly like GRUInference._buf (deque, maxlen=10).
        # Warmup: fill with first HISTORY ground-truth rows.
        from collections import deque
        buf = deque(list(X_raw[:HISTORY]), maxlen=HISTORY)

        # Full autoregressive rollout
        pred_all      = []
        gt_all        = []
        current_state = X_raw[HISTORY - 1, :5].copy()  # GT state at last warmup row

        for t in range(total):
            step_idx = HISTORY + t
            # Controls always from CSV ground truth
            ctrl = X_raw[step_idx, 5:7]
            # New row: predicted state + csv control (same as GRUInference.step())
            new_row = np.concatenate([current_state, ctrl]).astype(np.float32)
            buf.append(new_row)

            # Feed full 10-step window from h=0 (matches training exactly)
            x_t = torch.tensor(np.array(list(buf)), dtype=torch.float32,
                               device=device).unsqueeze(0)   # (1, 10, 7)
            with torch.no_grad():
                pred_raw = model(x_t).squeeze(0).cpu().numpy()  # (5,)

            pred_all.append(pred_raw)
            gt_all.append(Y_raw[step_idx])
            current_state = pred_raw.copy()  # autoregressive: prediction feeds next step

        pred_arr = np.array(pred_all)   # (total, 5)
        gt_arr   = np.array(gt_all)     # (total, 5)
        err      = pred_arr - gt_arr

        ch_names = ['vx', 'vy', 'yr', 'ax', 'ay']
        ch_rmse  = {f'rmse_{c}': float(np.sqrt((err[:, i]**2).mean()))
                    for i, c in enumerate(ch_names)}
        rmse_3ch = float(np.sqrt((err[:, :3]**2).mean()))
        rmse_5ch = float(np.sqrt((err[:, :5]**2).mean()))

        # Divergence check
        max_vx = float(np.abs(pred_arr[:, 0]).max())
        max_yr = float(np.abs(pred_arr[:, 2]).max())
        diverged = (max_vx > 20.0 or max_yr > 15.0 or np.any(np.isnan(pred_arr)))

        results[label] = {
            'n_steps': total,
            'duration_s': total * DT,
            'rmse_3ch': rmse_3ch,
            'rmse_5ch': rmse_5ch,
            **ch_rmse,
            'max_pred_vx': max_vx,
            'max_pred_yr': max_yr,
            'diverged': diverged,
        }

    return {'name': name, 'n_rows': n_rows, **results}


# ── Aggregate over multiple CSVs ──────────────────────────────────────────────

def aggregate(results: list, tier: str) -> dict:
    """Weighted mean (by n_steps) over all valid CSVs for a given duration tier."""
    valid = [r for r in results if tier in r and 'error' not in r[tier] and not r[tier].get('diverged')]
    if not valid:
        return {'mean_rmse_3ch': float('nan'), 'mean_rmse_5ch': float('nan'),
                'median_rmse_3ch': float('nan'), 'n_sessions': 0}
    weights    = np.array([r[tier]['n_steps'] for r in valid], dtype=float)
    rmse3      = np.array([r[tier]['rmse_3ch'] for r in valid])
    rmse5      = np.array([r[tier]['rmse_5ch'] for r in valid])
    w_mean3    = float(np.average(rmse3, weights=weights))
    w_mean5    = float(np.average(rmse5, weights=weights))
    return {
        'n_sessions':      len(valid),
        'mean_rmse_3ch':   w_mean3,
        'mean_rmse_5ch':   w_mean5,
        'median_rmse_3ch': float(np.median(rmse3)),
        'worst_rmse_3ch':  float(rmse3.max()),
        'per_session':     [{'name': r['name'], 'rmse_3ch': r[tier]['rmse_3ch'],
                              'rmse_5ch': r[tier]['rmse_5ch']} for r in valid],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Standard full-rollout CL evaluator for GRU checkpoint selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--csv', help='Single validation CSV')
    grp.add_argument('--csv-dir', help='Directory of validation CSVs (all *_20ms.csv)')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()

    print('=' * 65)
    print(' Full Rollout CL Evaluator — Standard Protocol')
    print('=' * 65)

    model, cfg = load_gru(args.checkpoint, device=args.device)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  val_loss   : {cfg.get('best_val_loss', 'N/A')}")
    print(f"  Protocol   : warmup={HISTORY} steps, full autoregressive, no reset")
    print(f"  ⚠ Day1 CSV is blocked at runtime; use Day2/Day3 dev sessions only")
    print()

    # Collect CSVs
    if args.csv:
        csv_list = [args.csv]
    else:
        csv_dir = Path(args.csv_dir)
        csv_list = sorted(str(p) for p in csv_dir.glob('*_20ms.csv'))
        # Block known-excluded files
        csv_list = [c for c in csv_list
                    if '20260721' not in c and '20260728_124427' not in c]
        if not csv_list:
            print(f"[CL-eval] No valid CSVs found in {csv_dir}")
            sys.exit(1)

    print(f"  CSVs to evaluate: {len(csv_list)}")
    print()

    results = []
    for csv_path in csv_list:
        print(f"  ▶ {Path(csv_path).name}")
        try:
            r = eval_csv(csv_path, model, cfg, device=args.device)
        except Exception as e:
            r = {'name': Path(csv_path).name, 'error': str(e)}
        results.append(r)

        # Print per-CSV summary
        for tier in ['60s', '125s']:
            if tier in r:
                t = r[tier]
                if 'error' in t:
                    print(f"    [{tier}] {t['error']}")
                else:
                    div_str = '  ⚠ DIVERGED' if t['diverged'] else ''
                    print(f"    [{tier}] 3ch={t['rmse_3ch']:.4f}  5ch={t['rmse_5ch']:.4f}"
                          f"  vx={t['rmse_vx']:.4f}  vy={t['rmse_vy']:.4f}"
                          f"  yr={t['rmse_yr']:.4f}{div_str}")
        if 'error' in r:
            print(f"    ERROR: {r['error']}")
        print()

    # Aggregate summary
    print('=' * 65)
    print(' AGGREGATE RESULTS')
    print('=' * 65)
    for tier in ['60s', '125s']:
        agg = aggregate(results, tier)
        print(f"  [{tier}] sessions={agg['n_sessions']}"
              f"  weighted_mean_3ch={agg['mean_rmse_3ch']:.4f}"
              f"  weighted_mean_5ch={agg['mean_rmse_5ch']:.4f}"
              f"  median_3ch={agg['median_rmse_3ch']:.4f}"
              f"  worst_3ch={agg.get('worst_rmse_3ch', float('nan')):.4f}")

    print()
    print('  NOTE: These numbers are the deployment-selection metric.')
    print('  val_loss and windowed CL (K=10 reset) are training diagnostics only.')
    print('  Checkpoint ranking must be done on full rollout, NOT windowed CL.')
    print()
    print('  PROVISIONAL STATUS: If the source CSV for previously reported')
    print('  jul29 CL=0.5372 / jul30 CL=0.4626 is not verified, those numbers')
    print('  must be recomputed using this script on clean Day2/Day3 sessions.')
    print('=' * 65)


if __name__ == '__main__':
    main()
