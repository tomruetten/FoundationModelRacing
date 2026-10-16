"""
14_eval_and_plot.py — Evaluierung + Visualisierung beider Modelle
==================================================================
Lädt MS-NN (exp9) und GRU (Run30), wertet auf Day3 Val und Day1 Test aus,
erstellt Vergleichsplots für Felix V1-Präsentation.

Ausgabe:
  results_msnn_vs_gru.png      ← Haupt-Vergleichsplot
  results_comparison.csv       ← Zahlentabelle aller Metriken
  plot_yawrate_day3.png        ← ψ̇ Zeitverlauf Val
  plot_yawrate_day1.png        ← ψ̇ Zeitverlauf Test (cross-day)

Ausführen:
  conda activate nnodely-env        (für MS-NN)
  pip install matplotlib pandas     (falls nicht vorhanden)
  python3 14_eval_and_plot.py

  Falls beide Modelle getrennt:
    python3 14_eval_and_plot.py --model msnn     (nur MS-NN)
    python3 14_eval_and_plot.py --model gru      (nur GRU)
    python3 14_eval_and_plot.py --model both     (Standard)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR / "src"))  # inference_msnn.py / gru_inference.py live in src/

# ── Konfiguration ─────────────────────────────────────────────────────────────

DATA_DIR   = Path(os.environ.get('DATA_DIR', str(_DIR.parent / 'training' / 'processed_data')))
MSNN_MODEL = str(_DIR / 'trained_models' / 'msnn_exp9_phase2_backup')
GRU_CKPT   = str(_DIR / 'trained_models' / 'gru_run30_best.pth')

VAL_CSV  = DATA_DIR / 'session_20260728_163324_seg1_20ms.csv'   # Day3
TEST_CSV = DATA_DIR / 'session_20260721_20ms.csv'               # Day1 (cross-day)

PERSIST_YR = 0.1098  # rad/s — Day3 163324 Val-Section
PERSIST_AX = 0.1487  # m/s²


# ── Daten laden ───────────────────────────────────────────────────────────────

def load_csv(path, split=None):
    """Lädt CSV, gibt (df, ds_dict) zurück.
    split: None=alles, 'val'=letzte 2167 Zeilen, 'train'=erste 4333 Zeilen
    """
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y'])
    # BUGFIX 2026-08-13: this file has 10294 rows after dropna, not ~6500 --
    # "last 2167 rows of the full file" (the old code below) lands on t≈162-
    # 206s, but exp9_withax_combined.py's ACTUAL val split (the one that
    # produced the documented 0.1158) truncates to the first 6500 rows FIRST
    # or (d3_tr=int(len(df_d3)*2/3)=4333), THEN takes the last third of
    # THAT slice -- rows[4333:6500], t≈87-130s. Evaluating against the wrong
    # 2167-row window is a genuinely different segment of the lap, not the
    # same data under a different name -- this was the real source of the
    # Day3 Val yr RMSE mismatch (0.4908 vs 0.1158), not anything in
    # inference_msnn.py (three rounds of fixes there changed nothing because
    # this was never the actual bug).
    if split == 'val':
        df = df.iloc[:6500].reset_index(drop=True)
        tr = int(len(df) * 2 / 3)
        df = df.iloc[tr:].reset_index(drop=True)
    elif split == 'train':
        df = df.iloc[:4333].reset_index(drop=True)
    ds = {
        'yaw_rate': df['IMU_angular_velocity_z'].values.astype(np.float64),
        'vel':      df['odom_speed'].values.astype(np.float64),
        'delta':    df['steering_cmd'].values.astype(np.float64),
        'acc':      df['IMU_linear_acceleration_x'].values.astype(np.float64),
        'accy':     df['IMU_linear_acceleration_y'].values.astype(np.float64),
        'crt':      df['current_motor'].values.astype(np.float64),
        'speed_cmd':df['speed_cmd'].values.astype(np.float64) if 'speed_cmd' in df.columns
                    else df['odom_speed'].values.astype(np.float64),
    }
    return df, ds


# ── MS-NN Evaluation ──────────────────────────────────────────────────────────

def eval_msnn(ds, label='val'):
    """Open-Loop Evaluation. Gibt (yr_pred, ax_pred, yr_true, ax_true) zurück."""
    try:
        from nnodely.basic.relation import NeuObj
        NeuObj.names.clear(); NeuObj.names.append('SampleTime')
    except ImportError:
        print("[WARNUNG] nnodely nicht gefunden — MS-NN Evaluation übersprungen")
        return None

    from inference_msnn import MSNNInference

    infer = MSNNInference(model_dir=MSNN_MODEL)
    nsp = infer.NUM_SAMPLES_PAST
    infer.reset()

    n = len(ds['yaw_rate']) - nsp - 5
    if n <= 0:
        return None

    # BUGFIX 2026-08-13: reset() zero-fills the buffer; without priming it
    # with real rows first, the first ~nsp recorded predictions below are
    # computed from a mostly-zero window (0 is not a valid "no history yet"
    # sentinel, the model reads it as real sensor data). Push rows [0:nsp)
    # first and discard their predictions, so by the time the recorded loop
    # below makes its first call (row nsp), the buffer already holds real
    # data for the full backward window.
    for i in range(nsp):
        infer.step(
            delta=float(ds['delta'][i]), vel=float(ds['vel'][i]),
            acc=float(ds['acc'][i]), accy=float(ds['accy'][i]),
            crt=float(ds['crt'][i]),
        )

    yr_pred, ax_pred = [], []
    for i in range(nsp, nsp + n):
        p = infer.step(
            delta=float(ds['delta'][i]),
            vel=float(ds['vel'][i]),
            acc=float(ds['acc'][i]),
            accy=float(ds['accy'][i]),
            crt=float(ds['crt'][i]),
        )
        yr_pred.append(p[0])
        ax_pred.append(p[1])

    yr_pred = np.array(yr_pred)
    ax_pred = np.array(ax_pred)
    yr_true = ds['yaw_rate'][nsp:nsp + n]
    ax_true = ds['acc'][nsp:nsp + n]

    yr_rmse = float(np.sqrt(np.mean((yr_pred - yr_true) ** 2)))
    ax_rmse = float(np.sqrt(np.mean((ax_pred - ax_true) ** 2)))
    print(f"  [MS-NN {label}] yr={yr_rmse:.4f} ({yr_rmse/PERSIST_YR:.2f}×p)  "
          f"ax={ax_rmse:.4f}")
    return yr_pred, ax_pred, yr_true, ax_true, yr_rmse, ax_rmse


# ── GRU Evaluation (Open-Loop Rollout) ───────────────────────────────────────

def eval_gru(df, label='val'):
    """Open-loop GRU evaluation, architecture-agnostic.

    Reads input/output column names and history length from the checkpoint's
    own config (all GRU checkpoints from training_v2.py store them), so it
    works for the onboard-only 6-in/4-out family as well as the MoCap
    7-in/5-out family (gru_run30_best.pth). Each window of HISTORY real rows
    predicts the NEXT row; predictions are compared against that next row.
    """
    import torch
    from gru_inference import RNN_jomoh_1

    ckpt = torch.load(GRU_CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    in_cols, out_cols = list(cfg['input_columns']), list(cfg['output_columns'])
    HISTORY = int(cfg.get('history', 10))
    model = RNN_jomoh_1(input_size=len(in_cols), output_size=len(out_cols))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  [GRU] {len(in_cols)}in/{len(out_cols)}out, history={HISTORY}, "
          f"inputs={in_cols}")

    def _pick(cands):
        for c in cands:
            if c in out_cols and c in df.columns:
                return c
        raise KeyError(f"none of {cands} in checkpoint outputs {out_cols}")
    yr_col = _pick(['mocap_angular_velocity', 'IMU_angular_velocity_z'])
    ax_col = _pick(['a_x_local', 'IMU_linear_acceleration_x'])
    yr_idx, ax_idx = out_cols.index(yr_col), out_cols.index(ax_col)

    missing = [c for c in in_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV lacks GRU input columns {missing}")
    X = df[in_cols].values.astype(np.float32)
    n = len(X)
    # windows [i-HISTORY+1 .. i] -> prediction for row i+1
    idx = np.arange(HISTORY - 1, n - 1)
    windows = np.stack([X[i - HISTORY + 1:i + 1] for i in idx])
    with torch.no_grad():
        pred = model(torch.from_numpy(windows)).numpy()   # (N, out)
    yr_pred = pred[:, yr_idx]
    ax_pred = pred[:, ax_idx]
    yr_true = df[yr_col].values.astype(np.float64)[idx + 1]
    ax_true = df[ax_col].values.astype(np.float64)[idx + 1]

    yr_rmse = float(np.sqrt(np.mean((yr_pred - yr_true) ** 2)))
    ax_rmse = float(np.sqrt(np.mean((ax_pred - ax_true) ** 2)))
    print(f"  [GRU {label}]   yr={yr_rmse:.4f} ({yr_rmse/PERSIST_YR:.2f}×p)  "
          f"ax={ax_rmse:.4f}   [yr={yr_col}, ax={ax_col}]")
    return yr_pred, ax_pred, yr_true, ax_true, yr_rmse, ax_rmse


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_timeseries(results: dict, ds_label: str, out_path: str):
    """Zeitverlaufsplot: ψ̇ und aₓ für alle Modelle."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f'Fahrzeugdynamik Vergleich — {ds_label}', fontsize=13, fontweight='bold')

    colors = {'GT': '#333333', 'MS-NN': '#1B4FBA', 'GRU': '#15803D'}
    t_gt = None

    for ax_idx, (key, label, unit) in enumerate([('yr', 'ψ̇  Gierrate', 'rad/s'),
                                                   ('ax', 'aₓ Längsbeschleunigung', 'm/s²')]):
        ax = axes[ax_idx]

        if 'gt' in results:
            vals = results['gt'][key]
            t_gt = np.arange(len(vals)) * 0.02
            ax.plot(t_gt, vals, color=colors['GT'], lw=1.0, alpha=0.7,
                    label='Ground Truth', zorder=3)

        for model in ['MS-NN', 'GRU']:
            if model in results and results[model] is not None:
                vals = results[model][key]
                t = np.arange(len(vals)) * 0.02
                rmse = results[model][f'{key}_rmse']
                ax.plot(t, vals, color=colors[model], lw=1.2, alpha=0.85,
                        label=f'{model}  RMSE={rmse:.4f} {unit}')

        ax.set_ylabel(f'{label} [{unit}]')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='#aaa', lw=0.5)

    axes[-1].set_xlabel('Zeit [s]')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Plot gespeichert: {out_path}")


def plot_summary_bar(records: list, out_path: str):
    """Balkendiagramm: RMSE-Vergleich aller Konfigurationen."""
    labels  = [r['label'] for r in records]
    yr_vals = [r['yr_rmse'] for r in records]
    ax_vals = [r['ax_rmse'] for r in records]
    ratios  = [r['yr_rmse'] / PERSIST_YR for r in records]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MS-NN vs GRU — Ergebnisvergleich', fontsize=13, fontweight='bold')

    # ψ̇ RMSE
    bars = axes[0].bar(x, yr_vals, width=w * 1.5, color=['#1B4FBA', '#1B4FBA', '#15803D', '#15803D'],
                       alpha=0.8, edgecolor='white')
    axes[0].axhline(PERSIST_YR, color='#B91C1C', ls='--', lw=1.5, label=f'Persistence {PERSIST_YR:.4f}')
    for bar, val, ratio in zip(bars, yr_vals, ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f'{val:.4f}\n({ratio:.2f}×p)', ha='center', va='bottom', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('ψ̇ RMSE [rad/s]')
    axes[0].set_title('Gierrate ψ̇ RMSE')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # aₓ RMSE
    bars2 = axes[1].bar(x, ax_vals, width=w * 1.5, color=['#1B4FBA', '#1B4FBA', '#15803D', '#15803D'],
                        alpha=0.8, edgecolor='white')
    axes[1].axhline(PERSIST_AX, color='#B91C1C', ls='--', lw=1.5, label=f'Persistence {PERSIST_AX:.4f}')
    for bar, val in zip(bars2, ax_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('aₓ RMSE [m/s²]')
    axes[1].set_title('Längsbeschleunigung aₓ RMSE')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Zusammenfassung gespeichert: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['msnn', 'gru', 'both'], default='both')
    parser.add_argument('--data-dir', default=None,
                        help='Pfad zu processed_data/ (Standard: auto-detect)')
    args = parser.parse_args()

    global DATA_DIR, VAL_CSV, TEST_CSV
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        VAL_CSV  = DATA_DIR / 'session_20260728_163324_seg1_20ms.csv'
        TEST_CSV = DATA_DIR / 'session_20260721_20ms.csv'

    out_dir = _DIR / 'eval_output'
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  14_eval_and_plot.py — MS-NN & GRU Evaluierung")
    print("=" * 60)

    # Daten laden
    print(f"\nLade Datensätze aus: {DATA_DIR}")
    if not VAL_CSV.exists():
        print(f"[FEHLER] {VAL_CSV} nicht gefunden!")
        print("  → --data-dir Pfad zu processed_data/ angeben")
        sys.exit(1)

    df_val, ds_val   = load_csv(VAL_CSV,  split='val')
    df_test, ds_test = load_csv(TEST_CSV) if TEST_CSV.exists() else (None, None)

    records = []
    summary = {}

    # ── MS-NN ────────────────────────────────────────────────────────────────
    if args.model in ('msnn', 'both'):
        print("\n[MS-NN exp9 withax Phase2]")
        msnn_val = eval_msnn(ds_val, 'Day3 Val')
        if msnn_val:
            summary['msnn_val'] = {
                'yr': msnn_val[0], 'ax': msnn_val[1],
                'yr_true': msnn_val[2], 'ax_true': msnn_val[3],
                'yr_rmse': msnn_val[4], 'ax_rmse': msnn_val[5],
            }
            records.append({'label': 'MS-NN\n(Day3 Val)',
                            'yr_rmse': msnn_val[4], 'ax_rmse': msnn_val[5]})

        if ds_test is not None:
            print("  [Cross-Day Day1 Test]")
            msnn_test = eval_msnn(ds_test, 'Day1 Test')
            if msnn_test:
                summary['msnn_test'] = {
                    'yr': msnn_test[0], 'ax': msnn_test[1],
                    'yr_true': msnn_test[2], 'ax_true': msnn_test[3],
                    'yr_rmse': msnn_test[4], 'ax_rmse': msnn_test[5],
                }
                records.append({'label': 'MS-NN\n(Day1 Test)',
                                'yr_rmse': msnn_test[4], 'ax_rmse': msnn_test[5]})

    # ── GRU ──────────────────────────────────────────────────────────────────
    if args.model in ('gru', 'both'):
        if not Path(GRU_CKPT).exists():
            print(f"\n[WARNUNG] GRU Checkpoint nicht gefunden: {GRU_CKPT}")
            print("  → GRU Evaluation übersprungen")
        else:
            print(f"\n[GRU Run30]")
            gru_val = eval_gru(df_val, 'Day3 Val')
            summary['gru_val'] = {
                'yr': gru_val[0], 'ax': gru_val[1],
                'yr_true': gru_val[2], 'ax_true': gru_val[3],
                'yr_rmse': gru_val[4], 'ax_rmse': gru_val[5],
            }
            records.append({'label': 'GRU\n(Day3 Val)',
                            'yr_rmse': gru_val[4], 'ax_rmse': gru_val[5]})

            if ds_test is not None:
                print("  [Cross-Day Day1 Test]")
                gru_test = eval_gru(df_test, 'Day1 Test')
                summary['gru_test'] = {
                    'yr': gru_test[0], 'ax': gru_test[1],
                    'yr_true': gru_test[2], 'ax_true': gru_test[3],
                    'yr_rmse': gru_test[4], 'ax_rmse': gru_test[5],
                }
                records.append({'label': 'GRU\n(Day1 Test)',
                                'yr_rmse': gru_test[4], 'ax_rmse': gru_test[5]})

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nErstelle Plots ...")

    if 'msnn_val' in summary or 'gru_val' in summary:
        gt_yr = summary.get('msnn_val', summary.get('gru_val'))['yr_true']
        gt_ax = summary.get('msnn_val', summary.get('gru_val'))['ax_true']
        plot_results = {
            'gt': {'yr': gt_yr, 'ax': gt_ax},
        }
        if 'msnn_val' in summary:
            plot_results['MS-NN'] = {
                'yr': summary['msnn_val']['yr'],
                'ax': summary['msnn_val']['ax'],
                'yr_rmse': summary['msnn_val']['yr_rmse'],
                'ax_rmse': summary['msnn_val']['ax_rmse'],
            }
        if 'gru_val' in summary:
            gru_n = min(len(summary['gru_val']['yr']), len(gt_yr))
            plot_results['GRU'] = {
                'yr': summary['gru_val']['yr'][:gru_n],
                'ax': summary['gru_val']['ax'][:gru_n],
                'yr_rmse': summary['gru_val']['yr_rmse'],
                'ax_rmse': summary['gru_val']['ax_rmse'],
            }
        plot_timeseries(plot_results, 'Day3 Val (Same-Day)',
                        str(out_dir / 'plot_yawrate_day3.png'))

    if ('msnn_test' in summary or 'gru_test' in summary) and ds_test is not None:
        gt_yr = summary.get('msnn_test', summary.get('gru_test'))['yr_true']
        gt_ax = summary.get('msnn_test', summary.get('gru_test'))['ax_true']
        plot_results = {'gt': {'yr': gt_yr, 'ax': gt_ax}}
        if 'msnn_test' in summary:
            plot_results['MS-NN'] = {
                'yr': summary['msnn_test']['yr'],
                'ax': summary['msnn_test']['ax'],
                'yr_rmse': summary['msnn_test']['yr_rmse'],
                'ax_rmse': summary['msnn_test']['ax_rmse'],
            }
        if 'gru_test' in summary:
            gru_n = min(len(summary['gru_test']['yr']), len(gt_yr))
            plot_results['GRU'] = {
                'yr': summary['gru_test']['yr'][:gru_n],
                'ax': summary['gru_test']['ax'][:gru_n],
                'yr_rmse': summary['gru_test']['yr_rmse'],
                'ax_rmse': summary['gru_test']['ax_rmse'],
            }
        plot_timeseries(plot_results, 'Day1 Test (Cross-Day)',
                        str(out_dir / 'plot_yawrate_day1.png'))

    if records:
        plot_summary_bar(records, str(out_dir / 'results_msnn_vs_gru.png'))

    # CSV speichern
    if records:
        df_out = pd.DataFrame(records)
        df_out['persistence_yr'] = PERSIST_YR
        df_out['yr_ratio'] = df_out['yr_rmse'] / PERSIST_YR
        df_out.to_csv(str(out_dir / 'results_comparison.csv'), index=False)
        print(f"  → CSV gespeichert: {out_dir}/results_comparison.csv")

    print("\n" + "=" * 60)
    print("ERGEBNIS-ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"{'Konfiguration':<22} {'ψ̇ RMSE':>10} {'×p':>7} {'aₓ RMSE':>10}")
    print("-" * 53)
    for r in records:
        lbl = r['label'].replace('\n', ' ')
        ratio = r['yr_rmse'] / PERSIST_YR
        print(f"  {lbl:<20} {r['yr_rmse']:>10.4f} {ratio:>6.2f}× {r['ax_rmse']:>10.4f}")
    print(f"  {'Persistence (Ref)':<20} {PERSIST_YR:>10.4f} {'1.00×':>7} {PERSIST_AX:>10.4f}")
    print(f"  {'Paper (Mohr 2026)':<20} {'—':>10} {'1.21×':>7} {'—':>10}")
    print("=" * 60)
    print(f"\nPlots: {out_dir}/")


if __name__ == '__main__':
    main()
