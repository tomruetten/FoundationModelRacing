"""
compile_results.py — wartet auf DD + GRU-8Hz und gibt den finalen
Vergleichsbericht aus.

Berechnet CL-RMSE direkt aus den gespeicherten Modell-Dateien, um das
Problem zu umgehen, dass training.py die Ergebnisse nicht ausgibt.
"""
import sys, os, glob, re, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

wandb_stub = types.ModuleType('wandb')
class _FR:
    summary = {}
    def log(self,d): pass
wandb_stub.run = _FR()
wandb_stub.init = lambda **kw: None
sys.modules['wandb'] = wandb_stub

import numpy as np
import torch
import pandas as pd

WORK_DIR  = os.path.dirname(os.path.abspath(__file__))
REPORT    = '/tmp/overnight_results.txt'

INPUTS    = ["v_x_local","v_y_local","mocap_angular_velocity","a_x_local","a_y_local","speed_cmd","steering_cmd"]
OUTPUTS   = ["v_x_local","v_y_local","mocap_angular_velocity","a_x_local","a_y_local"]
HISTORY   = 10
device    = 'cuda' if torch.cuda.is_available() else 'cpu'

RUN22_MODEL = os.path.join(WORK_DIR, 'trained_models', 'local_run___20260728212006410245.pth')
GRU8HZ_DIR  = os.path.join(WORK_DIR, 'trained_models')
DD_LOG      = '/tmp/dd_car02_log.txt'
GRU8HZ_LOG  = '/tmp/gru_8hz_log.txt'

TEST_1HZ    = os.path.join(WORK_DIR, 'processed_data',      'session_20260721_20ms.csv')
TEST_8HZ    = os.path.join(WORK_DIR, 'processed_data_8hz',  'session_20260721_20ms.csv')
DD_MODEL    = os.path.join(WORK_DIR, 'trained_models', 'deep_dynamics_car02', 'best_model.pth')


def log(msg):
    from datetime import datetime
    ts   = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPORT, 'a') as f:
        f.write(line + '\n')


def closed_loop_rmse_gru(model, test_file):
    df         = pd.read_csv(test_file, index_col=0)
    inputs_np  = df[INPUTS].to_numpy(dtype=np.float32)
    outputs_np = df[OUTPUTS].to_numpy(dtype=np.float32)
    T, H       = len(df), HISTORY
    out_idx    = list(range(len(OUTPUTS)))

    inputs_t   = torch.from_numpy(inputs_np).to(device)
    current_seq = inputs_t[:H].clone()
    preds      = []

    model.eval()
    with torch.no_grad():
        for step in range(T - H):
            x    = current_seq.unsqueeze(0)
            pred = model(x).squeeze(0)
            preds.append(pred)
            next_row = inputs_t[H + step].clone()
            next_row[out_idx] = pred
            current_seq = torch.cat([current_seq[1:], next_row.unsqueeze(0)], dim=0)

    preds_np = torch.stack(preds, dim=0).cpu().numpy()
    gt_np    = outputs_np[H:]
    rmse_ch  = np.sqrt(np.mean((preds_np - gt_np) ** 2, axis=0))
    return float(np.mean(rmse_ch)), rmse_ch


def eval_run22():
    from RNN_architectures import RNN_jomoh_1
    log("Evaluating Run22 ...")
    if not os.path.exists(RUN22_MODEL):
        log("  Run22 model not found!")
        return None
    model = RNN_jomoh_1(input_size=len(INPUTS), output_size=len(OUTPUTS)).to(device)
    ckpt  = torch.load(RUN22_MODEL, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    total, rmse_ch = closed_loop_rmse_gru(model, TEST_1HZ)
    for col, r in zip(OUTPUTS, rmse_ch):
        log(f"  Run22 {col:<26} {r:.4f}")
    log(f"  Run22 Total CL-RMSE: {total:.4f}")
    return total


def eval_gru_8hz():
    """Find the latest model saved after Run22, assume it's GRU-8Hz."""
    from RNN_architectures import RNN_jomoh_1
    log("Evaluating GRU-8Hz ...")

    pths = glob.glob(os.path.join(GRU8HZ_DIR, '*.pth'))
    # Exclude Run22's known model
    pths = [p for p in pths if 'deep_dynamics' not in p
            and os.path.basename(p) != 'local_run___20260728212006410245.pth']
    if not pths:
        log("  GRU-8Hz model not found!")
        return None

    def _ts(p):
        m = re.search(r'(\d{15,})', os.path.basename(p))
        return m.group(1) if m else '0'
    model_path = max(pths, key=_ts)
    log(f"  Model: {os.path.basename(model_path)}")

    model = RNN_jomoh_1(input_size=len(INPUTS), output_size=len(OUTPUTS)).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)

    total, rmse_ch = closed_loop_rmse_gru(model, TEST_8HZ)
    for col, r in zip(OUTPUTS, rmse_ch):
        log(f"  GRU-8Hz {col:<24} {r:.4f}")
    log(f"  GRU-8Hz Total CL-RMSE: {total:.4f}")
    return total


def eval_dd():
    """Parse DD CL-RMSE from its log file (run_deep_dynamics_car02.py prints it)."""
    log("Reading Deep Dynamics result ...")
    if not os.path.exists(DD_LOG):
        log("  DD log not found!")
        return None
    txt = open(DD_LOG).read()
    # run_deep_dynamics_car02.py prints: "  mean_CL_RMSE       X.XXXX"
    m = re.search(r'mean_CL_RMSE\s+(\d\.\d+)', txt)
    if not m:
        m = re.search(r'CL-RMSE=(\d\.\d+)', txt)
    if m:
        rmse = float(m.group(1))
        log(f"  DD CL-RMSE: {rmse:.4f}")
        return rmse
    log("  DD CL-RMSE not found in log — training may still be running.")
    return None


def wait_for_job(log_file, finish_markers, label, timeout_h=4):
    """Poll log file until a finish marker appears."""
    log(f"Waiting for {label} ...")
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        if os.path.exists(log_file):
            txt = open(log_file).read()
            if any(m in txt for m in finish_markers):
                log(f"  {label} finished.")
                return True
        time.sleep(60)
    log(f"  WARNING: {label} timed out.")
    return False


def final_report(results):
    baseline = 0.8420
    log("\n" + "=" * 70)
    log("  FINAL OVERNIGHT RESULTS — " + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))
    log("=" * 70)
    log(f"\n  {'Experiment':<44} {'CL-RMSE':>8}  {'vs Run12':>10}  Änderung")
    log("  " + "-" * 68)

    rows = [
        ("Run12  GRU  Day2 1.15Hz (baseline)",         baseline,                                     "—"),
        ("Run22  GRU  Day2+Day3 1.15Hz",               results.get('run22'),                          "mehr Daten"),
        ("DD     DeepDyn Day2+Day3 1.15Hz",            results.get('dd'),                             "physikal. Modell"),
        ("GRU8Hz GRU  Day2 8Hz (train=val=test)",      results.get('gru8hz'),                         "Filterkorrektur"),
    ]

    for name, rmse, var in rows:
        if rmse is None:
            log(f"  {name:<44} {'N/A':>8}             {var}")
        else:
            delta  = rmse - baseline
            marker = '↓ better' if delta < 0 else '↑ worse '
            log(f"  {name:<44} {rmse:>8.4f}  {delta:>+.4f} {marker}  {var}")

    log("\n" + "=" * 70)
    log(f"  Full report saved to: {REPORT}")


def main():
    log("compile_results.py — gathering final results")

    results = {}

    # Run22 — model already saved
    results['run22'] = eval_run22()

    # Wait for GRU-8Hz (orchestration subprocess; log file shows epoch progress)
    wait_for_job(GRU8HZ_LOG,
                 finish_markers=['Training completed', 'Early stopping triggered.'],
                 label='GRU-8Hz',
                 timeout_h=4)
    results['gru8hz'] = eval_gru_8hz()

    # Wait for Deep Dynamics
    wait_for_job(DD_LOG,
                 finish_markers=['Plots saved to', 'CL-RMSE='],
                 label='Deep Dynamics',
                 timeout_h=5)
    results['dd'] = eval_dd()

    final_report(results)


if __name__ == '__main__':
    main()
