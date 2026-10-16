"""
05_msnn_dynamics_server.py — MS-NN Dynamics Server (Prozess B)

Dieses Skript läuft in der nnodely-Umgebung und beantwortet Anfragen
von Isaac Sim (Prozess A, 05_msnn_dynamics_isaac.py) über eine File-Queue.

Warum separater Prozess?
  Isaac Sim bringt eine eigene Python/torch-Umgebung.
  nnodely==1.5.4 (CPU-only) könnte zwar in Isaac Sim Python importiert
  werden, aber dann müsste nnodely dort installiert sein.
  Sicherste Lösung: separater Prozess, identisches IPC-Muster wie beim
  Camera-Splat-Server.

IPC-Protokoll:
  Prozess A (Isaac Sim) schreibt:
    queue_dir/req_NNNNNN.json
    {"step": 42, "delta": 0.1, "vel": 2.0, "acc": 0.5, "accy": -0.3, "crt": 60.0}
    {"step": -1, "reset": true}    ← Episode-Reset

  Dieser Server schreibt:
    queue_dir/res_NNNNNN.json
    {"step": 42, "yr_pred": 0.15, "ax_pred": 0.3}

  Prozess A löscht req_NNNNNN.json nach Erhalt der Antwort.

Starten (ZUERST, vor Isaac Sim):
  conda run -n nnodely-env python3 05_msnn_dynamics_server.py \
    --model trained_models/msnn_exp9_phase2_backup \
    --queue-dir /tmp/isaac_dyn_queue
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
        default='trained_models/msnn_exp9_phase2_backup',
        help='Pfad zu trained_models/msnn_expX_... (enthält net.json)')
    parser.add_argument('--queue-dir',
        default='/tmp/isaac_dyn_queue')
    parser.add_argument('--poll-interval', type=float, default=0.02,
        help='Sekunden zwischen Queue-Abfragen')
    args = parser.parse_args()

    # nnodely-env Pfad muss stimmen
    # FIXED 2026-08-25: this pointed at 'dynamics/training', a directory
    # that no longer exists at this level (V2/dynamics/training) -- the repo
    # was reorganized at some point and inference_msnn.py now lives under
    # 'dynamics/msnn' instead (confirmed: V2/dynamics/msnn/inference_msnn.py
    # exists, V2/dynamics/training does not). Discovered while wiring up the
    # MS-NN server locally to validate --live-yr-filter (see report/
    # task_report_heading_noise_optimization_20260825.txt, not included in
    # this repo) -- unrelated to that feature, just a stale path left over
    # from the reorg.
    script_dir = Path(__file__).resolve().parents[2] / 'dynamics' / 'msnn' / 'src'
    sys.path.insert(0, str(script_dir))

    from inference_msnn import MSNNInference

    print(f"[dyn_server] Lade Modell: {args.model}")
    infer = MSNNInference(model_dir=args.model)
    infer.reset()
    print(f"[dyn_server] Modell geladen. Warte auf Anfragen in {args.queue_dir} ...")

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = queue_dir / 'STOP'

    n_served = 0
    while True:
        if stop_flag.exists():
            print('[dyn_server] STOP gesehen, beende.')
            stop_flag.unlink()
            break

        requests = sorted(queue_dir.glob('req_*.json'))
        if not requests:
            time.sleep(args.poll_interval)
            continue

        req_path = requests[0]
        idx_str  = req_path.stem.split('_', 1)[1]
        res_path = queue_dir / f'res_{idx_str}.json'

        try:
            req = json.loads(req_path.read_text())

            if req.get('reset', False):
                infer.reset()
                res = {'step': req.get('step', -1), 'reset_ack': True,
                       'yr_pred': 0.0, 'ax_pred': 0.0}
            else:
                yr_pred, ax_pred = infer.step(
                    delta=float(req['delta']),
                    vel=float(req['vel']),
                    acc=float(req['acc']),
                    accy=float(req['accy']),
                    crt=float(req['crt']),
                )
                res = {'step': req['step'], 'yr_pred': yr_pred, 'ax_pred': ax_pred}

            res_path.write_text(json.dumps(res))
            n_served += 1
            if n_served % 50 == 0:
                print(f'[dyn_server] {n_served} Schritte bedient '
                      f'(letzter: step={res["step"]})')

        except Exception as e:
            print(f'[dyn_server] Fehler bei {req_path.name}: {e}')
            res_path.write_text(json.dumps({'step': -1, 'yr_pred': 0.0,
                                            'ax_pred': 0.0, 'error': str(e)}))
        finally:
            req_path.unlink(missing_ok=True)

    print(f'[dyn_server] Fertig. {n_served} Schritte insgesamt bedient.')


if __name__ == '__main__':
    main()
