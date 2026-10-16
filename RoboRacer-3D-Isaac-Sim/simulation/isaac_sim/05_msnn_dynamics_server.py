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
    --queue-dir isaac_sim_output/msnn_queue

BEHOBEN (2026-08-13): MS-NN Day3-Val yr RMSE wich lange bei 0.408 statt der
dokumentierten 0.1158 ab. Root Cause (endgültig, per diag_eval.py + Batch-
Eval gegen net.json bestätigt -- nicht der Val-Split, nicht loadModel()/
neuralizeModel()-Reihenfolge, nicht die Checkpoint-Gewichte, alle drei
einzeln durch Quellcode-/Datenvergleich ausgeschlossen): inference_msnn.py's
step()-Puffer war mit max(20,15)+5=25 Zeilen 4 Zeilen zu lang. Das Modell
braucht laut net.json['Info']['ns'] exakt 20 Rückwärts- + 1 Vorwärtsschritt
(=21 Zeilen). getSamples(index=0, window=1) liefert bei einem 25-Zeilen-
Puffer das ÄLTESTE gültige Fenster (Zeilen 0:21), nicht das neueste -- jede
Vorhersage lief also permanent 4 Zeitschritte hinterher. Fix: Puffergröße
exakt aus dem geladenen Modell lesen (self._ns_backward + self._ns_forward),
nicht mit Sicherheitsmarge raten. Nach dem Fix: yr=0.1158 (exakt der
dokumentierte Wert), Day1 Cross-Day yr=0.1731. Details:
report/Fortschrittsbericht_VehicleDynamics_V1_20260813_DE.txt
"""
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)  # logs stay readable when redirected to a file
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
        default='isaac_sim_output/msnn_queue',
        # BUGFIX 2026-08-13: was /tmp/isaac_dyn_queue -- a GRU dynamics
        # server for the same V1 demo needs its own distinct default (see
        # 06_gru_dynamics_server.py), otherwise two servers sharing one
        # default queue-dir would silently cross-talk (GRU consuming MS-NN's
        # request or vice versa) instead of failing loudly.
        help='MUSS sich vom GRU-Server-Queue-Dir unterscheiden')
    parser.add_argument('--poll-interval', type=float, default=0.02,
        help='Sekunden zwischen Queue-Abfragen')
    args = parser.parse_args()

    # nnodely-env Pfad muss stimmen
    script_dir = Path(__file__).resolve().parents[2] / 'dynamics' / 'training'
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
        # BUGFIX 2026-08-13: write to a .tmp file then rename() (atomic on
        # POSIX) instead of writing res_path directly -- the direct-write
        # version let the client (polling for res_path to exist) potentially
        # read a partially-written file. Same fix already applied to the
        # camera splat server this session (splat_render_server_okession.py).
        tmp_path = queue_dir / f'res_{idx_str}.json.tmp'

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

            tmp_path.write_text(json.dumps(res))
            tmp_path.rename(res_path)
            n_served += 1
            if n_served % 50 == 0:
                print(f'[dyn_server] {n_served} Schritte bedient '
                      f'(letzter: step={res["step"]})')

        except Exception as e:
            print(f'[dyn_server] Fehler bei {req_path.name}: {e}')
            tmp_path.write_text(json.dumps({'step': -1, 'yr_pred': 0.0,
                                            'ax_pred': 0.0, 'error': str(e)}))
            tmp_path.rename(res_path)
        finally:
            req_path.unlink(missing_ok=True)

    print(f'[dyn_server] Fertig. {n_served} Schritte insgesamt bedient.')


if __name__ == '__main__':
    main()
