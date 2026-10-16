"""
06_gru_dynamics_server.py — GRU Dynamics Server (Prozess B)

Analog zu 05_msnn_dynamics_server.py, aber für das GRU-Modell. Läuft NICHT
in nnodely-env (nnodely wird hier nicht gebraucht) -- torch ist bereits im
roboracer conda env importierbar, kein neues env nötig.

IPC-Protokoll (eigene Queue-Dir, NICHT mit dem MS-NN-Server teilen -- siehe
BUGFIX-Hinweis unten):
  Prozess A (Isaac Sim) schreibt:
    queue_dir/req_NNNNNN.json
    {"step": 42, "speed": 2.0, "yr": 0.1, "ax": 0.2, "ay": -0.1,
     "steering_cmd": 0.05, "current_motor": 55.0}
    {"step": -1, "reset": true}    ← Episode-Reset

  Dieser Server schreibt:
    queue_dir/res_NNNNNN.json
    {"step": 42, "speed_pred": 2.1, "yr_pred": 0.15, "ax_pred": 0.3, "ay_pred": -0.05}

  Prozess A löscht req_NNNNNN.json nach Erhalt der Antwort.

WICHTIG -- Feldnamen 2026-08-13: gru_run30_best.pth ist NICHT das
dokumentierte MoCap-Modell (vx,vy,yr,ax,ay,speed_cmd,steering_cmd, 7in/5out)
-- es ist tatsächlich ein onboard-only Checkpoint (6in/4out), siehe
gru_inference.py's Moduldocstring für die volle Herleitung. Request-Felder
hier matchen GRUInference.step()'s tatsächliche Signatur:
  speed, yr, ax, ay, steering_cmd, current_motor  (NICHT vx/vy/speed_cmd)

BUGFIX 2026-08-13 (Closed-Loop-Vorbereitung): die Antwort gab bisher nur
yr_pred/ax_pred zurück und verwarf speed_pred/ay_pred, obwohl
GRUInference.step() alle 4 bereits liefert. Für echten Closed-Loop-Rollout
(alle 4 Zustandskanäle werden zurückgefüttert, nicht nur yr/ax -- sonst ist
es kein echter geschlossener Kreis und die CL-RMSE-Zahl wäre künstlich
geschönt, weil speed/ay weiter auf Ground Truth "geankert" blieben) braucht
der Aufrufer alle 4 Werte. Response jetzt vollständig.

Starten (ZUERST, vor Isaac Sim):
  conda run -n roboracer python3 06_gru_dynamics_server.py \
    --checkpoint trained_models/gru_run30_best.pth \
    --queue-dir isaac_sim_output/gru_queue
"""
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)  # logs stay readable when redirected to a file
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
        default='trained_models/gru_run30_best.pth',
        help='Pfad zum GRU .pth Checkpoint')
    parser.add_argument('--queue-dir',
        default='isaac_sim_output/gru_queue',
        # BUGFIX 2026-08-13 (same class of bug as 05_msnn_dynamics_server.py):
        # MUSS sich vom MS-NN-Server-Queue-Dir unterscheiden, sonst
        # Cross-Talk zwischen den beiden Modellen (unterschiedliche Feld-
        # namen -> KeyError, nicht sofort offensichtlich woher).
        help='MUSS sich vom MS-NN-Server-Queue-Dir unterscheiden')
    parser.add_argument('--poll-interval', type=float, default=0.02,
        help='Sekunden zwischen Queue-Abfragen')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[2] / 'dynamics' / 'msnn'
    sys.path.insert(0, str(script_dir))

    from gru_inference import GRUInference

    print(f"[gru_server] Lade Modell: {args.checkpoint}")
    infer = GRUInference(checkpoint=args.checkpoint)
    infer.reset()
    print(f"[gru_server] Modell geladen. Warte auf Anfragen in {args.queue_dir} ...")

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = queue_dir / 'STOP'

    n_served = 0
    while True:
        if stop_flag.exists():
            print('[gru_server] STOP gesehen, beende.')
            stop_flag.unlink()
            break

        requests = sorted(queue_dir.glob('req_*.json'))
        if not requests:
            time.sleep(args.poll_interval)
            continue

        req_path = requests[0]
        idx_str  = req_path.stem.split('_', 1)[1]
        res_path = queue_dir / f'res_{idx_str}.json'
        tmp_path = queue_dir / f'res_{idx_str}.json.tmp'

        try:
            req = json.loads(req_path.read_text())

            if req.get('reset', False):
                infer.reset()
                res = {'step': req.get('step', -1), 'reset_ack': True,
                       'speed_pred': 0.0, 'yr_pred': 0.0, 'ax_pred': 0.0, 'ay_pred': 0.0}
            else:
                out = infer.step(
                    speed=float(req['speed']),
                    yr=float(req['yr']),
                    ax=float(req['ax']),
                    ay=float(req['ay']),
                    steering_cmd=float(req['steering_cmd']),
                    current_motor=float(req['current_motor']),
                )
                res = {'step': req['step'], 'speed_pred': out['speed'],
                       'yr_pred': out['yr'], 'ax_pred': out['ax'], 'ay_pred': out['ay']}

            tmp_path.write_text(json.dumps(res))
            tmp_path.rename(res_path)
            n_served += 1
            if n_served % 50 == 0:
                print(f'[gru_server] {n_served} Schritte bedient '
                      f'(letzter: step={res["step"]})')

        except Exception as e:
            print(f'[gru_server] Fehler bei {req_path.name}: {e}')
            tmp_path.write_text(json.dumps({'step': -1, 'speed_pred': 0.0, 'yr_pred': 0.0,
                                            'ax_pred': 0.0, 'ay_pred': 0.0, 'error': str(e)}))
            tmp_path.rename(res_path)
        finally:
            req_path.unlink(missing_ok=True)

    print(f'[gru_server] Fertig. {n_served} Schritte insgesamt bedient.')


if __name__ == '__main__':
    main()
