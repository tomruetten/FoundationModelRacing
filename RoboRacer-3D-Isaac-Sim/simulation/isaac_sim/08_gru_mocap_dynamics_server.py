"""
08_gru_mocap_dynamics_server.py — GRU-MoCap Dynamics Server (Prozess B4)

Dritte GRU-Variante für V1 (2026-08-13), neben Open-Loop und Closed-Loop
der onboard-only Architektur (06_gru_dynamics_server.py): diese hier lädt
einen Checkpoint mit echter 7in/5out MoCap-Zustands-Architektur
(GRUMoCapInference in gru_inference.py) -- Details/Provenienz siehe dortige
Klassen-Docstring. Nur Open-Loop, kein Closed-Loop-Pfad für diese Variante.

IPC-Protokoll (eigene Queue-Dir):
  Prozess A (Isaac Sim) schreibt:
    queue_dir/req_NNNNNN.json
    {"step": 42, "vx": 2.0, "vy": 0.0, "yr": 0.1, "ax": 0.2, "ay": -0.1,
     "speed_cmd": 2.0, "steering_cmd": 0.05}
    {"step": -1, "reset": true}    ← Episode-Reset

  Dieser Server schreibt:
    queue_dir/res_NNNNNN.json
    {"step": 42, "vx_pred": 2.1, "vy_pred": 0.02, "yr_pred": 0.11,
     "ax_pred": 0.22, "ay_pred": -0.08}

Starten (ZUERST, vor Isaac Sim):
  conda run -n roboracer python3 08_gru_mocap_dynamics_server.py \
    --checkpoint ../../dynamics/msnn/local_run___20260731204259241445.pth \
    --queue-dir isaac_sim_output/gru_mocap_queue
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
        default='local_run___20260731204259241445.pth',
        help='Pfad zum MoCap-GRU .pth Checkpoint')
    parser.add_argument('--queue-dir',
        default='isaac_sim_output/gru_mocap_queue',
        help='MUSS sich von den anderen Server-Queue-Dirs unterscheiden')
    parser.add_argument('--poll-interval', type=float, default=0.02)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[2] / 'dynamics' / 'msnn'
    sys.path.insert(0, str(script_dir))

    from gru_inference import GRUMoCapInference

    print(f"[gru_mocap_server] Lade Modell: {args.checkpoint}")
    infer = GRUMoCapInference(checkpoint=args.checkpoint)
    infer.reset()
    print(f"[gru_mocap_server] Modell geladen. Warte auf Anfragen in {args.queue_dir} ...")

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = queue_dir / 'STOP'

    n_served = 0
    while True:
        if stop_flag.exists():
            print('[gru_mocap_server] STOP gesehen, beende.')
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
                       'vx_pred': 0.0, 'vy_pred': 0.0, 'yr_pred': 0.0,
                       'ax_pred': 0.0, 'ay_pred': 0.0}
            else:
                out = infer.step(
                    vx=float(req['vx']),
                    vy=float(req['vy']),
                    yr=float(req['yr']),
                    ax=float(req['ax']),
                    ay=float(req['ay']),
                    speed_cmd=float(req['speed_cmd']),
                    steering_cmd=float(req['steering_cmd']),
                )
                res = {'step': req['step'], 'vx_pred': out['vx'], 'vy_pred': out['vy'],
                       'yr_pred': out['yr'], 'ax_pred': out['ax'], 'ay_pred': out['ay']}

            tmp_path.write_text(json.dumps(res))
            tmp_path.rename(res_path)
            n_served += 1
            if n_served % 50 == 0:
                print(f'[gru_mocap_server] {n_served} Schritte bedient '
                      f'(letzter: step={res["step"]})')

        except Exception as e:
            print(f'[gru_mocap_server] Fehler bei {req_path.name}: {e}')
            tmp_path.write_text(json.dumps({'step': -1, 'vx_pred': 0.0, 'vy_pred': 0.0,
                                            'yr_pred': 0.0, 'ax_pred': 0.0, 'ay_pred': 0.0,
                                            'error': str(e)}))
            tmp_path.rename(res_path)
        finally:
            req_path.unlink(missing_ok=True)

    print(f'[gru_mocap_server] Fertig. {n_served} Schritte insgesamt bedient.')


if __name__ == '__main__':
    main()
