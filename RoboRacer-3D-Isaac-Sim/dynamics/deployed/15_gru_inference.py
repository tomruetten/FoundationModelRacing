"""
15_gru_inference.py — GRU Run30 Inference API für Isaac Sim
============================================================
Analog zu 13_inference_msnn.py, aber für das GRU-Modell.

Architektur: RNN_jomoh_1
  GRU(input=7, hidden=100) → FC(100,100)+LeakyReLU → FC(100,100)+LeakyReLU → FC(100,5)

Inputs (7 Kanäle, 20ms Schritte):
  v_x_local, v_y_local, mocap_angular_velocity,
  a_x_local, a_y_local, speed_cmd, steering_cmd

Outputs (5 Kanäle):
  v_x_local, v_y_local, mocap_angular_velocity, a_x_local, a_y_local
  Index 2 = ψ̇ (Gierrate) [rad/s]
  Index 3 = aₓ            [m/s²]

Vergleich mit MS-NN:
  GRU   → Cross-Day CL-RMSE = 0.4448 rad/s  (besser für neue Tage/Strecken)
  MS-NN → Same-Day  OL-RMSE = 0.1158 rad/s  (besser auf bekannter Strecke)
  → Isaac Sim V1: beide parallel nutzen, Ergebnisse vergleichen

Checkpoint: trained_models/gru_run30_best.pth  (liegt im selben Ordner)

Smoke-Test:
  conda activate f1tenth_dynamics
  python3 15_gru_inference.py
"""
import os
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).parent


# ── Architektur (identisch mit Training, RNN_jomoh_1) ────────────────────────

class RNN_jomoh_1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru1      = nn.GRU(input_size, 100, batch_first=True)
        self.fc1       = nn.Linear(100, 100)
        self.fc3       = nn.Linear(100, 100)
        self.fc5       = nn.Linear(100, output_size)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        gru_out, _ = self.gru1(x)
        x = gru_out[:, -1, :]
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc3(x))
        return self.fc5(x)


# ── Inference-Klasse ──────────────────────────────────────────────────────────

class GRUInference:
    """
    Zustandsbehaftete GRU-Inferenz für Isaac Sim (20ms Schritte).

    Spalten-Reihenfolge (Training):
      Inputs:  [v_x_local, v_y_local, mocap_angular_velocity,
                a_x_local, a_y_local, speed_cmd, steering_cmd]
      Outputs: [v_x_local, v_y_local, mocap_angular_velocity,
                a_x_local, a_y_local]
    """

    INPUT_COLS  = ['vx', 'vy', 'yr', 'ax', 'ay', 'speed_cmd', 'steering_cmd']
    OUTPUT_COLS = ['vx', 'vy', 'yr', 'ax', 'ay']
    HISTORY     = 10   # Anzahl vergangener Zeitschritte als Kontext

    def __init__(self, checkpoint: str = None, device: str = 'cpu'):
        if checkpoint is None:
            checkpoint = str(_SCRIPT_DIR / 'trained_models' / 'gru_run30_best.pth')

        self.device = torch.device(device)
        self.model  = RNN_jomoh_1(input_size=7, output_size=5).to(self.device)

        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        # Checkpoint kann direkt state_dict oder {'model_state_dict': ...} sein
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[GRUInference] Modell geladen: {checkpoint}")
        self.reset()

    def reset(self):
        """Sliding-Window-Buffer leeren. Vor jeder Episode aufrufen."""
        # Buffer: deque mit je HISTORY Nullvektoren (eine Zeile pro Zeitschritt)
        zero = [0.0] * len(self.INPUT_COLS)
        self._buf = deque([zero[:] for _ in range(self.HISTORY)],
                          maxlen=self.HISTORY)

    def step(self,
             vx: float, vy: float, yr: float,
             ax: float, ay: float,
             speed_cmd: float, steering_cmd: float
             ) -> dict:
        """
        Einen Zeitschritt vorhersagen.

        Args:
            vx, vy      : Längs-/Quergeschwindigkeit [m/s]
            yr          : Gierrate (mocap_angular_velocity) [rad/s]
            ax, ay      : Längs-/Querbeschleunigung [m/s²]
            speed_cmd   : Sollgeschwindigkeit [m/s]
            steering_cmd: Lenkwinkelbefehl [rad]

        Returns dict mit:
            'vx', 'vy', 'yr', 'ax', 'ay'  (vorhergesagte nächste Werte)
            Für Isaac Sim am wichtigsten: 'yr' und 'ax'
        """
        row = [vx, vy, yr, ax, ay, speed_cmd, steering_cmd]
        self._buf.append(row)

        x = torch.tensor(list(self._buf), dtype=torch.float32)  # (HISTORY, 7)
        x = x.unsqueeze(0).to(self.device)                      # (1, HISTORY, 7)

        with torch.no_grad():
            out = self.model(x).squeeze(0).cpu().numpy()        # (5,)

        return {
            'vx': float(out[0]),
            'vy': float(out[1]),
            'yr': float(out[2]),   # ← Gierrate ψ̇ [rad/s]
            'ax': float(out[3]),   # ← Längsbeschleunigung aₓ [m/s²]
            'ay': float(out[4]),
        }


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ckpt = _SCRIPT_DIR / 'trained_models' / 'gru_run30_best.pth'
    if not ckpt.exists():
        print(f"[FEHLER] Checkpoint nicht gefunden: {ckpt}")
        print("  → trained_models/gru_run30_best.pth ins selbe Verzeichnis legen")
        sys.exit(1)

    gru = GRUInference(str(ckpt))
    gru.reset()

    print("\nSmoke Test — 10 Schritte mit konstanten Eingaben:")
    print(f"{'Schritt':>6}  {'yr_pred':>9}  {'ax_pred':>9}  {'vx_pred':>9}")
    for i in range(10):
        out = gru.step(
            vx=2.0 + i * 0.05,
            vy=0.0,
            yr=0.1 * (i % 3),
            ax=0.2,
            ay=-0.1,
            speed_cmd=2.0,
            steering_cmd=0.05 * (i % 4),
        )
        print(f"  {i:4d}   {out['yr']:+9.4f}  {out['ax']:+9.4f}  {out['vx']:+9.4f}")

    print("\n✓ GRUInference smoke test OK")
    print(f"  Für Isaac Sim: out['yr'] = ψ̇ [rad/s], out['ax'] = aₓ [m/s²]")
