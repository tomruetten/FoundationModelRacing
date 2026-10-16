"""
15_gru_inference.py — GRU "Run30" Inference API für Isaac Sim
============================================================
Analog zu 13_inference_msnn.py, aber für das GRU-Modell.

BUGFIX 2026-08-13: die Datei hier ging von der DOKUMENTIERTEN Run30-
Architektur aus (7in/5out, MoCap v_x_local/v_y_local + speed_cmd). Das
tatsächlich auf der Maschine liegende `gru_run30_best.pth` (config direkt
aus dem Checkpoint gelesen, per torch.load) ist aber ein ANDERES Modell:
  input_columns:  ['odom_speed', 'IMU_angular_velocity_z',
                    'IMU_linear_acceleration_x', 'IMU_linear_acceleration_y',
                    'steering_cmd', 'current_motor']               (6 Kanäle)
  output_columns: ['odom_speed', 'IMU_angular_velocity_z',
                    'IMU_linear_acceleration_x', 'IMU_linear_acceleration_y'] (4 Kanäle)
Das ist kein MoCap-Modell -- deckt sich exakt mit dem in TECHNICAL_NOTES.md separat
dokumentierten "Run37, onboard-only (no MoCap)". Auf Nutzerentscheidung
(2026-08-13): nicht nach dem echten Run30 suchen, sondern diese Datei an
die tatsächliche Checkpoint-Architektur anpassen und damit weiterarbeiten
-- V1 muss die 0.4448 CL-RMSE nicht exakt reproduzieren, nur die Pipeline
zum Laufen bringen. Spaltenreihenfolge unten 1:1 aus dem Checkpoint-eigenen
`config['input_columns']`/`config['output_columns']` übernommen -- die
GRU-Gewichtsmatrix ist positionsbasiert, falsche Reihenfolge lädt zwar
ohne Fehler, liefert aber stillschweigend Unsinn.

Architektur: RNN_jomoh_1
  GRU(input=6, hidden=100) → FC(100,100)+LeakyReLU → FC(100,100)+LeakyReLU → FC(100,4)

Inputs (6 Kanäle, 20ms Schritte, exakte Reihenfolge wichtig):
  odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x,
  IMU_linear_acceleration_y, steering_cmd, current_motor

Outputs (4 Kanäle):
  odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x, IMU_linear_acceleration_y
  Index 1 = ψ̇ (Gierrate) [rad/s]
  Index 2 = aₓ            [m/s²]

Vergleich mit MS-NN:
  GRU (dieser Checkpoint) → onboard-only, kein dokumentierter CL-RMSE-Wert
                             für exakt diese Gewichte (siehe Hinweis oben)
  MS-NN → Same-Day  OL-RMSE = 0.1158 rad/s  (besser auf bekannter Strecke)
  → Isaac Sim V1: beide parallel nutzen, Ergebnisse vergleichen

Checkpoint: trained_models/gru_run30_best.pth  (liegt im selben Ordner)

Smoke-Test:
  conda run -n roboracer python3 15_gru_inference.py
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

    Spalten-Reihenfolge (1:1 aus dem Checkpoint-eigenen config['input_columns']/
    config['output_columns'] übernommen -- NICHT die ursprünglich dokumentierte
    MoCap-Run30-Architektur, siehe Modul-Docstring):
      Inputs:  [odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x,
                IMU_linear_acceleration_y, steering_cmd, current_motor]
      Outputs: [odom_speed, IMU_angular_velocity_z, IMU_linear_acceleration_x,
                IMU_linear_acceleration_y]
    """

    INPUT_COLS  = ['speed', 'yr', 'ax', 'ay', 'steering_cmd', 'current_motor']
    OUTPUT_COLS = ['speed', 'yr', 'ax', 'ay']
    HISTORY     = 10   # Anzahl vergangener Zeitschritte als Kontext

    def __init__(self, checkpoint: str = None, device: str = 'cpu'):
        if checkpoint is None:
            checkpoint = str(_SCRIPT_DIR / 'trained_models' / 'gru_run30_best.pth')

        self.device = torch.device(device)
        self.model  = RNN_jomoh_1(input_size=len(self.INPUT_COLS),
                                   output_size=len(self.OUTPUT_COLS)).to(self.device)

        # BUGFIX 2026-08-13: the roboracer conda env's torch (2.6.0) changed
        # torch.load()'s weights_only default from False to True; this
        # checkpoint's saved 'config' dict contains numpy arrays
        # (data_mean/data_std), which aren't in torch's default safe-globals
        # allowlist, causing an UnpicklingError. Checkpoint is this
        # project's own training output (trusted source), safe to disable.
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
             speed: float, yr: float, ax: float, ay: float,
             steering_cmd: float, current_motor: float,
             ) -> dict:
        """
        Einen Zeitschritt vorhersagen.

        Args:
            speed         : odom_speed, Radencoder-Geschwindigkeit [m/s]
            yr            : Gierrate (IMU_angular_velocity_z) [rad/s]
            ax, ay        : Längs-/Querbeschleunigung (IMU) [m/s²]
            steering_cmd  : Lenkwinkelbefehl [rad]
            current_motor : Motorstrom [A]

        Returns dict mit:
            'speed', 'yr', 'ax', 'ay'  (vorhergesagte nächste Werte)
            Für Isaac Sim am wichtigsten: 'yr' und 'ax'
        """
        row = [speed, yr, ax, ay, steering_cmd, current_motor]
        self._buf.append(row)

        x = torch.tensor(list(self._buf), dtype=torch.float32)  # (HISTORY, 6)
        x = x.unsqueeze(0).to(self.device)                      # (1, HISTORY, 6)

        with torch.no_grad():
            out = self.model(x).squeeze(0).cpu().numpy()        # (4,)

        return {
            'speed': float(out[0]),
            'yr': float(out[1]),   # ← Gierrate ψ̇ [rad/s]
            'ax': float(out[2]),   # ← Längsbeschleunigung aₓ [m/s²]
            'ay': float(out[3]),
        }


# ── MoCap-Zustand-Variante ("Run30"-Familie, 2026-08-13) ─────────────────────

class GRUMoCapInference:
    """
    Zweite GRU-Inferenz, echte MoCap-Zustands-Architektur (7in/5out) --
    NICHT dieselbe wie GRUInference oben (das ist die onboard-only 6in/4out
    Architektur, die tatsächlich in gru_run30_best.pth steckt).

    Checkpoint: local_run___20260731204259241445.pth -- vom Nutzer als
    "Run30" identifiziert. Verifiziert (config direkt aus dem Checkpoint
    gelesen, wie bei GRUInference oben): input_size=7, output_size=5,
    RNN_jomoh_1, history=10 -- strukturell die Run30-KLASSE (echte MoCap-
    Architektur), aber die exakten Spalten (IMU_linear_acceleration_x statt
    a_x_local, a_y_corr statt a_y_local) matchen keine einzelne Zeile in
    TECHNICAL_NOTES.md's Experiment-Tabelle 1:1 -- eher eine Mischung aus den dort
    dokumentierten Run35 (IMU ax) und Run36 (a_y_corr) Ideen. Der
    Datei-Zeitstempel (20260731204259) stimmt außerdem NICHT mit dem in
    TECHNICAL_NOTES.md für gru_run30_best.pth dokumentierten Original
    (local_run___20260731211812374678.pth) überein -- vermutlich ein
    Geschwister-Run aus den ~32 Kandidaten, nicht bitgenau "Run30" selbst,
    aber die richtige ARCHITEKTUR-FAMILIE (echter MoCap-Zustand statt
    Onboard-Sensorik).

    config['normed']==False hier ebenfalls (verifiziert) -- training_v2.py's
    `if normed:`-Zweig, der RNN_Seq_Dataset_norm/evaluate_model_denormalized
    anwenden würde, lief beim Training nie; data_mean/data_std im Checkpoint
    sind rein deskriptiv. Rohwerte ohne Normalisierung sind hier korrekt.

    Inputs (7 Kanäle): v_x_local, v_y_local, mocap_angular_velocity,
      IMU_linear_acceleration_x, a_y_corr, speed_cmd, steering_cmd
    Outputs (5 Kanäle): v_x_local, v_y_local, mocap_angular_velocity,
      IMU_linear_acceleration_x, a_y_corr
      Index 2 = ψ̇ (Gierrate, MoCap) [rad/s]
      Index 3 = aₓ (IMU-direkt)      [m/s²]

    Alle Spalten existieren im Day1-Test-CSV (verifiziert vor Einbau, siehe
    Fortschrittsbericht) -- Voraussetzung, die der Nutzer explizit als
    Go/No-Go-Kriterium vorgegeben hat.
    """

    INPUT_COLS  = ['v_x_local', 'v_y_local', 'mocap_angular_velocity',
                   'IMU_linear_acceleration_x', 'a_y_corr', 'speed_cmd', 'steering_cmd']
    OUTPUT_COLS = ['v_x_local', 'v_y_local', 'mocap_angular_velocity',
                   'IMU_linear_acceleration_x', 'a_y_corr']
    HISTORY = 10

    def __init__(self, checkpoint: str = None, device: str = 'cpu'):
        if checkpoint is None:
            checkpoint = str(_SCRIPT_DIR / 'local_run___20260731204259241445.pth')

        self.device = torch.device(device)
        self.model  = RNN_jomoh_1(input_size=len(self.INPUT_COLS),
                                   output_size=len(self.OUTPUT_COLS)).to(self.device)

        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[GRUMoCapInference] Modell geladen: {checkpoint}")
        self.reset()

    def reset(self):
        zero = [0.0] * len(self.INPUT_COLS)
        self._buf = deque([zero[:] for _ in range(self.HISTORY)],
                          maxlen=self.HISTORY)

    def step(self, vx: float, vy: float, yr: float, ax: float, ay: float,
             speed_cmd: float, steering_cmd: float) -> dict:
        """
        Args:
            vx, vy        : v_x_local, v_y_local (MoCap) [m/s]
            yr            : mocap_angular_velocity [rad/s]
            ax            : IMU_linear_acceleration_x [m/s²]
            ay            : a_y_corr [m/s²]
            speed_cmd     : Geschwindigkeitsbefehl [m/s]
            steering_cmd  : Lenkwinkelbefehl [rad]

        Returns dict mit 'vx','vy','yr','ax','ay' (vorhergesagte nächste Werte)
        """
        row = [vx, vy, yr, ax, ay, speed_cmd, steering_cmd]
        self._buf.append(row)

        x = torch.tensor(list(self._buf), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x).squeeze(0).cpu().numpy()

        return {
            'vx': float(out[0]), 'vy': float(out[1]),
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
    print(f"{'Schritt':>6}  {'yr_pred':>9}  {'ax_pred':>9}  {'speed_pred':>10}")
    for i in range(10):
        out = gru.step(
            speed=2.0 + i * 0.05,
            yr=0.1 * (i % 3),
            ax=0.2,
            ay=-0.1,
            steering_cmd=0.05 * (i % 4),
            current_motor=50.0,
        )
        print(f"  {i:4d}   {out['yr']:+9.4f}  {out['ax']:+9.4f}  {out['speed']:+10.4f}")

    print("\n✓ GRUInference smoke test OK")
    print(f"  Für Isaac Sim: out['yr'] = ψ̇ [rad/s], out['ax'] = aₓ [m/s²]")

    mocap_ckpt = _SCRIPT_DIR / 'local_run___20260731204259241445.pth'
    if mocap_ckpt.exists():
        gm = GRUMoCapInference(str(mocap_ckpt))
        gm.reset()
        print("\nGRUMoCapInference Smoke Test — 10 Schritte:")
        for i in range(10):
            out = gm.step(vx=2.0 + i*0.05, vy=0.0, yr=0.1*(i % 3), ax=0.2, ay=-0.1,
                          speed_cmd=2.0, steering_cmd=0.05*(i % 4))
            print(f"  {i:4d}   yr={out['yr']:+9.4f}  ax={out['ax']:+9.4f}  vx={out['vx']:+9.4f}")
        print("✓ GRUMoCapInference smoke test OK")
    else:
        print(f"\n[info] MoCap-Checkpoint nicht gefunden ({mocap_ckpt}), übersprungen")
