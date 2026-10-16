# MS-NN Fahrzeugdynamik — Technisches Referenzdokument
**F1TENTH Car02 · Supervisor: Felix Jahncke (TUM) · Stand: 2026-08-13**

Dieses TECHNICAL_NOTES.md ist für die Weiterarbeit am Labor-PC gedacht.
Es beschreibt den vollständigen technischen Stand des MS-NN-Projekts.

---

## Projekt-Kontext

Ziel: Fahrzeugdynamik-Modell für F1TENTH Car02, das ψ̇ (Gierrate) und
aₓ (Längsbeschleunigung) vorhersagt — für Isaac Sim Integration.

Nächster Schritt (Felix-Direktive, 2026-08-13):
> "Integration der Dynamics in Isaac und dann machen wir mal eine V1,
>  schauen was alles schon funktioniert und was wir noch brauchen."

---

## Ergebnisse (abgeschlossen)

| Experiment | Architektur | Phase | ψ̇ RMSE | Ratio |
|------------|-------------|-------|---------|-------|
| **exp9 ★★ BEST** | withax 3D | P1+2 only | **0.1158** | **1.05×p** |
| exp8 | noax 2D | P1+2+3 | 0.1191 | 1.08×p |
| exp6c (Paper) | withax 3D | P1+2+3 | 0.1196 | 1.09×p |
| exp10 (3 Sessions) | noax 2D | P1+2+3 | 0.1232 | 1.12×p |
| Paper-Referenz | noax | P1+2+3 | — | 1.21×p |
| exp11 (Mirror) | noax 2D | P1+2+3 | 0.3060 | 2.79×p ❌ |

**Persistence-Baseline** (Day3 163324 Val): ψ̇ = 0.1098 rad/s · aₓ = 0.1487 m/s²

**OL vs CL** (exp6c):
- ψ̇: immer OL (laterales Modell hat kein accy-Feedback) → CL ändert ψ̇ nicht
- aₓ Day3 Val: OL=0.3646 → CL=0.3398 (−6.8%) ← CL hilft aₓ
- aₓ Day1 Test: OL=1.1845 → CL=1.1898 (+0.4%) ← cross-day: kaum Effekt

---

## Architektur (withax 3D-LocalModel)

### Lateral (ψ̇) — 20 Knoten

```
Fuzzy-Dimensionen: 5δ × 2vx × 2ax = 20 Knoten
Knotenformel: vx × (k1 + k2×(δ−δ₀) + k3×(vx−v₀) + k4×(ax−a₀))
```

- `nsp = 20` Zeitschritte Lookback (Sliding Window)
- `nd=5, nv=2, nax=2` Fuzzy-Kanäle
- Fuzzify: Triangular membership functions
- Ausgabe: `yr_out = LocalModel(FIR)(...)`
- Zusatzausgabe: `accy_computed = yr_out × vel.last()` (für Phase3 CL)

### Longitudinal (aₓ) — Physikalisches Modell

```
aₓ = (1/m) × (2×(T/r) − Kd×v² − Cv×v − Fy×sin(δ)) / (1 + 2Iw/(m×r²)) − Cr×g
```

- `cw = 15` Zeitschritte Lookback
- Tpos/Tneg: FIR-gefilterte Drehmomente (positive/negative Motorströme getrennt)
- Fy(accy): LocalModel (4 Kanäle accy × 2 Kanäle vx) für Querkraft
- Lernbare Parameter: r1, mass, Kd, Cv, Cr, Iw1 + Fy-LocalModel-Gewichte

### Phase3 Closed-Loop (NUR für noax empfohlen)

```python
dyn.addConnect(accy_computed, accy)
dyn.addClosedLoop(acc_out, accy)
```

**WICHTIG für withax:** Phase3 NICHT ausführen!
- withax hat 80 lat. Parameter (vs noax: 30)
- CL-Feedback destabilisiert Gradienten → +4.27% schlechter
- exp9 Deploy: Phase2-Backup verwenden

---

## Daten-Setup

### Kritische Constraint (Felix, nie verletzen)
```
TEST ONLY:   processed_data/session_20260721_20ms.csv      # Day1
AUSSCHLUSS:  processed_data/session_20260728_124427_seg1_20ms.csv  # 100% NaN IMU
```

### exp9 Datensplit (BESTES Modell)
```
Train: session_20260728_163324_seg1_20ms.csv  erste 4333 Zeilen (2/3 von 6500)
     + session_20260722_113922_seg1_20ms.csv  ab Zeile 1000 (skip Warmup)
     = 8453 Trainingszeilen

Val:   session_20260728_163324_seg1_20ms.csv  letzte 2167 Zeilen (t=87–130s)
       → Racing-Section, höhere Dynamik, repräsentativer Val-Fehler
```

### Spalten-Mapping
```python
'yaw_rate' ← IMU_angular_velocity_z     [rad/s]
'vel'      ← odom_speed                 [m/s]
'delta'    ← steering_cmd               [rad]
'acc'      ← IMU_linear_acceleration_x  [m/s²]
'accy'     ← IMU_linear_acceleration_y  [m/s²]
'crt'      ← current_motor              [A]
```

---

## nnodely-Besonderheiten (wichtig für Debugging)

### 1. NeuObj global name registry
```python
# Problem: NeuObj.names ist class-level, wird zwischen build_model()-Aufrufen geteilt
# Lösung: am Anfang jeder build_model()-Funktion:
from nnodely.basic.relation import NeuObj
NeuObj.names.clear()
NeuObj.names.append('SampleTime')
```

### 2. seed=42 überschreibt ALLE Parameter
```python
dyn = nnodely(seed=42, ...)  # Überschreibt alle values=[[...]] mit deterministischem Wert
# → Reproduzierbarkeit garantiert, aber initiale Werte im Code sind nur Hinweise
```

### 3. Dataset-Namen müssen prozess-global eindeutig sein
```python
# Fehler: NameError: name 'eval_set' already used as Dataset
# Lösung:
_ctr = [0]
def _dsname(prefix):
    _ctr[0] += 1
    return f'{prefix}_{_ctr[0]}'
```

### 4. withax eval: acc.sw(nsp) gibt Fenster zurück
```python
# FALSCH: acc_true = samples['acc']  # gibt Fenster, nicht Skalar
# RICHTIG:
acc_true = ds_val['acc'][nsp:nsp+n]   # direkt aus Datensatz
yr_true  = ds_val['yaw_rate'][nsp:nsp+n]
```

### 5. CL accy length mismatch
```python
# Problem: cl_accy Länge = n-5, concat mit 20 Warmup = n-25+20 < n_total
# Lösung:
pad = len(ds['accy']) - len(cl_accy)
if pad > 0:
    cl_accy = np.concatenate([cl_accy, ds['accy'][len(cl_accy):]])
```

---

## Dateien in diesem Ordner

| Datei | Beschreibung |
|-------|-------------|
| `12_train_msnn_exp9_best.py` | ★★ BESTES Modell: withax Phase1+2, yr=0.1158 |
| `11_train_msnn_full.py` | Paper-Protokoll: Phase1+2+3, yr=0.1196 |
| `13_inference_msnn.py` | Isaac Sim API: `MSNNInference.step()` |
| `README.txt` | Schritt-für-Schritt Anleitung |
| `TECHNICAL_NOTES.md` | dieses Dokument |

---

## Trainierte Modelle (Pfade auf Lab-PC)

```
ma_jonathanmohr-main/trained_models/
  msnn_exp9_phase2_backup/net.json   ← DEPLOY (Isaac Sim)
  msnn_exp9_withax_best/net.json     ← Vollständiger Checkpoint
  msnn_exp6c_combined/net.json       ← Paper-Protokoll Referenz
  msnn_exp8_combined_lowlr/net.json  ← noax Vergleich
  msnn_exp10_three_sessions/net.json ← 3-Session Vergleich
```

---

## Isaac Sim Integration (nächste Schritte)

Felix-Direktive: "Integration der Dynamics in Isaac Sim, dann V1"

### Option A: nnodely direkt in Isaac Sim Python
```python
# Falls nnodely in Isaac Sim Python installierbar:
pip install nnodely==1.5.4  # CPU-only, kein CUDA-Konflikt
from inference_msnn import MSNNInference
model = MSNNInference(model_dir='trained_models/msnn_exp9_phase2_backup')
```

### Option B: Separater Prozess (sicher, kein env-Konflikt)
```
Terminal 1 (nnodely-env):
  python3 05_msnn_dynamics_server.py \
    --model trained_models/msnn_exp9_phase2_backup \
    --queue-dir /tmp/isaac_dyn_queue

Terminal 2 (Isaac Sim Python):
  # Schreibt req_NNNNNN.json → liest res_NNNNNN.json
  # IPC-Format: {"step": N, "delta": ..., "vel": ..., "acc": ..., "accy": ..., "crt": ...}
```

### V1-Ziel
1. Isaac Sim replay einer echten Trajektorie (CSV-Datei)
2. Für jeden Schritt: Control-Inputs → MS-NN → Vorhersage
3. Vergleich: Isaac Sim Physik vs MS-NN vs Ground Truth
4. Output: `dynamics_comparison.csv` + Plot

---

## Conda-Umgebung einrichten (falls nnodely-env fehlt)

```bash
conda create -n nnodely-env python=3.10
conda activate nnodely-env
pip install nnodely==1.5.4
pip install numpy pandas torch scipy matplotlib
```

Test:
```bash
conda run -n nnodely-env python3 13_inference_msnn.py
# Erwartet: "inference_msnn.py smoke test OK"
```

---

---

## GRU Run30 (Alternative für Isaac Sim)

| Eigenschaft | MS-NN exp9 | GRU Run30 |
|-------------|------------|-----------|
| Same-Day ψ̇ RMSE | **0.1158** (1.05×p) | ~0.15–0.18 |
| Cross-Day ψ̇ RMSE | 0.1992 (2.26×p) | **0.4448** (cross-day CL) |
| Interpretierbar | Ja (Physikstruktur) | Nein (Black Box) |
| Isaac Sim Eignung | Bekannte Strecke | Neue Strecken/Tage |

**Wann welches Modell?**
- MS-NN wenn: Strecke bekannt, same-day, physikalische Erklärbarkeit nötig
- GRU wenn: neue Strecke/neuer Tag, cross-day Robustheit wichtig

### GRU Architektur (RNN_jomoh_1)
```python
GRU(input=7, hidden=100)
→ FC(100,100) + LeakyReLU(0.1)
→ FC(100,100) + LeakyReLU(0.1)
→ FC(100,5)

Inputs  (7): [vx, vy, yr, ax, ay, speed_cmd, steering_cmd]
Outputs (5): [vx, vy, yr, ax, ay]
History: 10 Zeitschritte (200ms Lookback)
```

### GRU Checkpoint
```
trained_models/gru_run30_best.pth   ← 212 KB, Jul 31 2026 21:30
Original: local_run___20260731211812374678.pth
Training: 200ep / patience=30 / lr=0.003 / AdamW
```

### GRU Inference (Isaac Sim)
```python
from gru_inference import GRUInference  # = 15_gru_inference.py

gru = GRUInference(checkpoint='trained_models/gru_run30_best.pth')
gru.reset()   # vor jeder Episode!

out = gru.step(
    vx=2.0, vy=0.0, yr=0.1,
    ax=0.2, ay=-0.1,
    speed_cmd=2.0, steering_cmd=0.05
)
yr_pred = out['yr']   # [rad/s]
ax_pred = out['ax']   # [m/s²]
```

### Mirror-Augmentierung (NUR GRU)
```
Training mit gespiegelten Daten: δ→−δ, vy→−vy, yr→−yr
→ GRU profitiert (keine strukturierten Knoten)
→ MS-NN NICHT! (LocalModel Fuzzy-Center werden gestört)
```

---

## Evaluierung & Plots

```bash
# Beide Modelle evaluieren + Plots erstellen:
python3 14_eval_and_plot.py --data-dir /Pfad/zu/processed_data/

# Nur MS-NN:
python3 14_eval_and_plot.py --model msnn --data-dir ...

# Nur GRU:
python3 14_eval_and_plot.py --model gru --data-dir ...
```

Ausgabe in `eval_output/`:
- `results_msnn_vs_gru.png` — Balkendiagramm RMSE-Vergleich
- `plot_yawrate_day3.png`   — ψ̇ Zeitverlauf Day3 Val
- `plot_yawrate_day1.png`   — ψ̇ Zeitverlauf Day1 Test (cross-day)
- `results_comparison.csv`  — Zahlentabelle

---

## Offene Fragen / Future Work

- [ ] Isaac Sim V1: MS-NN + GRU parallel → vergleichen welches besser passt
- [ ] nnodely in Isaac Sim Python installierbar? (CPU-only, kein CUDA-Konflikt)
- [ ] Cross-Day Domain Adaptation: yr 1.05×p (same-day) vs 2.26×p (cross-day)
- [ ] withax Phase3 stabilisieren: anderer lr-Schedule oder Gradient-Clipping?
- [ ] Online-Adaptation: Modell während Fahrt auf neuen Tag fine-tunen?
