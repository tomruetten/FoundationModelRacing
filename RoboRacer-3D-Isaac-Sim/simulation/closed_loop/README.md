================================================================
  Isaac Sim V2 — Closed-Loop Control Demo
  F1TENTH Car02 | Supervisor: Felix Jahncke (TUM)
  Erstellt: 2026-08-14 | Aktualisiert: 2026-08-14
================================================================

V1 war ein Replay (aufgezeichnete Steuerbefehle → Modell → Vergleich).
V2 ist ein echter Regelkreis: Pure Pursuit erzeugt Steuerbefehle,
GRU-Modell sagt den Fahrzeugzustand vorher, State Integrator berechnet
die Position → das Fahrzeug "fährt wirklich" im Simulator.

================================================================
  HINTERGRUND: GRU CHECKPOINT (Stand V1 → V2, historisch)
================================================================

Das in V1 ursprünglich eingesetzte gru_run30_best.pth war RUN37
(6 Eingaben/4 Ausgaben, nur IMU-Spalten) — NICHT Run30. V2 benötigt
zwingend 7 Eingaben/5 Ausgaben. Dieser Austausch ist in diesem Repo-
Snapshot BEREITS erledigt — der Checkpoint unter
dynamics/deployed/trained_models/gru_run30_best.pth ist schon die
korrekte Run30-Version. Kein manueller Austausch mehr nötig, die
folgenden Infos sind nur zur Einordnung/Verifikation:

  # Optional zur Verifikation:
  python3 -c "
  import torch
  ckpt = torch.load('../../dynamics/deployed/trained_models/gru_run30_best.pth',
                    map_location='cpu', weights_only=False)
  cfg = ckpt['config']
  print('input_size:', cfg['input_size'])          # MUSS 7 sein
  print('input_col_4:', cfg['input_columns'][3])   # MUSS a_x_local sein
  print('val_loss:', cfg['best_val_loss'])          # MUSS ~0.17-0.20 sein
  "

  Checkpoint im Paket (dynamics/deployed/trained_models/gru_run30_best.pth):
    Datei:    local_run___20260730204404090131.pth  (Run30 Retrain)
    Spalten:  a_x_local, a_y_local  ← korrekte MoCap-Spalten
    val_loss: 0.2000
    3ch CL-RMSE: 0.4626 (bestätigt im Bericht vom 2026-07-31)

  Alternative (noch niedrigerer val_loss, NICHT in diesem Repo enthalten,
  auf Anfrage verfügbar):
    gru_run30_jul29_val0173.pth  (local_run___20260729104728844540)
    val_loss: 0.1733

================================================================
  VORAUSSETZUNGEN
================================================================

Conda-Umgebung: f1tenth_dynamics (nur torch + numpy + pandas + matplotlib)
  conda activate f1tenth_dynamics
  [Stand 2026-09-07: auf der aktuellen Lab-Maschine heißt die äquivalente
  Umgebung "roboracer" (torch 2.6.0+cu124) — falls f1tenth_dynamics dort
  nicht existiert, "roboracer" verwenden]

Für die Visualisierung in Isaac Sim selbst (statt reiner numpy/torch-
Simulation): Isaac Sim 6.0.1-rc.7, conda env "env_isaac6"
(Python 3.12.13) — siehe simulation/isaac_sim/README.md.

GRU-Checkpoint (bereits korrekt im Paket, siehe oben):
  dynamics/deployed/trained_models/gru_run30_best.pth

Referenz-CSV (Day3, Trainingsdaten — NICHT Day1!):
  ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv
  (NICHT in diesem Repo enthalten — Rohdaten liegen auf der Lab-Maschine,
  auf Anfrage verfügbar, siehe Haupt-README.md. Alle --csv/--ref-csv unten
  sind Pflichtparameter, also durch den eigenen tatsächlichen Pfad ersetzen.)

================================================================
  SCHRITT 1 — Referenzpfad extrahieren (einmalig)
================================================================

python3 09_v2_reference_extract.py \
  --csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv \
  --output reference_path_day3.csv

→ Erzeugt reference_path_day3.csv  (Wegpunkte x, y, θ)
→ Erzeugt reference_path_day3_plot.png  (Draufsicht + Speed-Plot)

================================================================
  SCHRITT 2a — Nur GRU (einfachste Variante)
================================================================

TERMINAL 1:
  conda activate f1tenth_dynamics
  python3 10_v2_main_demo.py \
    --gru ../../dynamics/deployed/trained_models/gru_run30_best.pth \
    --ref-csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv \
    --duration 60 \
    --target-speed 1.5

→ Erzeugt v2_output/v2_trajectory.csv
→ Erzeugt v2_output/v2_trajectory.png

================================================================
  SCHRITT 2b — GRU + MS-NN ψ̇-Vergleich (empfohlen für Thesis)
================================================================

TERMINAL 1 (zuerst starten, nnodely-env):
  conda run -n nnodely-env python3 05_msnn_dynamics_server.py \
    --model ../../dynamics/deployed/trained_models/msnn_exp9_phase2_backup \
    --queue-dir /tmp/isaac_msnn_v2

TERMINAL 2 (nachdem Server "Warte auf Anfragen" meldet):
  conda activate f1tenth_dynamics
  python3 10_v2_main_demo.py \
    --gru ../../dynamics/deployed/trained_models/gru_run30_best.pth \
    --ref-csv ../ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv \
    --msnn-queue /tmp/isaac_msnn_v2 \
    --duration 60

→ Zusätzlich: ψ̇ Vergleich GRU vs MS-NN im Plot

================================================================
  PARAMETER-TUNING (Pure Pursuit)
================================================================

--lookahead    [m]    Vorausschau-Distanz (default: 0.5)
                      Kleiner → engere Kurven, mehr Überschwingen
                      Größer  → weicheres Verhalten, abgeschnittene Kurven

--target-speed [m/s]  Sollgeschwindigkeit (default: 1.5)
                      0.5-1.0: konservativ (GRU stabiler)
                      1.5-2.0: realistisch für Racing

Beim ersten Durchlauf: target-speed=1.0, lookahead=0.5 empfohlen.
Dann schrittweise erhöhen.

Alpha-Sweep (Hybrid Plant GRU+MS-NN):
  --alpha 0.0   Nur GRU (Baseline)
  --alpha 0.3   Empfehlung Start
  --alpha 0.5   Gleichgewicht
  --alpha 0.7   MS-NN-dominant
  --alpha 1.0   Nur MS-NN ψ̇ (MS-NN Warmup beachten: erste 500ms α=0 erzwungen)

================================================================
  AUSGABE-METRIKEN
================================================================

v2_trajectory.png enthält:
  [0,0] Draufsicht: Referenzpfad (gestrichelt) vs GRU-Pfad (blau)
  [0,1] Tracking-Fehler [m] über Zeit
  [1,0] Geschwindigkeit: speed_cmd vs GRU-Vorhersage
  [1,1] Gierrate: GRU (+ MS-NN falls aktiv)
  [2,0] ψ̇ Differenz GRU−MS-NN (falls MS-NN aktiv)
  [2,1] Histogramm der |Δψ̇| (falls MS-NN aktiv)

Console-Output am Ende:
  Duration, Path length, Tracking error (mean/P95/max),
  GRU speed/yr stats, GRU−MS-NN ψ̇ RMSE

================================================================
  CHECKPOINT-HINTERGRUND (Recherche 2026-08-14)
================================================================

Alle local_run_*.pth bis Juli 30 verwenden korrekte Run30-Spalten
(a_x_local, a_y_local). Nur die Juli-31-Modelle wechselten auf
IMU-Spalten (Run31-35 Experimente) — das sind die falschen.

Die "0.4448" CL-RMSE aus dem Juli-29-Bericht wurde mit einer
Sliding-Window-Evaluation (10 Schritte Reset) gemessen, nicht
mit dem vollen 125s-Rollout. Beides sind gültige Metriken,
aber nicht direkt vergleichbar.

================================================================
  WEITERE SKRIPTE (nicht Teil des Standard-Ablaufs oben)
================================================================

11_full_rollout_cl_eval.py
  Standard-Protokoll zur GRU-Checkpoint-Bewertung per voller autoregressiver
  Rollout-Simulation (nicht windowed-CL). Warmup 10 Schritte mit
  Ground-Truth, danach volle Autoregression. Metriken: 3ch/5ch RMSE.
    conda activate f1tenth_dynamics
    python3 11_full_rollout_cl_eval.py --checkpoint <pfad-zu>/gru_run30_best.pth \
      --csv <pfad-zu>/ma_jonathanmohr-main/processed_data/session_20260728_163324_seg1_20ms.csv
  (--csv-dir für mehrere CSVs gewichtet gemittelt, --device cuda optional)

12_v2_msnn_plant_demo.py  (EXPERIMENTELL)
  Variante von 10_v2_main_demo.py, bei der MS-NN (statt GRU) den
  Regelkreis antreibt. MS-NN liefert nur (yr_pred, ax_pred), kein vx/vy —
  vx wird lokal integriert, vy wird als 0 angenommen (kinematische
  Bicycle-Näherung). Siehe Skript-Docstring für Details/Caveats.

16_raceline_retest_correct_checkpoint.py, 17_raceline_video_config_D.py
  Task-5-Raceline-Vergleich (B/C/D Speed-Envelopes) mit dem korrekten
  Run30-Checkpoint, aufbauend auf PurePursuit/GRUInference/StateIntegrator/
  run_v2 aus 10_v2_main_demo.py (via importlib geladen). 16_ druckt nur
  Zahlen und schreibt output/*.json; 17_ rendert zusätzlich ein mp4
  (Top-Down-Replay + Tracking-Error/Speed-Telemetrie) nach output/videos/.
  Referenz-Raceline: reference/raceline_ftm_halle/ (siehe dortiges
  README.md).
    python3 16_raceline_retest_correct_checkpoint.py
    python3 17_raceline_video_config_D.py   # braucht ffmpeg im PATH

================================================================
  WICHTIGER HINWEIS: DATEN-CONSTRAINT (Felix)
================================================================

  --ref-csv MUSS Trainingsdaten oder bekannte Daten sein.
  session_20260721_20ms.csv (Day1) = TEST ONLY → NICHT als Referenzpfad!

  Empfohlene Referenz-CSV:
    session_20260728_163324_seg1_20ms.csv  (Day3, Trainingsdaten)

================================================================
