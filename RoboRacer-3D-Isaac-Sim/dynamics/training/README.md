================================================================
dynamics/training — Anleitung zur Reproduktion
Ziel: GRU (0.4448) · Deep Dynamics (0.5675) · MS-NN exp9 (yr=0.1158)
================================================================

KRITISCHE DATEN-CONSTRAINT — NIE VERLETZEN
───────────────────────────────────────────
  TEST ONLY:   processed_data/session_20260721_20ms.csv    ← Day1, nur finale Eval
  TRAIN/VAL:   session_20260722_*.csv  (Day2)
               session_20260728_*.csv  (Day3)
  AUSSCHLUSS:  session_20260728_124427_seg1_20ms.csv       ← 100% NaN IMU

VORAUSSETZUNGEN
───────────────
  Verzeichnis:  $DYNAMICS_WORKDIR/ — Pfad zu Felix/Jonathans Original-
                Codebase (ma_jonathanmohr-main/, siehe dynamics/msnn/
                src/), nicht in diesem Repo enthalten.
  Skripte hier laufen aus diesem Verzeichnis (WORK-Pfad im Code).

  Persistence-Baseline (Day3 163324, Val):
    yr-persistence  = 0.1098 rad/s
    ax-persistence  = 0.1487 m/s²

================================================================
MODELL A — GRU Run30 (Cross-Day 3ch-CL-RMSE = 0.4448)
================================================================

  CONDA-UMGEBUNG: f1tenth_dynamics

  cd $DYNAMICS_WORKDIR/
  conda activate f1tenth_dynamics
  python3 [Pfad/zu/Zusammenfassung/]dynamics/training/06_train_gru_run30_best.py

  ARCHITEKTUR:
    GRU(hidden=100) + FC×3 Schichten
    Eingabe (7 Kanäle): v_x_local, v_y_local, mocap_angular_velocity,
                         a_x_local, a_y_local, speed_cmd, steering_cmd
    Ausgabe (5 Kanäle): v_x_local, v_y_local, mocap_angular_velocity,
                         a_x_local, a_y_local

  HYPERPARAMETER:
    lr=0.003, batch_size=700, num_epochs=200, patience=30
    history=10, seq_length=500, stride=5

  DATEN (40 Dateien total):
    Train:  Day2 (alle Segmente: 113922, 115055, 120608) + Mirror-Kopien
            Day3 (alle Segmente außer 124427) + Mirror-Kopien
    Val:    session_20260722_115055_seg2_20ms.csv (KEINE Mirror)
    Test:   session_20260721_20ms.csv (Day1 — nur Evaluation am Ende)

  HINWEIS: Mirror-Augmentierung = δ→−δ, v_y→−v_y, ψ̇→−ψ̇.
           Funktioniert für GRU (kein strukturiertes Modell).
           NICHT für MS-NN verwenden (Gradientenkonflikte)!

  LAUFZEIT: ~15-20 min (200 Epochen, GPU)
  CHECKPOINT: trained_models/gru_run30_best.pth
  ERGEBNIS: 3ch-CL-RMSE Day1 = 0.4448 rad/s

================================================================
MODELL B — Deep Dynamics v8 (Cross-Day 3ch-CL-RMSE = 0.5675)
================================================================

  CONDA-UMGEBUNG: f1tenth_dynamics

  python3 [Pfad/zu/Zusammenfassung/]dynamics/training/10_train_deep_dynamics_v8_best.py

  ARCHITEKTUR:
    Pacejka-Reifenmodell mit 6 lernbaren Parametern
    K=10 Rollout-Schritte für Closed-Loop Training

  CHECKPOINT: trained_models/deep_dynamics_v8_best.pth
  ERGEBNIS: 3ch-CL-RMSE Day1 = 0.5675 rad/s

  HINWEIS: DD schlechter als GRU auf Day1 (cross-day), weil Pacejka-
           Nichtlinearität nicht gut auf Day1-Fahrverhalten generalisiert.

================================================================
MODELL C1 — MS-NN exp6c (Paper-Protokoll Phase1+2+3, yr=0.1196)
================================================================

  CONDA-UMGEBUNG: nnodely-env  (nnodely==1.5.4, CPU-only!)

  cd $DYNAMICS_WORKDIR/
  conda run -n nnodely-env python3 \
    [Pfad/zu/Zusammenfassung/]dynamics/training/11_train_msnn_full.py

  ARCHITEKTUR: withax 3D-LocalModel
    Lateral ψ̇: 5δ × 2vx × 2ax = 20 Knoten, je 4 Parameter
               vx × (k1 + k2×(δ−δ₀) + k3×(vx−v₀) + k4×(ax−a₀))
    Longitudinal aₓ: Physikalisches Gleichgewicht + Fy(accy)-Feedback

  TRAINING (vollständiges Drei-Phasen-Protokoll wie im Paper):
    Phase1: Lateral ψ̇   — 1500ep, patience=300, lr=1e-3
    Phase2: Longitudinal — 3000ep, patience=500, lr=1e-3
    Phase3: CL-Feedback  — 1000ep, patience=200, lr=1e-4
            (addConnect + addClosedLoop für aₓ)

  DATEN:
    Train: Day2 alle Segmente (session_20260722_*.csv)
    Val:   Day3 alle Segmente (session_20260728_*.csv)
    Test:  Day1 (session_20260721_20ms.csv)

  LAUFZEIT: ~50-60 min auf CPU
  CHECKPOINT: trained_models/msnn_exp6c_combined/net.json
  ERGEBNIS: ψ̇ RMSE = 0.1196 rad/s (1.09×persistence)
            aₓ Phase3 CL = 0.3398 m/s² (−6.8% durch CL — CL hilft aₓ!)

================================================================
MODELL C2 — MS-NN exp9 withax BEST (Phase1+2 only, yr=0.1158) ★★
================================================================

  CONDA-UMGEBUNG: nnodely-env  (nnodely==1.5.4, CPU-only!)

  cd $DYNAMICS_WORKDIR/
  conda run -n nnodely-env python3 \
    [Pfad/zu/Zusammenfassung/]dynamics/training/12_train_msnn_exp9_best.py

  ARCHITEKTUR: withax 3D-LocalModel (identisch mit exp6c)

  TRAINING:
    Phase1: Lateral ψ̇   — 1500ep, patience=300, lr=1e-3
    Phase2: Longitudinal — 3000ep, patience=500, lr=1e-3
    Phase3: NICHT ausführen! (destabilisiert withax: +4.27% schlechter)

  DATEN (kombiniertere Aufteilung als exp6c):
    Train: Day3 163324 erste 2/3 (4333 Zeilen)
           + Day2 113922 skip erste 1000 Zeilen (4120 Zeilen)
           = 8453 Trainingszeilen gesamt
    Val:   Day3 163324 letzte 1/3 (2167 Zeilen, t=87–130s, Racing-Section)
    Test:  Day1 (session_20260721_20ms.csv)

  LAUFZEIT: ~35 min auf CPU (nur Phase1+2, kein Phase3)
  CHECKPOINT: trained_models/msnn_exp9_phase2_backup/net.json  ← DEPLOY DIESEN!
  ERGEBNIS: ψ̇ RMSE = 0.1158 rad/s (1.05×persistence) — schlägt Paper (1.21×p)!

  WARUM KEIN PHASE3?
    withax hat 80 lat. Parameter (vs noax: 30). CL-Feedback mit so vielen
    Parametern führt zu Gradientenkonflikten → Phase3 verschlechtert um +4.27%.
    Für noax (exp6c/exp8): Phase3 hilft oder ist neutral (−0.3% bis +0.1%).

================================================================
EVALUIERUNG (MS-NN)
================================================================

  12_eval_msnn.py ist ein ÄLTERES Skript aus einer früheren Modell-
  Iteration (evaluiert trained_models/msnn_full_car02/, NICHT das
  deployte exp9_phase2_backup). Kein argparse, keine CLI-Argumente:

    cd $DYNAMICS_WORKDIR/
    conda run -n nnodely-env python3 \
      [Pfad/zu/Zusammenfassung/]dynamics/training/12_eval_msnn.py

  Erwartet: $DYNAMICS_WORKDIR/processed_data/session_20260721_20ms.csv
  und $DYNAMICS_WORKDIR/trained_models/msnn_full_car02/ (WORK-Pfad im
  Code, siehe VORAUSSETZUNGEN). Gibt Ω/aₓ RMSE auf Day1 aus, speichert
  JSON+Plot nach /tmp/eval_msnn_results.json bzw. /tmp/eval_msnn_plots.png.

  FÜR DAS AKTUELL DEPLOYTE MODELL (msnn_exp9_phase2_backup) STATTDESSEN:
    dynamics/msnn/14_eval_and_plot.py --data-dir /Pfad/zu/processed_data/
  (evaluiert MS-NN exp9 UND GRU Run30 zusammen, siehe dynamics/msnn/README.md
  SCHRITT 5 — das ist der methodisch aktuelle Vergleichsweg, gibt genau
  die Day3-OL/CL- und Day1-OL-Zahlen aus der ERGEBNIS-ÜBERSICHT unten aus.)

================================================================
ISAAC SIM INFERENCE API
================================================================

  CONDA-UMGEBUNG: nnodely-env

  Modul heißt im Ordner 13_inference_msnn.py, liegt hier aber zusätzlich
  unter dem unnummerierten Namen inference_msnn.py (Python-Identifier
  dürfen nicht mit einer Ziffer beginnen — "from 13_inference_msnn import"
  wäre ein SyntaxError):

  from inference_msnn import MSNNInference

  model = MSNNInference(
      model_dir='/Pfad/zu/dynamics/deployed/trained_models/msnn_exp9_phase2_backup'
  )
  model.reset()  # Vor jeder neuen Episode aufrufen!

  for t in range(N_steps):
      yr_pred, ax_pred = model.step(
          delta  = steering_cmd,   # [rad]
          vel    = odom_speed,     # [m/s]
          acc    = accel_x,        # [m/s²] IMU_linear_acceleration_x
          accy   = accel_y,        # [m/s²] IMU_linear_acceleration_y
          crt    = current_motor   # [A] Motorstrom
      )
      # yr_pred: Gierrate [rad/s], ax_pred: Längsbeschleunigung [m/s²]

  HINWEIS: Sliding-Window-Buffer (nsp=20, cw=15) wird intern verwaltet.
           reset() MUSS am Episodenanfang aufgerufen werden.

================================================================
REIHENFOLGE DER AUSFÜHRUNG
================================================================

  1. Erst: 01_extract_bag.py          (ROS2 Bag → CSV)
  2. Dann: 02_data_preparation.py     (Feature Engineering: v_x_local etc.)
  3. Dann: 03_mirror_augmentation.py  (NUR für GRU — vor GRU-Training)
  4. GRU:  06_train_gru_run30_best.py
  5. DD:   10_train_deep_dynamics_v8_best.py
  6. MSNN: 11_train_msnn_full.py  ODER  12_train_msnn_exp9_best.py
  7. Eval: 08_eval_closed_loop.py (GRU/DD) · 12_eval_msnn.py (MS-NN)

================================================================
FEHLERBEHEBUNG
================================================================

Problem: "NameError: 'yaw_rate' already used as NeuObj"
  → nnodely global state: NeuObj.names nicht zurückgesetzt.
  → Lösung: neuen Python-Prozess starten (kein zweites Modell im selben Prozess)
  → Oder am Skript-Anfang:
      from nnodely.basic.relation import NeuObj
      NeuObj.names.clear(); NeuObj.names.append('SampleTime')

Problem: "NameError: 'eval_set' already used as Dataset"
  → Eval-Funktion mehrfach aufgerufen ohne neuen Prozess.
  → Eindeutige Dataset-Namen verwenden (Zähler-Suffix).

Problem: MS-NN RMSE > 0.20 (schlechter als Persistence)
  → Phase3 mit withax? → Phase3 weglassen (nur Phase1+2 für exp9).
  → Seed=42 prüfen: nnodely überschreibt alle Parameterwerte mit seed!
  → Daten: session_20260728_124427 ausgeschlossen? (100% NaN IMU)

Problem: GRU konvergiert nicht (Loss stagniert nach 10 Epochen)
  → lr auf 0.001 reduzieren
  → Sicherstellen, dass Mirror-CSVs vorhanden sind (03_mirror_augmentation.py laufen lassen)

================================================================
ERGEBNIS-ÜBERSICHT
================================================================

  Modell                  Eval-Modus      Datensatz   RMSE          Ratio
  ──────────────────────────────────────────────────────────────────────
  GRU Run30               CL (3ch)        Day1 Test   0.4448 rad/s  cross-day
  Deep Dynamics v8        CL (3ch)        Day1 Test   0.5675 rad/s  cross-day
  MS-NN exp6c [P1+2+3]    OL              Day3 Val    0.1196 rad/s  1.09×p
  MS-NN exp6c [P1+2+3]    CL (aₓ)        Day3 Val    0.3398 m/s²   aₓ-6.8%
  MS-NN exp9 [P1+2] ★★    OL              Day3 Val    0.1158 rad/s  1.05×p  BEST
  Paper-Referenz          OL              —            —             1.21×p
