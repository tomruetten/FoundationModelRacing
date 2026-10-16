================================================================
MS-NN Fahrzeugdynamik — Lab-Anleitung
F1TENTH Car02 | Supervisor: Felix Jahncke (TUM) | 2026-08-13
================================================================

Dieses Paket enthält alles, um die MS-NN-Ergebnisse am Labor-PC
zu reproduzieren und in Isaac Sim zu integrieren.

ERGEBNISSE (Zusammenfassung)
─────────────────────────────
  Bestes Modell   exp9 withax Phase2:  ψ̇ = 0.1158 rad/s  (1.05× persistence)
  Paper-Protokoll exp6c Phase1+2+3:   ψ̇ = 0.1196 rad/s  (1.09× persistence)
  Paper-Referenz  (Mohr 2026):               ≈ 1.21× persistence
  → Beide Modelle schlagen das Paper!

  Persistence-Baseline (Day3 163324 Val-Section):
    ψ̇ = 0.1098 rad/s  ·  aₓ = 0.1487 m/s²

================================================================
INHALT DIESES ORDNERS
================================================================

  11_train_msnn_full.py        ← MS-NN Paper-Protokoll (Phase1+2+3, yr=0.1196)
  12_train_msnn_exp9_best.py   ← MS-NN BESTES Modell (Phase1+2, yr=0.1158)
  13_inference_msnn.py         ← MS-NN Isaac Sim API
  14_eval_and_plot.py          ← Evaluierung + Plots beider Modelle
  15_gru_inference.py          ← GRU Isaac Sim API (inkl. GRUMoCapInference)
  gru_inference.py             ← identische Kopie von 15_gru_inference.py,
                                  unter diesem unnummerierten Namen von
                                  simulation/isaac_sim/06_ und 08_ per bare import
                                  gebraucht (siehe dortige sys.path.insert)
  README.md                    ← diese Datei
  TECHNICAL_NOTES.md                    ← technisches Referenzdokument

  Deployte Checkpoints (msnn_exp9_phase2_backup/net.json, gru_run30_best.pth
  und local_run___20260731204259241445.pth für GRUMoCapInference) liegen
  NICHT hier, sondern in dynamics/deployed/trained_models/ bzw. direkt in
  diesem Ordner (lokale GRUMoCap-Variante) — siehe SCHRITT 3/4 unten.

================================================================
VORAUSSETZUNGEN (Lab-PC)
================================================================

  Conda-Umgebung: nnodely-env
    pip install nnodely==1.5.4
    pip install numpy pandas torch   (CPU-only, kein CUDA nötig)

  Daten-Verzeichnis: ma_jonathanmohr-main/processed_data/
    session_20260722_113922_seg1_20ms.csv   (Day2, Train)
    session_20260722_115055_seg1_20ms.csv   (Day2, Train)
    session_20260728_163324_seg1_20ms.csv   (Day3, Train + Val)
    session_20260721_20ms.csv               (Day1, NUR Test!)

  KRITISCH — nie verletzen:
    session_20260721_20ms.csv = TEST ONLY (Day1)
    session_20260728_124427_seg1_20ms.csv = AUSSCHLUSS (100% NaN IMU)

================================================================
SCHRITT 1 — Bestes Modell neu trainieren (exp9)
================================================================

  Laufzeit: ~35 Minuten auf CPU

  cd ~/[Pfad zu ma_jonathanmohr-main]/
  conda run -n nnodely-env python3 12_train_msnn_exp9_best.py

  Was passiert:
    Phase1: Lateral ψ̇  — 1500ep / patience=300 / lr=1e-3
    Phase2: Longitudinal aₓ — 3000ep / patience=500 / lr=1e-3
    [KEIN Phase3 — destabilisiert withax-Modell: +4.27% schlechter]

  Daten:
    Train: Day3 163324 erste 4333 Zeilen + Day2 113922 ab Zeile 1000
    Val:   Day3 163324 letzte 2167 Zeilen (Racing-Section, t=87–130s)

  Ausgabe:
    trained_models/msnn_exp9_withax_best/net.json
    trained_models/msnn_exp9_phase2_backup/net.json   ← DEPLOY DIESEN

  Erwartetes Ergebnis am Ende:
    [Phase2] yr=0.1158 (1.05×p)  ax=0.3947
    Backup saved to trained_models/msnn_exp9_phase2_backup/net.json

================================================================
SCHRITT 2 — Paper-Protokoll reproduzieren (exp6c)
================================================================

  Laufzeit: ~50–60 Minuten auf CPU

  conda run -n nnodely-env python3 11_train_msnn_full.py

  Was passiert (vollständig wie im Paper):
    Phase1: Lateral ψ̇  — 1500ep / patience=300 / lr=1e-3
    Phase2: Longitudinal aₓ — 3000ep / patience=500 / lr=1e-3
    Phase3: CL-Feedback — 1000ep / patience=200 / lr=1e-4
            (addClosedLoop für aₓ — ψ̇ bleibt immer Open-Loop)

  Daten:
    Train: Day2 alle Segmente (session_20260722_*.csv)
    Val:   Day3 alle Segmente (session_20260728_*.csv, ohne 124427)

  Erwartetes Ergebnis:
    [Phase2] yr≈0.1194  ax≈0.3935
    [Phase3] yr=0.1196 (1.09×p)  ax=0.3398   ← CL hilft aₓ: -6.8%

================================================================
SCHRITT 3 — Isaac Sim Integration
================================================================

  Für Isaac Sim Integration zwei Prozesse starten:

  TERMINAL 1 (nnodely-env, ZUERST starten):
    conda run -n nnodely-env python3 13_inference_msnn.py
    [Smoke Test läuft automatisch — prüft ob Modell geladen werden kann]

  Integration in Isaac Sim Python:

    import sys
    sys.path.insert(0, '/Pfad/zu/dynamics/msnn/src/')   # unnummerierte Module liegen dort
    from inference_msnn import MSNNInference

    model = MSNNInference(
        model_dir='/Pfad/zu/dynamics/deployed/trained_models/msnn_exp9_phase2_backup'
    )
    model.reset()   # MUSS vor jeder Episode aufgerufen werden!

    # Im Simulations-Loop (20ms Schritte):
    yr_pred, ax_pred = model.step(
        delta  = steering_cmd,    # [rad]  Lenkwinkel
        vel    = odom_speed,      # [m/s]  Längsgeschwindigkeit
        acc    = imu_accel_x,     # [m/s²] Längsbeschleunigung
        accy   = imu_accel_y,     # [m/s²] Querbeschleunigung
        crt    = motor_current    # [A]    Motorstrom
    )
    # yr_pred: vorhergesagte Gierrate [rad/s]
    # ax_pred: vorhergesagte Längsbeschleunigung [m/s²]

  HINWEIS: Falls nnodely nicht in Isaac Sim Python installierbar ist,
    den Dynamics-Server aus simulation/isaac_sim/05_msnn_dynamics_server.py
    als separaten Prozess starten (File-Queue IPC).

================================================================
SCHRITT 4 — GRU Run30 (Alternative für Isaac Sim)
================================================================

  GRU eignet sich besser für neue Tage/Strecken (cross-day robuster):
    MS-NN Same-Day:  yr=0.1158 (1.05×p)  ← besser auf bekannter Strecke
    GRU Cross-Day:   CL-RMSE=0.4448       ← besser auf neuen Strecken

  Smoke Test (nur torch nötig, kein nnodely):
    conda activate f1tenth_dynamics
    python3 15_gru_inference.py

  Integration in Isaac Sim (gru_inference.py unnummeriert, siehe INHALT
  DIESES ORDNERS oben, muss im Suchpfad liegen):
    from gru_inference import GRUInference
    gru = GRUInference(checkpoint='/Pfad/zu/dynamics/deployed/trained_models/gru_run30_best.pth')
    gru.reset()
    out = gru.step(vx=2.0, vy=0.0, yr=0.1, ax=0.2, ay=-0.1,
                   speed_cmd=2.0, steering_cmd=0.05)
    yr_pred = out['yr']   # Gierrate [rad/s]
    ax_pred = out['ax']   # Längsbeschleunigung [m/s²]

================================================================
SCHRITT 5 — Beide Modelle evaluieren + Plots erstellen
================================================================

  conda activate nnodely-env    (für MS-NN; GRU läuft auch ohne)
  pip install matplotlib        (falls nicht vorhanden)

  python3 14_eval_and_plot.py \
    --data-dir /Pfad/zu/ma_jonathanmohr-main/processed_data/

  Ausgabe in eval_output/:
    results_msnn_vs_gru.png    ← Balkendiagramm für Felix-Präsentation
    plot_yawrate_day3.png      ← ψ̇ Zeitverlauf Same-Day
    plot_yawrate_day1.png      ← ψ̇ Zeitverlauf Cross-Day
    results_comparison.csv     ← alle Zahlen

================================================================
FEHLERBEHEBUNG
================================================================

  "NameError: 'yaw_rate' already used as NeuObj"
    → Neuen Python-Prozess starten (kein zweites Modell im selben Prozess)
    → Oder am Skript-Anfang:
         from nnodely.basic.relation import NeuObj
         NeuObj.names.clear(); NeuObj.names.append('SampleTime')

  "NameError: 'eval_set' already used as Dataset"
    → Eindeutige Dataset-Namen verwenden (Zähler-Suffix im Code vorhanden)

  "FileNotFoundError: net.json"
    → Training noch nicht abgeschlossen, oder falscher model_dir Pfad

  ψ̇ RMSE > 0.15 nach Training
    → Prüfen: seed=42 gesetzt? (nnodely überschreibt alle Parameter mit seed)
    → Prüfen: session_20260728_124427 ausgeschlossen?
    → Phase3 nicht für withax ausführen (verschlechtert exp9!)

================================================================
BEKANNTE EINSCHRÄNKUNGEN
================================================================

  Cross-Day Domain Shift:
    MS-NN auf Day3 Val: yr=0.1158 (1.05×p)  ← gut
    MS-NN auf Day1 Test: yr≈0.1992 (2.26×p) ← erwarteter Shift zwischen Tagen

  CL-Evaluierung:
    ψ̇ ist IMMER Open-Loop (laterales Modell nutzt kein accy-Feedback)
    CL betrifft NUR aₓ via Fy(accy)-Pfad (±5–10% Verbesserung)

  Mirror-Augmentierung:
    NICHT für MS-NN verwenden! (LocalModel Fuzzy-Center werden gestört)
    Mirror funktioniert nur für GRU (kein strukturiertes Lokalmodell)
