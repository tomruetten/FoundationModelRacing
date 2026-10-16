================================================================
perception/lidar_gs — Anleitung zur Reproduktion
Ziel: Chamfer Distance = 13.44 cm (Livox Mid-360)
================================================================

VORAUSSETZUNGEN
───────────────
  conda env:   nerfstudio-py310  (gsplat, torch+CUDA, open3d, scipy)
  Hardware:    GPU mit ≥6 GB VRAM
  LIDAR3DGS_ROOT / PROBE_ROOT: Wurzelverzeichnisse der rohen Aufnahmedaten
               (nicht in diesem Repo enthalten) — vor dem Ausführen setzen:
               export LIDAR3DGS_ROOT=/pfad/zu/Newmocap/layout_02_slow_run_02
               export PROBE_ROOT=/pfad/zu/Probe
  Daten:       ROS2-Bag: $LIDAR3DGS_ROOT/run_20260327_105224_0.db3
  COLMAP:      Camera-COLMAP transforms.json muss bereits fertig sein
               (aus perception/camera_gs, Pfad: ...new_recording_20260714/dataset/transforms.json)
  Sync-CSV:    camera_sync.csv im BAG-Verzeichnis (aus Schritt 1 unten)

================================================================
SCHRITT-FÜR-SCHRITT
================================================================

SCHRITT 1 — camera_sync.csv erstellen (LiDAR ↔ Kamera Timestamps)
───────────────────────────────────────────────────────────────────
  WARUM: Jeder LiDAR-Scan bekommt den zeitlich nächsten Kamera-Frame
         zugewiesen. So können COLMAP-Posen für LiDAR-Frames genutzt werden.

  conda run -n nerfstudio-py310 python3 02_sync_lidar_camera.py

  Ausgabe: $LIDAR3DGS_ROOT/camera_sync.csv
  Format:  lidar_timestamp_ns, camera_frame_idx, matched (True/False)

SCHRITT 2 — T_cl Kalibrierung prüfen / anpassen (optional)
────────────────────────────────────────────────────────────
  Die Kalibrierungswerte sind bereits im Skript 04_train_lidar_3dgs.py
  eingetragen (AprilGrid-Kalibrierung 2026-07-14):

    T_CL_YAML_Q = [-0.47431, 0.49898, -0.50795, 0.51772]  # xyzw
    T_CL_YAML_T = [ 0.02267, 0.06987, -0.04441]            # [m]

  WICHTIG: Diese Werte verwenden ROS TF child-frame-Konvention.
           Das Skript rechnet automatisch R_cl = R_yaml.T (Transponierte).
           Nicht direkt ändern ohne 01_calibrate_tcl.py neu auszuführen.

  Zur Neu-Kalibrierung:
    conda run -n nerfstudio-py310 python3 01_calibrate_tcl.py

SCHRITT 3 — LiDAR 3DGS trainieren
────────────────────────────────────
  WARUM: --use-colmap eliminiert MoCap-Drift-Fehler (MoCap hatte Pitch-Offset
         von -8.91° und Z-Fehler von +13.4 cm — COLMAP-Posen sind präziser).
         --densify klont Gaussianer an Stellen mit hohem Gradienten.
         --raydrop lernt, welche Pixel wirklich LiDAR-Returns haben.

  conda run -n nerfstudio-py310 \
    python3 04_train_lidar_3dgs.py \
      --use-colmap \
      --steps 30000 \
      --densify \
      --raydrop \
      --exp lidar_3dgs_colmap_v1

  Laufzeit: ~45 min auf RTX 3090 (30k Schritte, ~1000 LiDAR-Frames)
  Checkpoint: $PROBE_ROOT/experiments/lidar_3dgs_colmap_v1/best.pt

SCHRITT 4 — Evaluierung (Chamfer Distance)
────────────────────────────────────────────
  conda run -n nerfstudio-py310 \
    python3 04_train_lidar_3dgs.py \
      --use-colmap \
      --densify \
      --raydrop \
      --render \
      --exp lidar_3dgs_colmap_v1

  Ausgabe:
    F-score@0.05m: ...
    F-score@0.10m: 0.78
    F-score@0.20m: ...
    Chamfer Distance: 0.1344 m   ← Zielwert

  Außerdem gespeichert:
    $PROBE_ROOT/experiments/lidar_3dgs_colmap_v1/synthetic_lidar/synthetic.ply
    $PROBE_ROOT/experiments/lidar_3dgs_colmap_v1/synthetic_lidar/real_holdout.ply

================================================================
PARAMETERERKLÄRUNG
================================================================

  --use-colmap   Nutzt COLMAP c2w × T_cl für LiDAR-Weltposen
                 (Alternative ohne Flag: MoCap-Interpolation, ~2 cm schlechter)

  --steps        30000 reicht, mehr bringt kaum Verbesserung

  --densify      Clone/Prune alle 1000 Schritte (erste 60% des Trainings)
                 Erhöht Anzahl der Gaussianer an geometrisch komplexen Stellen

  --raydrop      Kanal 1 der "colors" wird als Raydrop-Logit trainiert
                 (binäre Klassifikation: hat dieser Pixel einen LiDAR-Return?)
                 Verbessert geometrische Qualität in Randbereichen

  --cubemap      6-Face Cubemap für 360° Livox FOV (Standard: 6 Faces aktiv)

================================================================
FEHLERBEHEBUNG
================================================================

Problem: "camera_sync.csv not found"
  → Schritt 1 (02_sync_lidar_camera.py) zuerst ausführen

Problem: Chamfer > 20 cm
  → T_cl prüfen: in 04_train_lidar_3dgs.py print(T_cl) einfügen und
    sicherstellen, dass R_cl_opencv = R.from_quat(Q).as_matrix().T (mit .T!)

Problem: Wenige Frames (<500), Training zu kurz
  → BAG-Pfad in 04_train_lidar_3dgs.py prüfen (ROOT, BAG Variablen)
  → camera_sync.csv: n_matched prüfen (sollte >500 sein)

Problem: CUDA out of memory
  → MAX_RANGE_M auf 6.0 reduzieren (weniger Punkte pro Frame)

================================================================
ERZIELTE ERGEBNISSE
================================================================

  Chamfer Distance = 13.44 cm   (0.1344 m)
  F-score@0.10m   = 0.78

  Konfigurationsvergleich:
    MoCap (ohne --use-colmap):  ~15-18 cm  — schlechter wegen Drift
    COLMAP + densify + raydrop: 13.44 cm   ← beste Konfiguration
