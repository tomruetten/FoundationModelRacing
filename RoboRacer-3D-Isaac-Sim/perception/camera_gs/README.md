================================================================
perception/camera_gs — Anleitung zur Reproduktion
Ziel: PSNR = 27.61 dB mit nerfstudio splatfacto
================================================================

VORAUSSETZUNGEN
───────────────
  conda env:   nerfstudio-py310  (nerfstudio, gsplat, opencv-python, torch+CUDA)
  Hardware:    GPU mit ≥8 GB VRAM (Training ~2h auf RTX 3090)
  PROBE_ROOT:  Wurzel der rohen Probe-Aufnahmedaten (nicht in diesem Repo
               enthalten) — export PROBE_ROOT=/pfad/zu/Probe setzen
  Daten:       Rohdaten: $PROBE_ROOT/colmap_out/images/  (PNG-Sequenz aus ROS2-Bag)
  Vocab-Tree:  $PROBE_ROOT/colmap_undistort/colmap/vocab_tree.bin
               (Download: https://demuc.de/colmap/ → vocab_tree_flickr100K_words32K.bin)

================================================================
SCHRITT-FÜR-SCHRITT
================================================================

SCHRITT 1 — Bilder entzerren
─────────────────────────────
  WARUM: OAK-D Pro W hat HFOV=97°. Ohne Entzerrung schätzt COLMAP
         eine falsche Kameramatrix (fl≈1079 statt fl≈566), was zu
         systematischen Pose-Fehlern bei Kurvenfahrten führt.

  conda run -n nerfstudio-py310 python3 01_undistort_frames.py

  Eingabe:  $PROBE_ROOT/colmap_out/images/*.png
  Ausgabe:  $PROBE_ROOT/colmap_undistort/images/*.png
  (Alle Bilder werden auf 1280×600 gecroppt — Fahrzeugkarosserie entfernt)

  Kalibrierungsparameter (bereits im Skript, NICHT ändern):
    K = [[566.26, 0, 641.96], [0, 565.95, 388.61], [0, 0, 1]]
    dist = [-0.0624, 0.0467, ...]  # 8-Parameter rational_polynomial

SCHRITT 2 — COLMAP ausführen
──────────────────────────────
  WARUM: PINHOLE-Modell nach Entzerrung korrekt (keine weitere Verzeichnung).
         Loop-Detection entscheidend für Rundstrecke (geschlossene Trajektorie).

  bash 03_colmap_commands.sh

  Was passiert intern:
    a) feature_extractor: SIFT-Features, PINHOLE, feste K aus Schritt 1
    b) sequential_matcher: overlap=20 Frames + Vocab-Tree Loop-Detection
    c) mapper: inkrementelles Bundle Adjustment
    d) model_converter → transforms.json für nerfstudio

  Ausgabe: $PROBE_ROOT/experiments/new_recording_20260714/dataset/transforms.json

  Kontrolle: Am Ende steht "N images registered" — sollte ≥900 sein.

SCHRITT 3 — splatfacto Training
────────────────────────────────
  WARUM: --use-scale-regularization verhindert elongierte Gaussianer.
         50k Iterationen (statt Standard 30k) für volle Konvergenz.

  bash 04_train_and_eval.sh

  Was passiert intern:
    a) ns-train splatfacto ... --max-num-iterations 80000
    b) ns-eval → PSNR, SSIM, LPIPS werden ausgegeben

  Laufzeit: ~2 Stunden auf RTX 3090
  Erwartetes Ergebnis: PSNR ≈ 27.61 dB

  Checkpoint-Pfad (automatisch ermittelt):
    $PROBE_ROOT/experiments/new_recording_20260714/outputs/camera_base/
      dataset/splatfacto/DATUM_UHRZEIT/config.yml

  Für Isaac Sim: diesen config.yml-Pfad in simulation/isaac_sim/01_camera_render_server.py
                 als --config Argument eintragen.

================================================================
FEHLERBEHEBUNG
================================================================

Problem: COLMAP registriert <500 Bilder
  → Vocab-Tree-Pfad prüfen (VOCAB_TREE in 03_colmap_commands.sh)
  → overlap auf 30 erhöhen

Problem: PSNR < 25 dB
  → Entzerrung prüfen: Bilder in colmap_undistort/images/ sollten
    gerade Linien zeigen (kein Fischaugeneffekt mehr)
  → COLMAP-Trajektorie in nerfstudio Viewer prüfen (ns-viewer --load-config)

Problem: CUDA out of memory
  → Batch-Größe reduzieren: --pipeline.model.num-random-rays-per-batch 8192
  → Oder GPU mit mehr VRAM nutzen (≥16 GB für komfortables Training)

================================================================
ERZIELTE ERGEBNISSE
================================================================

  PSNR  = 27.61 dB
  SSIM  = 0.885
  LPIPS = 0.228
  Checkpoint: 2026-07-14_231816

  Vergleich mit Baseline (ohne Entzerrung, COLMAP selbst-kalibriert):
    PSNR ≈ 22-24 dB  — Entzerrung bringt +3 bis +5 dB!

================================================================
WICHTIGER HINWEIS FÜR NEUE AUFNAHMEN (Ergänzung 2026-09-07)
================================================================

  PROBLEM: Die Kamera braucht nach dem Einschalten ca. 1 Sekunde für
  die automatische Belichtungsanpassung (Auto-Exposure-Warmup). Die
  Frames aus diesem Fenster sind sichtbar überbelichtet ("Weißlicht").
  Werden sie mit ins Training genommen, sinkt der PSNR messbar.

  BELEG: Auf zwei unabhängigen Strecken (Sessions 20260827_142014 und
  20260827_143212, siehe 27.08_neue_Strecke_und_Dynamik/, nicht in
  diesem Repo enthalten) wurde das Ausschließen
  dieser Startframes ("noflare"-Varianten) unabhängig verifiziert:
  PSNR jeweils +0.34 dB, PSNR-Standardabweichung ebenfalls niedriger
  (konsistente Qualität über alle Ansichten, kein Zufallseffekt).

  EMPFEHLUNG FÜR SCHRITT 1: Vor dem Entzerren die ersten Frames der
  neuen Aufnahme stichprobenartig ansehen (überbelichtet ja/nein? nicht
  jede Session zeigt den Effekt). Falls ja: die ersten ~1 Sekunde
  Rohvideo verwerfen, BEVOR 01_undistort_frames.py läuft.

  Für neuere Sessions steht dafür bereits ein fertiger Parameter in der
  aktuelleren Extraktions-Pipeline zur Verfügung (zeitbasiert, nicht
  framezahl-basiert, weil die FPS zwischen Sessions unterschiedlich
  ist -- 30fps vs. ~60fps je nach Aufnahme):
    extract_undistort_frames_full.py --skip-seconds 1.0
  (dieses Skript lag in einem alten, mittlerweile gelöschten
  Arbeitsordner ("1707-Test") -- nicht in diesem Repo enthalten,
  bei Bedarf melde ich mich mit dem aktuellen Pfad)

  Volle Herleitung/Details (nicht in diesem Repo enthalten):
  27.08_neue_Strecke_und_Dynamik/COLMAP_Pose_3DGS_Punktwolke_
  Fahrzeugtrajektorie_Sync_Gesamtablauf_20260902.md,
  Abschnitt "0. Vor Beginn einer neuen Session".
