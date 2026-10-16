================================================================
F1TENTH Car02 — Code-Paket
Supervisor: Felix Jahncke (TUM)
Stand: 2026-09-07
================================================================

Dieses Paket ist eine kuratierte Auswahl aus meiner Projektablage
(Zusammenfassung/) — nur Code, kleine Referenzdaten (Checkpoints,
Kalibrierung, Raceline) und README-Dateien. Rohdaten (Videos, ROS-Bags,
COLMAP-Rekonstruktionen, volle Trainings-CSVs, Trainingslogs, Experiment-
Sweeps) sind bewusst NICHT enthalten — die liegen auf der Lab-Maschine
auf Anfrage zugänglich.

================================================================
SCHNELLÜBERBLICK: WAS IST WAS
================================================================

perception/camera_gs/        Kamera → 3D Gaussian Splatting (nerfstudio splatfacto)
                       Ältere Referenz-Pipeline (layout02-Szene, 26.20dB).
                       Methodisch sauber, funktioniert noch.

perception/lidar_gs/          LiDAR-Punktwolke → eigenes 3DGS-Training
                       Ziel: Chamfer Distance 13.44cm (Livox Mid-360)

simulation/isaac_sim/           Kamera- und LiDAR-Splat live in Isaac Sim rendern
                       (Dual-Process IPC: Isaac Sim ↔ separater Render-Server,
                       nötig weil Isaac Sim seine eigene Python-Umgebung hat)

simulation/closed_loop/        Closed-Loop Regelung in Isaac Sim: Pure-Pursuit-
                       Pfadverfolgung + GRU/MS-NN als Dynamik-Modell im
                       Regelkreis, getestet auf 7 Strecken

dynamics/training/    GRU / Deep Dynamics / MS-NN — Daten-Pipeline,
                       Architektur, Training, Inferenz (nummerierte Schritte
                       01-13, chronologisch durchlaufen)

dynamics/msnn/        Felix' Original-Codebase (in src/, direkt aus
                       ma_jonathanmohr-main/ übernommen: Dataset_Classes.py,
                       RNN_architectures.py, training.py etc.) + unsere
                       Trainings-/Eval-Experimente + die 3 Paper-Notebooks
                       (Lateral/Longitudinal/Closed-Loop Dynamics)

dynamics/deployed/     Schlanke, sofort lauffähige Inferenz-Skripte +
                       die zwei aktuell deployten Checkpoints:
                       gru_run30_best.pth, msnn_exp9_phase2_backup/net.json

dynamics/retraining_20260815/ Fünf Nachtrainings-Experimente vom 2026-08-15
                       (a1_widemargin, a2_phase3_lowerlr, run38 scheduled
                       sampling, run39 h200, run40 2layer) — Skripte +
                       resultierende Checkpoints. Trainingslogs nicht
                       enthalten (auf Anfrage verfügbar).

reference/        Kalibrierung (Kamera-Intrinsics, Extrinsics,
                       Sensor-TF-Nutzung) + Raceline/Track-JSON für die
                       V2-Demo (FTM Halle, min-time Trajektorie)

reference/vehicle/   Gemessene Fahrzeugparameter (Masse, Radstand,
                       Außenmaße, Sensor-Montagepositionen) + Belegfotos
                       der Handmessungen und Achslast-Wiegung

================================================================
AKTUELLE ZAHLEN (Quick Reference)
================================================================

Modell/Pipeline                          Metrik               Ergebnis
──────────────────────────────────────────────────────────────────────
Camera 3DGS (aktuelle Referenz-Pipeline) PSNR                 27.24 dB
LiDAR 3DGS (deployed)                    Chamfer / F@10cm     26.3 cm / 46.1%
Isaac Sim Camera+LiDAR Live-Demo         Frames / Real-Hit    530 / 99.8%
V2 Closed-Loop (7 Strecken)              Tracking-Fehler      0.05–0.15 m
V2 Adaptive Hybrid-A (7 Strecken)        max θ-Drift          14.7° (Limit 28.6°)
GRU Run30 (deployed)                     3ch-CL-RMSE Day1     0.4448 rad/s
MS-NN exp9 Phase2 (deployed)             yr RMSE gleichtagig  0.1158 rad/s

Hinweis: Die aktuellste, methodisch sauberste Camera/LiDAR-3DGS-Pipeline
(inkl. neuer Streckenaufnahmen vom 2026-08-27 und 2026-09-04, "noflare"-
Belichtungskorrektur, Isaac-Sim-Integration mit echten Fahrzeugtrajektorien)
läuft aktuell weiter und ist nicht Teil dieses Code-Schnappschusses — bei
Bedarf gerne den aktuellen Stand direkt zeigen/teilen.

================================================================
SOFTWARE- UND HARDWARE-VERSIONEN (Stand 2026-09-07)
================================================================

System
  OS:              Ubuntu 22.04.5 LTS
  GPU:             NVIDIA GeForce RTX 4080 SUPER (Treiber 580.119.02)
  ROS2:            Humble

Isaac Sim
  Version:         6.0.1-rc.7 (release build 42383)
  conda env:       env_isaac6 (Python 3.12.13 — eigene, von den
                   anderen envs isolierte Python-Version, daher die
                   Dual-Process-IPC-Architektur in simulation/isaac_sim/)

Camera/LiDAR-3DGS + GRU (conda env "roboracer", Python 3.10.20)
  torch:           2.6.0+cu124
  nerfstudio:      1.1.5
  gsplat:          1.5.3+pt24cu124
  open3d:          0.19.0
  COLMAP:          3.7 (CPU-Build, ohne CUDA — SiftExtraction/Matching
                   laufen in allen Skripten bewusst mit --use_gpu 0,
                   damit sie parallel zu GPU-Training laufen können)

MS-NN (conda env "nnodely-env", Python 3.10.20, CPU-only)
  nnodely:         1.5.4

AutoDRIVE-Bridge (conda env "autodrive-bridge", Python 3.10.20)
  xdotool-basierte Steuerung des AutoDRIVE-Simulators, siehe simulation/closed_loop/

Ältere Camera/LiDAR-3DGS-Skripte (1_/2_/3_ Ordner) liefen ursprünglich
unter einer separaten "nerfstudio-py310"-Umgebung; funktional identisch
zur obigen "roboracer"-Umgebung, nur ein anderer conda-Name aus einer
früheren Projektphase.

================================================================
UMGEBUNGEN EINRICHTEN
================================================================

Dieses Projekt braucht mehrere getrennte conda-Umgebungen (unterschiedliche
Python-Versionen / teils inkompatible torch-Builds — NICHT in eine einzige
Umgebung zusammenlegen). Fertige environment.yml-Dateien liegen in envs/:

  envs/environment-gs.yml       Umgebung "roboracer" (in ein paar älteren
                                      READMEs noch als "f1tenth_dynamics"
                                      bezeichnet, ist dieselbe Umgebung) —
                                      für perception/camera_gs/, perception/lidar_gs/,
                                      GRU/Deep Dynamics (dynamics/training/),
                                      simulation/closed_loop/. torch 2.6.0+cu124,
                                      nerfstudio 1.1.5, gsplat, open3d.

  envs/environment-nnodely.yml       Umgebung "nnodely-env" (CPU-only!) — für
                                      MS-NN (dynamics/training/, dynamics/msnn/).

  envs/environment-isaac-sim.yml     Umgebung "env_isaac6" (Python 3.12.13) —
                                      Python-Seite der IPC-Anbindung an Isaac Sim
                                      (die App selbst NICHT enthalten, separat via
                                      NVIDIA installieren: Isaac Sim 6.0.1-rc.7).

  Einrichten:
    conda env create -f envs/environment-gs.yml
    conda env create -f envs/environment-nnodely.yml
    conda env create -f envs/environment-isaac-sim.yml

  Für die "autodrive-bridge"-Umgebung (xdotool-Steuerung für AutoDRIVE
  Simulator, siehe Software-Tabelle oben) habe ich noch keine Paketliste
  aufgeschrieben — melde dich einfach bei mir, falls die gebraucht wird,
  dann ergänze ich das.

  nerfstudio-Installation kann eigenwillig sein (kompilierte CUDA-Extensions
  über gsplat/tiny-cuda-nn). Falls environment-gs.yml daran scheitert,
  lieber der offiziellen nerfstudio-Anleitung für Version 1.1.5 folgen:
  https://docs.nerf.studio/quickstart/installation.html

  Ausführung: kein `pip install -e .` / kein Package-Import — die Ordner
  (1_..., 2_..., etc.) beginnen mit Ziffern und sind keine gültigen Python-
  Modulnamen. Skripte werden direkt ausgeführt, z.B.:
    conda activate roboracer
    python dynamics/training/06_train_gru_run30_best.py

================================================================
SENSOREN & KALIBRIERUNG
================================================================

Kamera: OAK-D Pro W (Luxonis), RGB-Sensor CAM_A
  Auflösung (alle Aufnahmen): 1280x800
  HFOV:            97°
  Intrinsics (factory calibration, via depthai API):
    fx=566.2562  fy=565.9493  cx=641.9587  cy=388.6107
  Distortion-Modell: rational_polynomial (8 Koeffizienten)
  Volle Werte inkl. Left-Mono-Kamera (640x400) und Stereo-Baseline
  (7.5cm): reference/calibration/camera_intrinsics.yaml
  Kamera→Fahrzeug-Extrinsics: reference/calibration/
  extrinsics_candidate_checked_v2.yaml

LiDAR: Livox Mid-360
  Auch Quelle des integrierten IMU (/livox/imu, 200Hz → downsampled
  auf 50Hz für Vehicle-Dynamics-Training)

Fahrzeug: F1TENTH Car02
  Masse:           3.788 kg (gemessen, SkyRC Corner Weight Scale)
  Radstand:        0.3225 m
  lf / lr:         0.1584 m / 0.1641 m (CoG → Vorder-/Hinterachse)
  Außenmaße:       0.42 m lang / 0.26 m breit (über Räder/Reifen),
                   Reifenbreite 0.04 m
  Lenkwinkel:      ±0.29 rad (≈±16.6°)
  Geschwindigkeit: 0–3.5 m/s
  ERPM→m/s:        ERPM_TO_MS = 1/7614.0 (kalibrierter VESC-Wert;
                   ACHTUNG: eine rein physikalische Herleitung ergibt
                   ~5832, also ≈30% Abweichung — 7614 ist der korrekte,
                   kalibrierte Wert. Betrifft nur odom_speed, nicht
                   speed_cmd, also nicht das GRU/MS-NN-Training)
  Sensor-Montage:  Kamera/LiDAR/MoCap-Marker-Offsets relativ zu
                   base_link, inkl. Kamera↔LiDAR-Extrinsik (T_cl)
  Volle Parameter-Übersicht inkl. Außenmaßen, Sensor-Montagepositionen,
  Kamera-/LiDAR-Spezifikationen, gelernter MS-NN-Modellparameter und
  Belegfotos der Handmessungen:
  reference/vehicle/vehicle_parameters.txt + reference/vehicle/fotos/

================================================================
KRITISCHE DATEN-CONSTRAINT (nie verletzen)
================================================================

TEST ONLY:  session_20260721_20ms.csv           ← Day1, nur finale Evaluation
TRAIN/VAL:  session_20260722_*.csv              ← Day2
            session_20260728_*.csv              ← Day3
AUSSCHLUSS: session_20260728_124427_*.csv       ← 100% NaN IMU

================================================================
HINWEISE ZUR NUTZUNG
================================================================

- Jeder Ordner 1-6 hat eine eigene README.md mit Voraussetzungen,
  Schritt-für-Schritt-Anleitung und erwartetem Ergebnis.
- dynamics/training/ und dynamics/msnn/src/ teilen sich dieselben
  Kernmodule (Dataset_Classes.py, RNN_architectures.py) — letztere
  bringt sie direkt mit, erstere setzt voraus, dass diese Module im
  Suchpfad liegen (siehe jeweilige README).
- Große Rohdaten (Videos, ROS-Bags, volle CSV-Trainingssätze,
  COLMAP-Rekonstruktionen) sind nicht enthalten, da sie den Rahmen
  eines Code-Pakets sprengen würden — bei Bedarf einfach melden.
- Stand: 2026-09-07
