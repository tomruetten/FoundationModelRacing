================================================================
simulation/isaac_sim — Anleitung zur Reproduktion
Ziel: Camera- und LiDAR-Splat in Isaac Sim rendern (Dual-Process IPC)
================================================================

VORAUSSETZUNGEN
───────────────
  Prozess B (Render-Server):
    conda env: nerfstudio-py310  (nerfstudio, gsplat, pillow)
    Fertige Checkpoints:
      Camera: ...splatfacto/2026-07-14_231816/config.yml   (27.61 dB)
      LiDAR:  $PROBE_ROOT/experiments/lidar_3dgs_colmap_v1/best.pt

  Prozess A (Isaac Sim Client):
    Isaac Sim 6.0.1-rc.7 (Omniverse Kit, release build 42383)
    conda env: env_isaac6 (Python 3.12.13 — eigene Isaac-Sim-Python-
    Version, getrennt von den anderen conda-envs, daher die Dual-Process-
    IPC-Architektur statt eines gemeinsamen Prozesses)

  WARUM zwei Prozesse?
    Isaac Sim bringt eine eigene gebündelte Python/torch/CUDA-Umgebung.
    nerfstudio+gsplat brauchen einen eigenen torch-Build.
    Beide im selben Prozess → CUDA-Konflikte.
    Lösung: Filesystem-Queue (JSON + PNG) als IPC-Kanal.

================================================================
IPC-PROTOKOLL (wie die Prozesse kommunizieren)
================================================================

  Prozess A schreibt:   queue_dir/request_N.json
                        {"c2w": [[...4×4 Matrix...]], "already_in_model_frame": false}

  Prozess B rendert und schreibt:  queue_dir/response_N.png
  Prozess B löscht:    request_N.json  (= "fertig"-Signal)

  Prozess A liest response_N.png sobald vorhanden → nächster Frame

================================================================
SCHRITT-FÜR-SCHRITT: CAMERA SPLAT
================================================================

TERMINAL 1 — Render-Server starten (ZUERST!)
──────────────────────────────────────────────
  conda run -n nerfstudio-py310 \
    python3 01_camera_render_server.py \
      --queue-dir /tmp/isaac_queue/camera \
      --checkpoint cropped

  Checkpoint-Optionen (in splat_camera_renderer.py definiert):
    cropped    → 27.61 dB  BEST (1280×600, kein Fahrzeugfront)  ← empfohlen
    lidarinit  → 19.71 dB aber LPIPS=0.277 (besser bei Perceptual-Metrik)
    neartriage → 25.05 dB  (Nah-Triage-Variante)

  Warten bis erscheint: "[splat_render_server] watching /tmp/isaac_queue/camera ..."

TERMINAL 2 — Isaac Sim Client starten
───────────────────────────────────────
  # In Isaac Sim Python-Umgebung (nicht conda!):
  python3 02_camera_splat_client_isaac.py \
    --queue-dir /tmp/isaac_queue/camera

  Was passiert:
    - Isaac Sim fährt eine Trajektorie ab
    - Für jeden Schritt: c2w-Matrix → request_N.json schreiben
    - Warten auf response_N.png → Frame in Isaac Sim einblenden

================================================================
SCHRITT-FÜR-SCHRITT: LIDAR SPLAT (optional, parallel zu Camera)
================================================================

TERMINAL 3 — LiDAR Render-Server
──────────────────────────────────
  conda run -n nerfstudio-py310 \
    python3 03_lidar_render_server.py \
      --queue-dir /tmp/isaac_queue/lidar

TERMINAL 4 — LiDAR Isaac Sim Client
──────────────────────────────────────
  python3 04_lidar_splat_client_isaac.py \
    --queue-dir /tmp/isaac_queue/lidar

================================================================
KOORDINATENTRANSFORMATION (wichtig!)
================================================================

  Isaac Sim nutzt OpenGL-Konvention: X-rechts, Y-oben, Z-zurück
  nerfstudio nutzt dieselbe Konvention (OpenGL)
  → c2w direkt übergeben: {"already_in_model_frame": false}

  SplatCameraRenderer.render() übernimmt die interne Konvertierung:
    Isaac Sim c2w → nerfstudio-Kamerakoordinaten

  BEKANNTE FEHLERQUELLE (behoben 2026-06):
    Isaac Sim generierte c2w mit falscher Convention → Render ~10 cm versetzt.
    Fix: Koordinaten-Offset-Korrektur in SplatCameraRenderer.render().
    Status: BEHOBEN. Bei neuem Isaac Sim Update prüfen.

================================================================
SERVER STOPPEN
================================================================

  Stop-Signal senden (sauber):
    touch /tmp/isaac_queue/camera/STOP

  Danach beendet sich der Server nach dem aktuellen Frame.

================================================================
FEHLERBEHEBUNG
================================================================

Problem: "ModuleNotFoundError: nerfstudio"
  → Sicherstellen, dass Terminal 1 in nerfstudio-py310 läuft:
    conda run -n nerfstudio-py310 python3 01_camera_render_server.py ...

Problem: response_N.png kommt nicht (Isaac Sim hängt)
  → Terminal 1 prüfen: Fehlermeldung?
  → Queue-Verzeichnis prüfen: ls /tmp/isaac_queue/camera/
  → Render-Server neu starten, STOP-Datei löschen falls vorhanden

Problem: Schwarze Bilder / falscher Winkel
  → c2w-Matrix prüfen: muss 4×4 float-Matrix sein (nicht 3×4)
  → "already_in_model_frame": false setzen (nicht true)

Problem: CUDA-Fehler beim Server-Start
  → Anderen GPU-Prozess prüfen: nvidia-smi
  → Isaac Sim läuft auf derselben GPU? → GPU-ID in splat_camera_renderer.py trennen

================================================================
SCHRITT-FÜR-SCHRITT: DYNAMIK-IN-THE-LOOP DEMO (optional, V1)
================================================================

Analog zu Camera/LiDAR-Splat (Dual-Process IPC), aber für die
Fahrzeugdynamik-Modelle (MS-NN/GRU) statt für Rendering. 07_ spielt
eine reale Trajektorie in Isaac Sim ab und vergleicht MS-NN/GRU-
Vorhersagen live gegen die echten Sensordaten.

TERMINAL 1 — MS-NN Dynamics-Server (nnodely-env):
  conda run -n nnodely-env python3 05_msnn_dynamics_server.py \
    --model ../../dynamics/deployed/trained_models/msnn_exp9_phase2_backup \
    --queue-dir isaac_sim_output/msnn_queue

TERMINAL 2 — GRU Dynamics-Server, Open-Loop (roboracer env):
  conda run -n roboracer python3 06_gru_dynamics_server.py \
    --checkpoint ../../dynamics/deployed/trained_models/gru_run30_best.pth \
    --queue-dir isaac_sim_output/gru_queue

TERMINAL 3 — GRU Dynamics-Server, Closed-Loop (ZWEITE, unabhängige
Instanz -- eigener Puffer-Zustand, MUSS anderes --queue-dir haben):
  conda run -n roboracer python3 06_gru_dynamics_server.py \
    --checkpoint ../../dynamics/deployed/trained_models/gru_run30_best.pth \
    --queue-dir isaac_sim_output/gru_cl_queue

TERMINAL 4 — GRU-MoCap Dynamics-Server, Open-Loop (roboracer env,
echte 7in/5out MoCap-Architektur statt onboard-only, siehe
dynamics/msnn/15_gru_inference.py's GRUMoCapInference-Docstring):
  conda run -n roboracer python3 08_gru_mocap_dynamics_server.py \
    --checkpoint ../../dynamics/msnn/local_run___20260731204259241445.pth \
    --queue-dir isaac_sim_output/gru_mocap_queue

TERMINAL 5 — Isaac Sim Demo starten (nachdem alle 4 Server "watching"/
"Warte auf Anfragen" melden):
  python3 07_dynamics_isaac_demo.py \
    --scene <pfad-zu>/scene.usda \
    --msnn-queue-dir isaac_sim_output/msnn_queue \
    --gru-queue-dir isaac_sim_output/gru_queue \
    --gru-cl-queue-dir isaac_sim_output/gru_cl_queue \
    --gru-mocap-queue-dir isaac_sim_output/gru_mocap_queue

  --csv (default: Day1-Testsatz), --output-dir, --stride, --gui
  (Standard: headless) sind optional, siehe --help. --scene zeigt auf
  eine exportierte .usda-Streckenszene (nicht in diesem Repo enthalten).

================================================================
TYPISCHER ABLAUF (Zusammenfassung)
================================================================

  1. Terminal 1: conda run -n nerfstudio-py310 python3 01_camera_render_server.py --checkpoint cropped
     (warten bis "watching ..." erscheint)

  2. Terminal 2: python3 02_camera_splat_client_isaac.py
     (Isaac Sim-Trajektorie startet automatisch)

  3. Frames werden in Echtzeit gerendert und in Isaac Sim angezeigt.

  4. Zum Beenden: touch /tmp/isaac_queue/camera/STOP
