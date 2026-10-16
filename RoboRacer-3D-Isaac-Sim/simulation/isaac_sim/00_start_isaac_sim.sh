#!/bin/bash
# Isaac Sim — Startreihenfolge für alle drei Fenster
#
# Architektur:
#   [Isaac Sim]  Car02-Mesh + Fahrzeugbewegung
#       ├── Fenster 1: Camera-3DGS (splat_render_server + isaac_sim_camera_splat_demo)
#       └── Fenster 2: LiDAR-3DGS (lidar_render_server + isaac_sim_lidar_splat_demo)
#
# Starten in dieser Reihenfolge (jeweils neues Terminal):
#
# PROBE_ROOT: Wurzel der rohen Probe-Aufnahmedaten (nicht in diesem Repo
# enthalten, siehe root README) — vor dem Ausführen setzen:
#   export PROBE_ROOT=/pfad/zu/Probe

PROBE="${PROBE_ROOT:-./data/Probe}"
CAMERA_CONFIG="$PROBE/experiments/new_recording_20260714/outputs/camera_base/dataset/splatfacto/2026-07-14_231816/config.yml"
LIDAR_CHECKPOINT="$PROBE/experiments/new_recording_20260714/outputs/lidar_3dgs_raydrop/best.pt"

echo "=== Startreihenfolge ==="
echo ""
echo "Terminal 1 — Camera-3DGS Render-Server (nerfstudio-py310 Env):"
echo "  conda run -n nerfstudio-py310 python3 $PROBE/scripts/splat_render_server.py \\"
echo "      --config $CAMERA_CONFIG"
echo ""
echo "Terminal 2 — LiDAR Render-Server (nerfstudio-py310 Env):"
echo "  conda run -n nerfstudio-py310 python3 $PROBE/scripts/lidar_render_server.py \\"
echo "      --checkpoint $LIDAR_CHECKPOINT"
echo ""
echo "Terminal 3 — Isaac Sim (startet Car02-Mesh + Bewegung):"
echo "  [Isaac Sim GUI starten, dann USD-Scene laden]"
echo "  [Camera-Splat-Client wird als Extension/Script in Isaac Sim ausgeführt:]"
echo "  Script: $PROBE/scripts/isaac_sim_camera_splat_demo.py"
echo ""
echo "Terminal 4 — LiDAR-Splat-Client (in Isaac Sim):"
echo "  Script: $PROBE/scripts/isaac_sim_lidar_splat_demo.py"
echo ""
echo "Warum zwei Prozesse pro Renderer?"
echo "  nerfstudio: Python 3.10 + PyTorch 2.x"
echo "  Isaac Sim:  eigenes eingebettetes Python/CUDA-Stack"
echo "  → CUDA-Versionskonflikt → Datei-Queue als IPC-Bridge"
