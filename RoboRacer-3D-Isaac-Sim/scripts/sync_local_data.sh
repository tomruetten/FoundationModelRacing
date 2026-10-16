#!/usr/bin/env bash
# Copy the run-time data (checkpoints, poses, CSVs, scenes, scripts) from the
# T7 data drive into data/local/. Skips raw bags, videos and frame dumps. ~14 GB.
#   source scripts/env.sh && bash scripts/sync_local_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${DATA_DRIVE_ROOT:?source scripts/env.sh first}"
L="$ROOT/data/local"; mkdir -p "$L/sessions"
M="$D/Zusammenfassung/5_MSNN_for_Lab/Code von Felix/ma_jonathanmohr-main"
rsync -a "$D/New1/" "$L/New1/"
rsync -a "$D/Probe/colmap_undistort" "$D/Probe/experiments" "$L/Probe/"
rsync -a "$M/processed_data/" "$L/processed_data/"
rsync -a "$M/trained_models/" "$L/dynamics_trained_models/"
rsync -a --exclude '*.bak' --exclude '__pycache__' "$D/1707-Test/scripts/" "$L/pipeline_scripts_1707/"
rsync -a "$D/calibration_summary/" "$L/calibration_summary/"
rsync -a "$D/27.08新赛道和Dynamik/3dgs session_20260827_142014_noflare/" "$L/sessions/session_20260827_142014_noflare/"
rsync -a --exclude 'isaac_sim_output/live_render*' \
  "$D/27.08新赛道和Dynamik/dynamik session_20260827_143212_noflare/" "$L/sessions/session_20260827_143212_noflare/"
S3="$D/07,09,2026 New 3DGS/zhui tong session_20260904_161841"; T3="$L/sessions/session_20260904_161841_zhuitong"
mkdir -p "$T3/isaac_sim_output"
rsync -a --exclude 'images_full' "$S3/outputs" "$S3/dataset" "$T3/"
rsync -a --include '*.usd' --include '*.usda' --include '*.json' --exclude '*' "$S3/isaac_sim_output/" "$T3/isaac_sim_output/"
du -sh "$L"
