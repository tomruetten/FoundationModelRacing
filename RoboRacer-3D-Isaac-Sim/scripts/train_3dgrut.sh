#!/usr/bin/env bash
# Train a 3DGRUT (3D Gaussian Ray Tracing, NVIDIA) model on one of our COLMAP
# sessions and export it as a native Isaac Sim ParticleField USD with PPISP
# post-processing -- the same asset type NVIDIA's NuRec demo scenes use.
#
#   source scripts/env.sh
#   bash scripts/train_3dgrut.sh 142014_noflare                # full run, 30k iterations
#   ITERS=500 bash scripts/train_3dgrut.sh 142014_noflare      # smoke test
#   CONFIG=apps/colmap_3dgut_mcmc.yaml bash scripts/train_3dgrut.sh zhuitong
#
# Dataset dirs live in data/local/3dgrut/<name>/{images,sparse/0} (symlinks to
# the session's COLMAP model and the undistorted frames on the T7 drive).
# Runs are written to $RUNS_DIR (default: T7 drive, local disk is nearly full).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: train_3dgrut.sh <dataset-name>}"
DATASET="$ROOT/data/local/3dgrut/$NAME"
RUNS_DIR="${RUNS_DIR:-/media/alienware_felix/T7/roboracer_runs}"
CONFIG="${CONFIG:-apps/colmap_3dgrt_mcmc.yaml}"
ITERS="${ITERS:-30000}"
EXP="${EXP:-${NAME}_$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M)}"
[ -d "$DATASET/sparse/0" ] || { echo "no COLMAP model at $DATASET/sparse/0"; exit 1; }
[ -d "$DATASET/images" ] || { echo "no images at $DATASET/images"; exit 1; }

cd "$ROOT/third_party/3dgrut"
source .venv/bin/activate            # exports CUDA_HOME, TORCH_CUDA_ARCH_LIST, ... persisted by the installer
PY="$ROOT/third_party/3dgrut/.venv/bin/python"   # explicit: on some shells `python` still resolves to /usr/bin
export PYTHONNOUSERSITE=1
mkdir -p "$RUNS_DIR"
echo "== 3DGRUT train: dataset=$DATASET config=$CONFIG iters=$ITERS -> $RUNS_DIR/$EXP"
"$PY" train.py --config-name "$CONFIG" \
    path="$DATASET" out_dir="$RUNS_DIR" experiment_name="$EXP" \
    n_iterations="$ITERS" \
    export_usd.enabled=true export_usd.format=standard \
    export_usd.apply_normalizing_transform=false \
    export_usd.export_cameras=true \
    export_ply.enabled=true \
    "${@:2}"
echo "== done. outputs (for Isaac use build_3dgrut_scene.sh; it re-exports with PPISP folded into SH):"
find "$RUNS_DIR/$EXP" -maxdepth 2 \( -name "*.usd*" -o -name "*.ply" -o -name "*.pt" \) | head
