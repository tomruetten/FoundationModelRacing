#!/usr/bin/env bash
# Create (or repair) the unified `roboracer` conda environment.
#
# Usage:
#   bash scripts/setup_env.sh            # create/update env "roboracer"
#   ENV_NAME=myenv bash scripts/setup_env.sh
#
# Idempotent: re-running only installs what is missing.
set -euo pipefail

ENV_NAME="${ENV_NAME:-roboracer}"
PY_VER="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt24cu124"

# --- locate conda ---------------------------------------------------------
if [ -z "${CONDA_EXE:-}" ]; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then CONDA_BASE="$c"; break; fi
  done
else
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
fi
if [ -z "${CONDA_BASE:-}" ]; then
  echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "== [1/7] conda env '$ENV_NAME' (python $PY_VER)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c conda-forge "python=$PY_VER" pip ffmpeg
else
  echo "   exists, skipping create"
fi
# Keep ~/.local/lib/python3.10/site-packages (pip --user installs) out of this env.
conda env config vars set PYTHONNOUSERSITE=1 -n "$ENV_NAME" >/dev/null
# torch 2.6 defaults torch.load(weights_only=True); nerfstudio 1.1.5 checkpoints
# (ns-eval, ns-viewer, render servers) contain numpy scalars and refuse to load
# without this official escape hatch. Checkpoints here are our own -> trusted.
conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n "$ENV_NAME" >/dev/null
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade "pip<25" setuptools wheel -q

echo "== [2/7] torch 2.6.0 + cu124"
pip install --extra-index-url "$TORCH_INDEX" \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124"

echo "== [3/7] nerfstudio 1.1.5 (camera 3DGS, splatfacto)"
pip install "nerfstudio==1.1.5" "numpy<2"

echo "== [4/7] gsplat 1.5.3 prebuilt wheel (LiDAR 3DGS, replaces nerfstudio's source gsplat)"
pip install --no-deps --index-url "$GSPLAT_INDEX" "gsplat==1.5.3+pt24cu124" \
  || pip install --no-deps "gsplat==1.5.3"   # fallback: source build (slow)

echo "== [5/7] scientific stack + open3d"
pip install "open3d==0.19.0" "pandas==2.0.3" "scipy==1.15.2" "matplotlib==3.10.1" \
  "opencv-python==4.11.0.86" "scikit-learn==1.6.1" pyyaml tqdm wandb pillow casadi usd-core

echo "== [6/7] nnodely 1.5.4 (MS-NN) without its torch==2.2.2 pin"
pip install --no-deps "nnodely==1.5.4"
pip install "numpy<2" onnx onnxruntime reportlab graphviz   # nnodely runtime deps (keep numpy 1.x)
pip install "numpy==1.26.4"   # final pin: every compiled dep (pandas, open3d, sklearn) is built against numpy 1.x

echo "== [7/7] COLMAP 3.x (CPU build from conda-forge, no sudo needed)"
if ! command -v colmap >/dev/null 2>&1; then
  conda install -y -n "$ENV_NAME" -c conda-forge colmap || \
    echo "   WARN: conda colmap install failed; try: sudo apt install colmap"
else
  echo "   colmap already on PATH: $(command -v colmap)"
fi

echo
echo "== verifying"
python "$(dirname "$0")/check_env.py"
echo
echo "Done. Activate with:  conda activate $ENV_NAME"
