#!/usr/bin/env bash
# Isaac Sim 6.1 via pip (NVIDIA index) into conda env "env_isaac6".
# Needed for the native gaussian-splat scene (ParticleField3DGaussianSplat prim);
# Isaac Sim 5.1 (env_isaaclab) can only show the baked point-cloud version.
# ~26 GB on disk, ~15 min download. Driver >= 570 recommended.
set -euo pipefail
ENV_NAME="${ENV_NAME:-env_isaac6}"
VERSION="${ISAACSIM_VERSION:-6.1.0.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1
pip install --upgrade pip
pip install --no-cache-dir "isaacsim[all,extscache]==$VERSION" --extra-index-url https://pypi.nvidia.com
pip install --no-cache-dir numpy scipy pillow matplotlib pandas
conda env config vars set OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 -n "$ENV_NAME"
echo "done. First start takes ~1.5 min (extension cache). Test:"
echo "  conda activate $ENV_NAME && python -c 'from isaacsim import SimulationApp; a=SimulationApp({\"headless\":True}); print(\"ok\"); a.close()'"
