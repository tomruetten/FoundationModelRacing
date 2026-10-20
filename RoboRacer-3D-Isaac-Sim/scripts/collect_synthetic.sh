#!/usr/bin/env bash
# Run track_synth.collect inside the Isaac Sim 6 python (same environment as run_racing.sh).
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
isaac_python="${RACING_ISAAC_PYTHON:-$HOME/miniconda3/envs/env_isaac6/bin/python}"
exec "$isaac_python" -m track_synth.collect "$@"
