#!/usr/bin/env bash
# Full model evaluation in a fresh, dedicated conda environment.
#
#   bash scripts/run_evaluation.sh                 # creates env "roboracer-eval", runs everything
#   bash scripts/run_evaluation.sh --quick         # smoke run (subsampled), same env
#   ENV_NAME=roboracer bash scripts/run_evaluation.sh --reuse-env   # use an existing env
#   bash scripts/run_evaluation.sh --only dynamics
#
# Output: outputs/evaluation_<timestamp>/report.md (+ summary.json, CSVs, figures/).
# Requires scripts/env.sh (data paths) and the data linked via scripts/link_data.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-roboracer-eval}"
REUSE=0; PASS=()
for a in "$@"; do
  case "$a" in
    --reuse-env) REUSE=1 ;;
    *) PASS+=("$a") ;;
  esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ "$REUSE" = 0 ] || ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "== creating evaluation environment '$ENV_NAME' (same stack as roboracer + eval extras)"
  ENV_NAME="$ENV_NAME" bash "$ROOT/scripts/setup_env.sh"
  conda activate "$ENV_NAME"
  pip install -q rosbags usd-core
else
  conda activate "$ENV_NAME"
fi
export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
[ -f "$ROOT/scripts/env.sh" ] && source "$ROOT/scripts/env.sh" || echo "warning: scripts/env.sh missing, using defaults under data/local"
bash "$ROOT/scripts/link_data.sh" > /dev/null || true

cd "$ROOT"
echo "== running evaluation in env '$ENV_NAME' ($(python -c 'import torch; print(torch.__version__)'))"
python -m evaluation.run_all "${PASS[@]}"
