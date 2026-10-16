#!/usr/bin/env bash
# End-to-end smoke test that needs no raw data: loads the two deployed
# dynamics checkpoints and runs a few inference steps each.
#
#   conda activate roboracer && bash scripts/smoke_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONNOUSERSITE=1

echo "== GRU Run30 (torch)"
( cd "$ROOT/dynamics/deployed" && python 15_gru_inference.py | tail -3 )

echo "== MS-NN exp9 (nnodely)"
( cd "$ROOT/dynamics/deployed" && python 13_inference_msnn.py 2>&1 | grep -E "step [0-9]|smoke test" )

echo "== render/dynamics server CLIs import cleanly"
( cd "$ROOT/simulation/isaac_sim" \
  && python 01_camera_render_server.py --help >/dev/null 2>&1 \
  && python 03_lidar_render_server.py  --help >/dev/null 2>&1 \
  && python 05_msnn_dynamics_server.py --help >/dev/null 2>&1 \
  && python 08_gru_mocap_dynamics_server.py --help >/dev/null 2>&1 \
  && echo "   ok" )

echo "smoke test passed"
