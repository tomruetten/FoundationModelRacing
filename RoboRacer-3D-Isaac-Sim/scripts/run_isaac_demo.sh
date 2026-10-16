#!/usr/bin/env bash
# One-command Isaac Sim camera-splat replay demo (V1):
#   1. starts the camera 3DGS render server (conda env roboracer)
#   2. drives the car along the recorded COLMAP trajectory in Isaac Sim
#      (conda env env_isaaclab) and fetches a splat render for every pose
#   3. stops the server
#
# Usage:
#   source scripts/env.sh
#   bash scripts/run_isaac_demo.sh                 # headless, stride 2, ~517 frames
#   bash scripts/run_isaac_demo.sh --gui           # Isaac Sim window (needs a display)
#   bash scripts/run_isaac_demo.sh --gui --stride 20
#   bash scripts/run_isaac_demo.sh --gui --view chase     # low third-person view (default: onboard)
#   bash scripts/run_isaac_demo.sh --gui --opponent       # second F1TENTH (object splat) driving ahead (Isaac + renders)
#   bash scripts/run_isaac_demo.sh --gui --opponent=cone_lowpoly   # any $OBJECTS_DIR/<name>.{npz,usd}
#      OPPONENT_USD/OPPONENT_SPLAT override the files; --opponent-gap M (default 2.5 m ahead) /
#      --opponent-lateral M / --ground-offset H / --opponent-trajectory FILE (GRU-driven path from
#      scripts/simulate_opponent_path.py) / --no-articulation pass through to the client
#
# Env vars (from scripts/env.sh): CAMERA_3DGS_CONFIG, NEW1_ROOT.
# Override: TRANSFORMS=..., SCENE=..., OUT_DIR=...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${CAMERA_3DGS_CONFIG:?source scripts/env.sh first}"
: "${NEW1_ROOT:?source scripts/env.sh first}"
TRANSFORMS="${TRANSFORMS:-$NEW1_ROOT/dataset/transforms.json}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/isaac_demo_$(date +%Y%m%d_%H%M%S)}"
QUEUE="/tmp/isaac_queue/camera_$$"
GS_ENV="${GS_ENV:-roboracer}"
OBJECTS_DIR="${OBJECTS_DIR:-$ROOT/data/local/objects}"
CLIENT_ARGS=()
for a in "$@"; do
  case "$a" in
    --opponent|--opponent=*)
      NAME="${a#--opponent}"; NAME="${NAME#=}"; NAME="${NAME:-f1tenth}"
      OPPONENT_USD="${OPPONENT_USD:-$OBJECTS_DIR/$NAME.usd}"
      OPPONENT_SPLAT="${OPPONENT_SPLAT:-$OBJECTS_DIR/$NAME.npz}"
      [ -f "$OPPONENT_USD" ] && CLIENT_ARGS+=(--opponent-usd "$OPPONENT_USD") || echo "warning: $OPPONENT_USD missing (scripts/splat_to_usd_gsplat.py --object-npz), Isaac viewport has no opponent"
      [ -f "$OPPONENT_SPLAT" ] && CLIENT_ARGS+=(--opponent-splat "$OPPONENT_SPLAT") || echo "warning: $OPPONENT_SPLAT missing (scripts/import_object_splat.py / extract_object_splat.py)"
      ;;
    *) CLIENT_ARGS+=("$a") ;;
  esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
# Isaac Sim env + scene: prefer Isaac Sim 6 (env_isaac6) with the native
# gaussian-splat scene; fall back to Isaac Sim 5.1 (env_isaaclab) with the
# baked point-cloud scene. Override with ISAAC_ENV= / SCENE=.
if [ -z "${ISAAC_ENV:-}" ]; then
  if conda env list | awk '{print $1}' | grep -qx env_isaac6; then ISAAC_ENV=env_isaac6; else ISAAC_ENV=env_isaaclab; fi
fi
if [ -z "${SCENE:-}" ]; then
  if [ "$ISAAC_ENV" = env_isaac6 ] && [ -f "${ISAAC_SCENE_GSPLAT:-/nonexistent}" ]; then
    SCENE="$ISAAC_SCENE_GSPLAT"
  else
    SCENE="${ISAAC_SCENE_POINTS:-$NEW1_ROOT/scene_splat_points.usda}"
  fi
fi
echo "== Isaac env: $ISAAC_ENV   scene: $SCENE"
mkdir -p "$QUEUE" "$OUT_DIR"
cd "$ROOT/simulation/isaac_sim"

echo "== [1/3] camera render server ($GS_ENV) on $CAMERA_3DGS_CONFIG"
echo "   log: $OUT_DIR/render_server.log"
conda run --no-capture-output -n "$GS_ENV" python -u 01_camera_render_server.py \
    --queue-dir "$QUEUE" --config "$CAMERA_3DGS_CONFIG" > "$OUT_DIR/render_server.log" 2>&1 &
SERVER_PID=$!
trap 'touch "$QUEUE/STOP"; sleep 2; kill $SERVER_PID ${VIEWER_PID:-} 2>/dev/null || true; rm -rf "$QUEUE"' EXIT
for i in $(seq 1 240); do
  grep -q "watching" "$OUT_DIR/render_server.log" 2>/dev/null && break
  if grep -q "Traceback" "$OUT_DIR/render_server.log" 2>/dev/null; then
    echo "render server failed, see $OUT_DIR/render_server.log"; tail -20 "$OUT_DIR/render_server.log"; exit 1
  fi
  if (( i % 15 == 0 )); then
    echo "   still loading ($((i*2))s) -- last log line: $(tail -n 1 "$OUT_DIR/render_server.log" | cut -c1-100)"
  fi
  sleep 2
done
echo "   server ready after ~$((i*2))s"

VIEWER_PID=""
if [[ " $* " == *" --gui "* ]]; then
  echo "== [2/3] live viewer for the 3DGS renders (close with q)"
  mkdir -p "$OUT_DIR/frames"
  conda run --no-capture-output -n "$GS_ENV" python -u "$ROOT/scripts/live_frame_viewer.py" \
      --watch-dir "$OUT_DIR/frames" --idle-exit 120 > /dev/null 2>&1 &
  VIEWER_PID=$!
fi
echo "== [2/3] Isaac Sim client ($ISAAC_ENV): $TRANSFORMS"
OMNI_KIT_ACCEPT_EULA=yes conda run --no-capture-output -n "$ISAAC_ENV" python -u 02_camera_splat_client_isaac.py \
    --transforms "$TRANSFORMS" --scene "$SCENE" \
    --queue-dir "$QUEUE" --output-dir "$OUT_DIR/frames" "${CLIENT_ARGS[@]}" 2>&1 | tee "$OUT_DIR/isaac_client.log" \
    | grep -E "^\s*\[[ 0-9]+/|isaac\]|trajectory\]|✅|Traceback|Error\]" || true

echo "== [3/3] done: $(ls "$OUT_DIR/frames" 2>/dev/null | wc -l) frames in $OUT_DIR/frames"
