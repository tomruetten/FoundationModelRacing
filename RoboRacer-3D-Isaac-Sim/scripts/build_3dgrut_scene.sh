#!/usr/bin/env bash
# Turn a finished 3DGRUT run into an Isaac Sim scene + matching trajectory.
#   source scripts/env.sh
#   bash scripts/build_3dgrut_scene.sh <run-dir-or-usdz> [session-dir] [meters-per-unit]
# Defaults: session 142014_noflare, 3.9868 m/unit (odometry fit, 2026-09-02).
# Env: DATASET_NAME (data/local/3dgrut/<name>, default 142014_noflare), SH_ITERS (PPISP bake, 3000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:?usage: build_3dgrut_scene.sh <run-dir-or-usdz> [session-dir] [m-per-unit]}"
SESS="${2:-$ROOT/data/local/sessions/session_20260827_142014_noflare}"
MPU="${3:-3.9868}"
if [ -d "$SRC" ]; then
  # Re-export from the latest checkpoint with PPISP folded into the SH coefficients:
  # works for any viewer camera (the runtime SPG variant only post-processes the
  # exported dataset cameras). Needs scripts/patches/3dgrut_sh_bake_pred_features.patch.
  CK=$(ls -v "$SRC"/ours_*/ckpt_*.pt 2>/dev/null | tail -1)
  [ -f "$CK" ] || { echo "no checkpoint under $SRC"; exit 1; }
  USDZ="$(dirname "$CK")/export_ppisp_sh.usdz"
  if [ ! -f "$USDZ" ]; then
    ( cd "$ROOT/third_party/3dgrut" && export PATH="$PWD/.venv/bin:$PATH" PYTHONNOUSERSITE=1 && \
      .venv/bin/python -m threedgrut.export.scripts.export_usd --checkpoint "$CK" \
        --dataset "$ROOT/data/local/3dgrut/${DATASET_NAME:-142014_noflare}" --output "$USDZ" \
        --format standard --no-transform --no-cameras --up-axis z \
        --ppisp-integration-mode sh-optimized --sh-optimization-num-iterations "${SH_ITERS:-3000}" )
  fi
else
  USDZ="$SRC"
fi
[ -f "$USDZ" ] || { echo "no usdz at $USDZ"; exit 1; }
echo "asset: $USDZ"
DP=$(ls "$SESS"/outputs/*/splatfacto/*/dataparser_transforms.json | tail -1)
TRAJ=$(ls "$SESS"/dataset/transforms_colmap_full_ALL*_ALIGNED.json 2>/dev/null | tail -1)
[ -f "$TRAJ" ] || TRAJ="$SESS/dataset/transforms_colmap_full.json"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${GS_ENV:-roboracer}"
python "$ROOT/scripts/wrap_3dgrut_usd.py" --usdz "$USDZ" --dataparser-json "$DP" \
    --meters-per-unit "$MPU" --transforms "$TRAJ" --out "$SESS/scene_3dgrut.usda"
echo "run the demo with:"
echo "  SCENE=$SESS/scene_3dgrut.usda TRANSFORMS=$SESS/scene_3dgrut_trajectory.json bash scripts/run_isaac_demo.sh --gui --nurec --stride 5"
