#!/bin/bash
# Camera 3DGS — Training + Evaluierung (nach COLMAP mit 03_colmap_commands.sh)
# Ergebnis: PSNR 27.61 dB | SSIM 0.885 | LPIPS 0.228
#
# PROBE_ROOT: Wurzel der rohen Probe-Aufnahmedaten (nicht in diesem Repo
# enthalten, siehe root README) — vor dem Ausführen setzen:
#   export PROBE_ROOT=/pfad/zu/Probe

PROBE_ROOT="${PROBE_ROOT:-./data/Probe}"
EXP="$PROBE_ROOT/experiments/new_recording_20260714"

# Training: 50.000 Iterationen splatfacto
# (30k reicht nicht — Szene braucht 50k für volle Konvergenz)
conda run -n nerfstudio-py310 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ns-train splatfacto \
        --data "$EXP/dataset" \
        --output-dir "$EXP/outputs/camera_base" \
        --pipeline.model.use-scale-regularization True \
        --max-num-iterations 50000 \
        --vis tensorboard

# Config-Pfad automatisch ermitteln
CONFIG=$(find "$EXP/outputs/camera_base" -name "config.yml" | sort | tail -1)
echo "Config: $CONFIG"

# Evaluierung
conda run -n nerfstudio-py310 ns-eval \
    --load-config "$CONFIG" \
    --output-path "$EXP/outputs/camera_base/eval_results.json"

python3 -c "
import json
d = json.load(open('$EXP/outputs/camera_base/eval_results.json'))
r = d.get('results', d)
print(f\"PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}\")
"

# Viewer starten (optional)
# conda run -n nerfstudio-py310 ns-viewer --load-config "$CONFIG"
