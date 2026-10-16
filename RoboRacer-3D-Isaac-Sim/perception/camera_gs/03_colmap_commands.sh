#!/bin/bash
# Camera 3DGS — COLMAP Pipeline (nach Entzerrung mit 01_undistort_frames.py)
#
# PROBE_ROOT: Wurzel der rohen Probe-Aufnahmedaten (nicht in diesem Repo
# enthalten, siehe root README) — vor dem Ausführen setzen:
#   export PROBE_ROOT=/pfad/zu/Probe
#
# Voraussetzung: Entzerrte Bilder liegen in colmap/images/
# Vocab-Tree: $PROBE_ROOT/colmap_undistort/colmap/vocab_tree.bin

PROBE_ROOT="${PROBE_ROOT:-./data/Probe}"
EXP="$PROBE_ROOT/experiments/new_recording_20260714"
VOCAB_TREE="$PROBE_ROOT/colmap_undistort/colmap/vocab_tree.bin"
DB="$EXP/colmap/database.db"
IMGS="$EXP/colmap/images"
SPARSE="$EXP/colmap/sparse"

mkdir -p "$SPARSE"

# Schritt 1: Feature-Extraktion (SIFT, feste Intrinsik PINHOLE)
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMGS" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "566.2562,565.9493,641.9587,388.6107" \
    --SiftExtraction.use_gpu 1

# Schritt 2: Sequential Matching MIT Vocab-Tree Loop-Closure
# (wichtig für Rundstrecke — ohne Loop-Closure: offene/fehlerhafte Trajektorie)
colmap sequential_matcher \
    --database_path "$DB" \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path "$VOCAB_TREE" \
    --SiftMatching.use_gpu 1

# Schritt 3: Mapper (inkrementelles Bundle Adjustment)
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMGS" \
    --output_path "$SPARSE"

echo "COLMAP done. Registered frames:"
cat "$SPARSE/0/cameras.txt" | head -3
python3 -c "
import struct, os
p = '$SPARSE/0/images.bin'
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
print(f'  {n} images registered')
" 2>/dev/null || ls "$SPARSE/0/"

# Schritt 4: Konvertierung zu nerfstudio transforms.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda run -n nerfstudio-py310 python3 \
    "$SCRIPT_DIR/02_colmap_to_nerfstudio.py" \
    "$SPARSE/0" \
    "$EXP/dataset"

echo "transforms.json written to $EXP/dataset/"
