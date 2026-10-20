# track_synth: synthetic track from an occupancy map

Turns a ROS map (PNG + YAML, e.g. `reference/iros2026/`) into an Isaac Sim USD scene with two-layer
white corrugated hoses (30 cm Ø) as the only track boundary, in a placeholder industrial hall, and
collects ego camera frames with exact poses in the format `Code/benchmark` reads.

```bash
# 1. Build scene (plain Python; needs usd-core, skan, opencv-python-headless, pillow, scipy)
python -m track_synth.build --map reference/iros2026/IROS2026.yaml --out outputs/iros2026
python -m track_synth.preview outputs/iros2026     # software sanity renders, not photoreal

# 2. Collect data (Isaac Sim 6 python, GPU)
PYTHONPATH=. python -m track_synth.collect --scene outputs/iros2026 \
    --dynamics-checkpoint dynamics/deployed/trained_models/gru_run30_best.pth \
    --episodes 20 --seconds 25 --output outputs/iros2026_data
```

Pipeline: wall lines of the map are skeletonised -> hose centre lines; free band -> medial-axis
centre line (closed, 0.1 m) and collision polygons. Each episode starts at a random point on the
centre line with lateral/heading offset and a random speed; `RaceEngine` autopilot follows the
centre line with the GRU (or `--kinematic`) dynamics. Camera = the OAK-D intrinsics and mount from
`configs/racing_142014.json`.

## Assumptions to check
- Map wall lines are taken as hose **centres**. The free width is ~2.2 m between the drawn lines,
  so about 1.9 m between hose surfaces.
- Hose: 30 cm diameter, vertically flattened (0.82), two layers; no cones, no floor markings.
- The hall (walls, beams, clutter, lights) is generated geometry, not a scan. Light intensities are
  uncalibrated: check the first frames and adjust `build_hall` / `--subframes`.
- The GRU was trained on other surfaces/tracks; poses are exact simulator states, not real-car predictions.
- `collect.py` has not been run in Isaac Sim yet (no GPU where it was written).
