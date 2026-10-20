# track_synth: synthetic track from an occupancy map

Turns a ROS map (PNG + YAML, e.g. `reference/iros2026/`) into an Isaac Sim USD scene with two-layer
white corrugated hoses (30 cm Ø) as the only track boundary, in a placeholder industrial hall, and
collects ego camera frames with exact poses in the format `Code/benchmark` reads.

```bash
# 0. Real-hall backdrop from the camera 3DGS (roboracer env, GPU), once
python -m track_synth.splat_panorama --config data/local/New1/checkpoints/camera_3dgs/config.yml \
    --out reference/iros2026/hall_pano.png
#    (floor texture reference/iros2026/floor_real.jpg is already in git; rebuild it with
#     python -m track_synth.real_textures --frames <ego frames dir> --out reference/iros2026/floor_real.jpg)

# 1. Build scene (plain Python; needs usd-core, skan, opencv-python-headless, pillow, scipy)
python -m track_synth.build --map reference/iros2026/IROS2026.yaml --out outputs/iros2026 \
    --hdri reference/iros2026/hall_pano.png
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
- Hose: 30 cm diameter, soft (flattened to 0.72, upper layer 0.85 of that, irregular width, pinched
  stretches), brown straps every ~1.8 m, two layers everywhere by default. The real boundary is mostly
  single-layer: `--upper-fraction 0.5` at build time stacks only half of the wall (random stretches
  that taper out). No cones, no floor markings.
- Lighting (with `--hdri`): one dome (the panorama, visible background + image-based light) plus two
  invisible overhead DistantLights. `collect.py --dome-scale` changes the panorama only, `--fill-scale`
  the distant lights only (hoses/floor), `--light-scale` all of them. A second DomeLight must not be
  used: RTX picks a single environment and the background turned black.
- The GRU was trained on other surfaces/tracks; poses are exact simulator states, not real-car predictions.
- `collect.py` has not been run in Isaac Sim yet (no GPU where it was written).
