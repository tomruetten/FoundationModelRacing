# Miniature Porsche and changing hall

The five-second demo inserts a silver/black Porsche 911 Carrera 4S (46 cm long,
20.7 cm wide, 13.2 cm tall) alongside the F1TENTH opponent. Its wheels rotate with
travel distance. At 1.5 seconds, nine pallet/crate instances appear in the hall.
The learned camera and LiDAR receive the same object placements; the two LiDAR
panels show the same synthetic scan in 3D and BEV. No scene retraining is needed.

[4K video with high-resolution obstacles](../outputs/porsche_hall_demo_hd/porsche_hall_change_5s.mp4) ·
[Earlier 1080p version](../outputs/porsche_hall_demo/porsche_hall_change_5s.mp4)

The default now uses `_hd` obstacle assets, bilinear per-Gaussian texture sampling,
and a 2560 × 1600 camera render inside a 3840 × 2160 video. The camera image is
rendered at twice the original dimensions; it is not a 720p image enlarged to 4K.
The existing 8K wooden textures are retained. The plastic crate uses Poly Haven's
original 4K glTF/material because the BlenderKit GLB's baked colour texture was
entirely black. The cone retains its scanned 4K texture with finer sampling.

| Obstacle | Earlier Gaussians | High-resolution Gaussians | New spacing |
|---|---:|---:|---:|
| Wooden pallet | 146,680 | 914,254 | 2 mm |
| Wooden crate | 420,072 | 2,625,393 | 2 mm |
| Plastic crate | 179,660 | 718,610 | 2 mm |
| Photo-scanned cone | 71,764 | 287,095 | 1 mm |

These variants preserve object dimensions and increase surface sampling density;
they do not invent additional mesh geometry. Denser assets and larger camera
renders cost more memory and render time. LiDAR retains recorded ray sampling.

## Build and run

```bash
conda activate roboracer
python -m pip install DracoPy  # glTF Draco decoder, if not installed
python scripts/download_porsche_hall_assets.py
PYTHONNOUSERSITE=1 python scripts/build_textured_demo_assets.py
PYTHONNOUSERSITE=1 python scripts/render_porsche_hall_demo.py --preview
PYTHONNOUSERSITE=1 python scripts/render_porsche_hall_demo.py
PYTHONNOUSERSITE=1 python scripts/render_porsche_hall_demo.py --benchmark-only
```

The renderer needs New1's local camera/LiDAR checkpoints and floor plane, the
existing F1TENTH/cone assets, and ray-template NPZs from `render_livox_demo.py`.
Pass `--templates /path/to/scans` to select those files. It uses the current local
F1TENTH asset; it does not overwrite the default opponent. Download URLs, licences
and expected SHA-256 hashes are in `configs/porsche_hall_assets.json`.
Original plastic-crate dependencies are pinned in
`configs/plastic_crate_original_files.json`. To rebuild only the obstacles, use
`python scripts/build_textured_demo_assets.py --asset obstacles --quality high`.
To render the earlier density/resolution, pass
`--obstacle-quality standard --resolution-scale 1 --out outputs/my_standard_demo`.
Full-resolution camera stills are saved as `matched_before_native.png` and
`matched_after_native.png` alongside the video.

For Isaac, export the object and select it as the opponent:

```bash
conda activate env_isaac6
OMNI_KIT_ACCEPT_EULA=yes python scripts/splat_to_usd_gsplat.py \
  --object-npz data/local/objects/porsche_911.npz \
  --out-usd data/local/objects/porsche_911.usd --opacity-thresh 0
bash scripts/run_isaac_demo.sh --gui --opponent=porsche_911
```

Repeat the export for `wooden_pallet_hd`, `wooden_crate_hd`, `plastic_crate_hd` and
`cone_photoscan_hd` if placing them in Isaac. These local USD exports are generated
with the high-resolution demo assets; the earlier unsuffixed versions are retained.
Modify the `props` list in `render_porsche_hall_demo.py` to change their positions:
`(asset_name, distance_along_path_m, lateral_offset_m, stack_height_m)`. Placements
are computed once relative to the starting path; they remain stationary.

## What the demo establishes

- Runtime insertion of a new opponent and static scene objects in both sensors.
- Texture-preserving Gaussian conversion and shared geometry/occlusion.
- CUDA-synchronised camera and LiDAR render timings in `benchmark.json` and `metrics.json`, including
  a paired same-pose comparison with/without hall props. Timings include model
  rendering, object compositing and transfers, but exclude image/video encoding.
  The first of four repeats is discarded. These are short local measurements,
  not a sustained real-time benchmark or an improvement over an older renderer.

The earlier standard-density, 1280 × 800 camera run on an RTX 4090 measured
these medians at the same pose (these numbers do not describe the new HD mode):

| Scene | Camera render | Livox-pattern LiDAR render |
|---|---:|---:|
| Porsche + F1TENTH + four cones | 23.5 ms | 81.3 ms |
| Same opponents/cones + nine hall props | 33.6 ms | 151.8 ms |

These are individual render latencies, not combined sensor-loop throughput.
The exported clip passed full decoding and 150-frame/five-second checks. Saved
scans passed finite/range/direction checks, static props remained fixed, and wheel
rotation preserved each axle centre. Preview artifacts are stored in `preview/`
to keep them separate from full-run scans.

The movie is encoded at 30 fps; that is playback speed. Motion follows a scripted
trajectory with rotating wheels, not closed-loop racing or reactive obstacle
avoidance. Gaussian surface colours include baked lighting; metallic reflections
and glass are approximations, not a physically correct moving PBR renderer.

LiDAR uses New1's LiDAR-specific checkpoint plus the object splats. Angular
sampling comes from recorded August Livox return directions, not New1 raw truth or
a complete firing pattern. Known New1 geometry artifacts remain. Scans use an
8 m radial limit, no accumulation and no intra-scan motion distortion. The scene
switch triggers an immediate fresh scan; other scans follow template timing.

## Asset credits and changes

| Asset | Creator/source | Licence |
|---|---|---|
| [Porsche 911 Carrera 4S](https://sketchfab.com/3d-models/free-porsche-911-carrera-4s-d01b254483794de3819786d93e0e1ebf) | Lionsharp Studios; GLB distributed with PlayCanvas's Web Components examples | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Scan Wooden Pallet](https://www.blenderkit.com/get-blenderkit/3822ba51-e0b0-4773-a209-9ab316d7129c/) | Free poly / River Yang | CC0 |
| [Wooden Crate 02](https://polyhaven.com/a/wooden_crate_02) | Poly Haven; James Ray Cock and Jurita Burger | CC0 |
| [Plastic Crate 02](https://www.blenderkit.com/get-blenderkit/e58013cc-d93f-44fe-b775-636dc90500c1/) | Poly Haven | CC0 |

All assets are converted from meshes to Gaussian surfaces using per-sample UV
colours, smooth normals and baked contact/diffuse lighting. They are not
photo-trained Gaussian captures. The Porsche is rescaled to 46 cm, its duplicate
clearcoat surface and original display ground are removed, wheel groups are
articulated, and a visual contact shadow is added in the demo. Attribution is
retained here and in the movie. Original GLBs are kept under
`data/local/objects/sources/`; generated models/videos are local ignored data.

## HD timing check

After export and encoding finished, the 2560 × 1600 / HD-assets run measured
48.5 ms camera and 106.8 ms LiDAR without hall props,
and 92.4 ms camera and 464.7 ms LiDAR with all nine props.
These are median individual render latencies on the RTX 4090 using the same
short paired benchmark; they are not combined real-time sensor throughput.
