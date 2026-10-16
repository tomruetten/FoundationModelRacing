# Opponent vehicles as Gaussian splats

An opponent (or any obstacle) is its own small Gaussian splat in an object frame. At render
time its gaussians are inserted into **both** scene maps, so the pseudo camera image and the
pseudo LiDAR scan see the same object at the same pose, and it is exported once more as a
gaussian-splat USD so the Isaac Sim viewport shows it too.

```
object splat (npz)  ──►  camera map (splatfacto)      pseudo image   (01_camera_render_server.py)
  metres, z up,     ──►  LiDAR map  (raw gsplat)      pseudo scan    (03_lidar_render_server.py)
  +x forward        ──►  cone.usd  (ParticleField)    Isaac viewport (/World/Opponent, 02/04 clients)
```

Nothing in the maps is modified. The camera server concatenates the posed object gaussians in
front of the rasteriser and restores the map afterwards (occlusion both ways comes out of the
rasteriser: an opponent behind a real cone is hidden, an opponent in front hides the track). The
LiDAR renderer renders the objects as a separate layer and composites by first surface: a ray
takes the object where the object layer is solid (alpha >= 0.9) and nearer than the map surface.
Rendering objects inside the LiDAR map would mix their depth with the map's low-opacity fog
through gsplat's expected depth (13 cm bias measured on the F1TENTH 2 m ahead). Legacy conversion timing with the
168k-gaussian car plus a 131k cone: 13 to 26 ms per camera frame, ~30 ms per LiDAR scan.

## Why this needs more than concatenating gaussians

* The camera map stores view-dependent colour as spherical harmonics (SH, degree 3) in the frame
  the gaussians live in. Rotating gaussians without rotating their SH coefficients gives colour
  casts (seen with the whole New1 scene). `object_splats.sh_rotation()` builds the exact 16x16
  real-SH rotation matrix for gsplat's basis (block diagonal per degree), and `ObjectSplat.posed()`
  rotates means, covariances (quaternions) and SH together, with an optional uniform scale
  (nerfstudio units vs metres).
* The LiDAR map has no view dependence but different channels (intensity, raydrop logit).
  Object gaussians get a luminance intensity, a never-dropped raydrop logit and an opacity of
  at least 0.95 (solid surface).

Checks (`tests/test_object_splats.py`, plus the renders in `outputs/opponent_checks/`):

| check | result |
| --- | --- |
| SH rotation vs analytic evaluation at rotated directions, degrees 0 to 3 | exact (1e-9) |
| cone cut out of New1, re-inserted at its source pose, rendered from a training view | 0.02/255 mean abs difference to the untouched scene |
| object preview vs preview of the object rotated 70 deg and scaled 2x | identical (0.0/255) |
| camera map with the F1TENTH splat 2.5 m ahead and a cone 4 m ahead (frames 150/600/900) | both rendered with correct occlusion, 0.4 to 0.9 % of pixels change |
| LiDAR map with the F1TENTH splat 2.2 m ahead (frames 150/600/900) | 297 to 611 returns on the car, median distance to the car surface 1.4 cm |
| mesh-sampled car, interior accumulated alpha (camera, side view) | 0.997 |
| Isaac viewport vs render-server frame, F1TENTH opponent at frame 900 | same car, same place |
| earlier: scene-cut cone 2.3 m ahead, objects rendered inside the LiDAR map | 295 returns, 8 cm radial error (expected-depth mixing, hence the first-surface compositing above) |

## Workflow

### 1. Get an object splat

Three sources feed the same container (`ObjectSplat`, npz). All objects live in
`$OBJECTS_DIR` (`data/local/objects`, git-ignored); `run_isaac_demo.sh --opponent=<name>` picks
`<name>.npz` (servers) and `<name>.usd` (Isaac) from there.

| name | source | gaussians | how |
| --- | --- | --- | --- |
| `f1tenth` / `f1tenth_detailed` (default local opponent) | AutoDRIVE CAD with reference-informed materials, smooth normals, baked occlusion, cables, stereo camera, and open rear support | 633k | `build_f1tenth_appearance.py` |
| `f1tenth_legacy` | Original flat-color AutoDRIVE conversion | 168k | `import_object_splat.py --usd` |
| `cone_lowpoly` | CC0 low-poly traffic cone mesh by Quaternius ([poly.pizza](https://poly.pizza/m/lAx8JytxGD)), scaled to 0.33 m | 131k | `import_object_splat.py --mesh` |
| `cone_photoscan` | CC0 **Traffic Cone (Photoscanned)** by Nik Kottmann ([BlenderKit](https://www.blenderkit.com/get-blenderkit/6a3dbebd-990f-4495-b023-cdcfe0b68431/)), captured texture with wear, scaled to 0.33 m | 72k | textured GLB converted with `import_object_splat.py --mesh` |
| `cone` | cut out of the trained New1 scene (seen from the track side only, blurry up close) | 1k | `extract_object_splat.py` |

The five-second demo uses `cone_photoscan` instead of `cone_lowpoly`. This is a
photograph-derived mesh converted to surface Gaussians, not a pretrained Gaussian
capture. Source GLB and CC0 metadata are in `data/local/objects/sources/`; the new
`.npz` and native Isaac `.usd` assets are in `data/local/objects/`. Draco-compressed
GLB import requires `DracoPy`. Reproduce the synchronized two-car/four-cone video
with `python scripts/render_learned_demo.py` in the `roboracer` environment;
output and provenance are under `outputs/learned_demo_5s_detailed/`. This is learned-sensor
replay along recorded poses, not a learned-dynamics rollout.

**A. Mesh assets (USD / GLB / OBJ) -> `scripts/import_object_splat.py`.** Surface sampling in
the mesh2splat / mesh2gaussian style, vectorised: every face gets `overlap * area / spacing^2`
flat gaussians (stochastic rounding, so a CAD mesh with a million tiny faces does not force one
gaussian per face), oriented in the face plane along the longest edge, `spacing / 4` thick,
opacity 0.99, constant colour from the bound material (USD: OmniPBR / UsdPreviewSurface
constants, glTF: baseColorFactor or texture, vertex colours). Measured on the car at 4 mm
spacing: interior accumulated alpha 0.997, so the surfaces are solid without any tuning
(`--footprint` scales the discs if a coarser mesh shows gaps).

```bash
conda activate roboracer && source scripts/env.sh
python scripts/import_object_splat.py \
    --usd "$DATA_DRIVE_ROOT/reference_repos/AutoDRIVE_F1TENTH_isaac_test/F1TENTH.usd" \
    --up Y --forward X --spacing-mm 4 --name f1tenth --out $OBJECTS_DIR/f1tenth.npz --preview
python scripts/import_object_splat.py --mesh $OBJECTS_DIR/sources/traffic_cone_quaternius.glb \
    --up Y --forward Z --height 0.33 --spacing-mm 3 --name cone_lowpoly --out $OBJECTS_DIR/cone_lowpoly.npz --preview
```

`--up/--forward` name the asset's own axes (the F1TENTH USD is Y up with the front at +X: front
wheels at x = +8.9 cm, left wheels at z = -13.8 cm). USD units come from `metersPerUnit`
(0.01 here); `--height` rescales a unit-less mesh to a real height. Colour space: glTF constants
are linear and get the sRGB transfer; the USD constants were copied from Unity materials and are
already sRGB (`--linear-colors` overrides either way). 7 s for the 937k-triangle car.

**B. Ready-made Gaussian splats (PLY) -> `import_object_splat.py --ply`.** Standard 3DGS PLY as
written by Scaniverse, Polycam, Postshot, SuperSplat, nerfstudio `ns-export gaussian-splat`,
3DGRUT. Same frame arguments plus `--height`, `--floor-percentile` (ground = that percentile of
the heights, floaters below are cut) and `--opacity-thresh`. Searched on 2026-09-10: SuperSplat
gallery, Sketchfab (`gaussiansplats` tag), SplatMart, Polycam and Scaniverse galleries hold no
traffic cone or RC car as a splat, hence the mesh route above. Downloaded assets go to
`$OBJECTS_DIR/sources/` with their licence noted in the object's `.json`.

**C. Cut out of a trained scene -> `scripts/extract_object_splat.py`** (the `cone` stand-in):

```bash
CK=$NEW1_ROOT/checkpoints/camera_3dgs
python scripts/extract_object_splat.py --ckpt $CK/nerfstudio_models/step-000049999.ckpt \
    --dataparser-json $CK/dataparser_transforms.json --transforms $NEW1_ROOT/dataset/transforms.json \
    --list-clusters                      # candidate objects: compact opaque clusters standing on the floor
python scripts/extract_object_splat.py --ckpt ... --dataparser-json ... --transforms ... \
    --center -2.63 0.12 -0.40 --size 0.36 0.36 0.44 --auto-floor --name cone --out $OBJECTS_DIR/cone.npz --preview
```

`--center/--size` is a box in the metric world frame of `transforms.json`, `--auto-floor`
estimates the ground height around the box, `--yaw-deg` is the heading of the object's forward
axis in that frame (so the saved object has +x forward).

**D. The real opponent: capture the second F1TENTH as its own scene once** (best realism).

1. Put the car on the hall floor in an open area (2 m free around it), plenty of light, no
   direct flare. Record a slow walk-around with the OAK-D or a phone: three loops at knee,
   hip and shoulder height, 5 to 8 s per loop, camera always pointing at the car, 1 to 1.5 m
   distance. 300 to 500 frames after extraction. Sharp frames only (drop motion blur).
   Alternatively use a phone app that writes a splat PLY directly (Scaniverse is free and
   on-device) and go to B.
2. Reconstruct with COLMAP and train exactly like a track scene
   (`ns-process-data images` or the existing COLMAP pipeline, then `ns-train splatfacto`),
   with background masks (rembg / SAM2; nerfstudio reads a `mask_path` per frame) if the
   object should come out clean. The floor texture and the surroundings give COLMAP its
   features; the car alone is too glossy. A 3DGRUT training works the same way
   (`scripts/train_3dgrut.sh`); import its exported PLY via B.
3. Scale: COLMAP is scale free. Either include a known object (the 0.32 m wheelbase of the
   car itself, a printed checkerboard) or pass `--height` (0.24 m for the F1TENTH with mast).
4. Extract with `--list-clusters` / `--center --size --yaw-deg --auto-floor` (C) or import the
   exported PLY (B), then clean floaters in [SuperSplat](https://superspl.at/editor) if needed.
   Check the preview from all four sides and, for C, the re-insertion test (render the source
   scene with and without the object at its source pose; the difference must be near zero).

**Export for Isaac** (any source): `conda activate env_isaac6 && python scripts/splat_to_usd_gsplat.py
--object-npz $OBJECTS_DIR/<name>.npz --out-usd $OBJECTS_DIR/<name>.usd --opacity-thresh 0`.

### 2. Render with the object

Servers accept object splats in every request; the Isaac clients send them automatically.

```bash
# request_N.json (camera or LiDAR server): T_wo is object -> world (same frame as the pose)
{"c2w": [[...]], "objects": [{"path": "/abs/path/cone.npz", "T_wo": [[...4x4...]]}]}
{"R_wl": [[...]], "t_wl": [...], "objects": [{"name": "opponent", "T_wo": [[...]]}]}
# names come from --object NAME=PATH.npz at server start; paths load on first use
python simulation/isaac_sim/01_camera_render_server.py --config ... --object opponent=/abs/opponent.npz
```

In Python: `renderer.render(c2w, objects=[(ObjectSplat.load(p), T_wo)])` and
`lidar.render(R_wl, t_wl, alpha_thresh=0.99, objects=[...])`.

### 3. Isaac Sim demo with an opponent

```bash
source scripts/env.sh
bash scripts/run_isaac_demo.sh --gui --opponent                 # second F1TENTH, 2.5 m ahead along the path
bash scripts/run_isaac_demo.sh --gui --opponent=cone_lowpoly --opponent-gap 4 --opponent-lateral 0.3
OPPONENT_USD=.../x.usd OPPONENT_SPLAT=.../x.npz bash scripts/run_isaac_demo.sh --gui --opponent
```

The opponent drives the recorded trajectory `--opponent-gap` metres ahead of the ego car
(path distance, so it does not collapse onto the ego when the car is slow; `--opponent-lead N`
uses a fixed frame offset instead). Its pose is derived from the trajectory: heading from the
camera forward axis; origin and "up" from the scene's floor plane (`floor_plane.json`, see
"Standing on the floor") when present, otherwise `--ground-offset` (0.15 m, the measured New1
camera height) below the camera along the best-fit plane of the trajectory, because COLMAP
frames are not gravity aligned (New1 is tilted 9 deg). The same pose goes to Isaac
(`/World/Opponent`, a second `ParticleField3DGaussianSplat` prim) and to the render server
(converted into the server's frame for 3DGRUT trajectories via `transform_matrix_source`).

`04_lidar_splat_client_isaac.py --opponent-splat ... --opponent-usd ...` does the same for the
combined camera + LiDAR replay.

## Standing on the floor

`scripts/build_floor_plane.py` fits the reconstructed floor of a scene as a plane (robust fit
to the modal height of opaque gaussians in 10 cm cells along the camera corridor) and writes
`floor_plane.json` next to `transforms.json`. Clients, the demo launcher and
`render_learned_demo.py` pick it up automatically; `pose_from_camera_frame(..., floor=plane)`
projects the camera position onto the plane and uses the plane normal as "up".

New1 numbers: 2131 of 2601 floor cells within 3 cm of the plane, inlier RMS 0.9 cm, plane tilted
8.9 deg against world z, camera 0.150 +- 0.004 m above it along the whole lap.

Why objects still looked as if they drove through the floor: the floor of a splat is not a
surface but a band of opaque gaussians. In New1 their centres spread from -6 to +7.5 cm around
the plane, and an object placed exactly on the plane loses its lowest centimetres in that band
(wheels look sunk). Culling the above-plane part of the band is not an option: it changes the
floor rendering by 8.8/255 mean absolute (PSNR 24.5 dB against the original renders), so those
gaussians carry the floor texture. Instead the plane carries a visual `clearance_m` (default
2 cm, `--clearance`): in a 0 to 4 cm sweep, 0 cm swallows the wheels, 2 cm shows them resting on
the floor, 3 to 4 cm starts to float. For a new scene: run `build_floor_plane.py`, check the
printed camera height (should be constant), then `flatten_floor.py` (next section) and rebuild the
plane with `--clearance 0.005`.

## A flat floor: fine-tuning the maps with the plane as a prior

The 2 cm clearance hides the problem; `scripts/flatten_floor.py` removes it. What did not work:
the accumulated New1 LiDAR cloud as depth supervision (after ICP it still disagrees with the
camera map by 21 cm median, and its own floor is +-7 cm thick); an expected-depth loss alone
(the band is symmetric around the plane, so the expected depth already sits on it; the band did
not move and PSNR fell 1.7 dB); projecting the whole +-10 cm band onto the plane (the part below
the plane carries the floor's reflections, PSNR fell 1.2 dB). What works:

1. project only the gaussians **above** the plane (h > 1 cm, within 1.5 m of the camera path,
   38.5k gaussians) down to 0.5 cm, and hold them there after every optimiser step,
2. fine-tune 6000 steps (3 min) with splatfacto's photometric loss plus a small floor-depth loss,
   at a quarter of the initial learning rates,
3. hard-flatten the LiDAR map's floor onto its own plane (appearance is irrelevant there).

New1 result: held-out PSNR 27.40 -> 27.20 dB, floor band p95 +7.5 -> +0.6 cm (p5 stays at
-6 cm, the reflection layer), map renders change by 2.7 to 3.6/255 mean absolute, cars stand on
the floor at 0.5 cm clearance without sinking. The flattened maps are the defaults in
`scripts/env.sh` (`checkpoints/camera_3dgs_flat`, `lidar_3dgs/best_flat.pt`,
`scene_splat_gsplat_flat.usd`); the originals stay next to them.

```bash
python scripts/flatten_floor.py --config $NEW1_ROOT/checkpoints/camera_3dgs/config.yml \
    --transforms $NEW1_ROOT/dataset/transforms.json --floor-plane $NEW1_ROOT/dataset/floor_plane.json \
    --images $DATA_DRIVE_ROOT/New1/dataset/images_real --camera-params $NEW1_ROOT/camera_params.json \
    --out $NEW1_ROOT/checkpoints/camera_3dgs_flat --lidar-ckpt $NEW1_ROOT/checkpoints/lidar_3dgs/best.pt \
    --steps 6000 --depth-weight 1
python scripts/build_floor_plane.py --ckpt .../camera_3dgs_flat/nerfstudio_models/step-000049999.ckpt \
    --dataparser-json .../camera_3dgs_flat/dataparser_transforms.json --transforms ... --clearance 0.005
python scripts/splat_to_usd_gsplat.py --ckpt .../camera_3dgs_flat/... --to-world --out-usd $NEW1_ROOT/scene_splat_gsplat_flat.usd   # env_isaac6
```

## Moving like a car: wheels, steering, attitude, contact shadow

Car assets carry **parts**: `import_object_splat.py --usd ... --parts f1tenth --shadow` and
`build_f1tenth_appearance.py` tag every gaussian by mesh group (`Wheel_1..4` and the hub/knuckle
solids -> `wheel_fr/rr/fl/rl`, everything else `body`) and store the axle centre, radius
(5.7 cm) and steerability of each wheel in the object's json (`parts`). `ObjectSplat.articulated(
wheel_angle, steer_angle, pitch, roll)` returns per-part rigid transforms; `posed()` applies them
before the world pose and rotates the SH coefficients of every part along, so spinning wheels keep
their view-dependent colour.

* wheel spin: angle = distance travelled / 0.057 m
* steering: Ackermann angle from the path curvature, `atan(wheelbase * kappa)`, front wheels only
* attitude: pitch 1.2 deg per m/s^2 of braking (nose down positive), roll 1.5 deg per m/s^2 of
  lateral acceleration (outside down), clipped at 4 deg (`attitude_from_accel`)
* contact shadow: a soft dark patch (`contact_shadow`, 1.1k flat gaussians, opacity 0.55 at the
  centre fading to the edge) as a part flagged `lidar: false`, so the camera sees a grounded car
  and the LiDAR layer ignores it

Servers take `"articulation": {"wheel_angle", "steer_angle", "pitch", "roll"}` per object in the
request; the Isaac clients and `render_learned_demo.py` compute it from the driving state
(`driving_state_from_path` for path followers, the trajectory file for the GRU opponent).
`--no-articulation` keeps the object rigid. Cost with the 634k-gaussian car: 30 to 60 ms per
camera frame including the per-part rotation. The Isaac viewport shows the opponent as one rigid
prim (no per-part USD yet); the pseudo sensors get the animation.

## Driving like a car: the opponent motion model

`scripts/simulate_opponent_path.py` produces the opponent's trajectory (JSON in the scene frame:
position, heading, speed, accelerations, steering, sideslip per 20 ms) that the demo and the
Isaac clients play back with `--opponent-trajectory`. Reference = the recorded lap projected onto
the floor plane, target speed = the ego's own recorded speed along the path (limited by a lateral
acceleration cap in corners), plus a gap controller that keeps the opponent `--gap` metres ahead
of where the ego was at that moment, so the ego never drives through it.

Two plants:

* `--plant kinematic` (default): bicycle model at the rear axle with vehicle limits
  (acceleration 3 m/s^2, braking 4 m/s^2, lateral 4 m/s^2, steering rate 3 rad/s, max
  0.30 rad), kinematic sideslip at the CoG, pure pursuit with speed-dependent lookahead. New1
  from frame 560: tracking error median 7 cm, speed 1.4 +- 0.2 m/s in the clip window,
  steering change 0.004 rad per frame, sideslip <= 4.6 deg, gap 2.2 m mean.
* `--plant gru`: the deployed GRU (run30) as plant with the V2 pure-pursuit controller. Above
  about 1 m/s on the tight New1 track the loop is unstable (speed hunting 0.1 to 1.3 m/s, 33 deg
  crab angle, tracking error over 1 m, and the ego drives through the slow car), consistent
  with the diverging closed-loop rollouts in the dynamics evaluation. Kept for experiments,
  not for visuals.

Post-processing is the same for both: positions low-passed at 5 Hz, body heading = travel
direction minus the (clipped, filtered) sideslip, yaw rate and accelerations from the smoothed
kinematics. Wheel spin, steering and attitude animation read these fields, so the visible
motion is consistent with the path.

```bash
python scripts/simulate_opponent_path.py --transforms $NEW1_ROOT/dataset/transforms.json \
    --floor-plane $NEW1_ROOT/dataset/floor_plane.json --start-frame 560 --lead 2.5 --gap 2.5 --lateral 0.1 \
    --out $NEW1_ROOT/dataset/opponent_gru.json           # add --plant gru for the learned plant
python scripts/render_learned_demo.py                     # picks up dataset/opponent_gru.json; --blur-samples 3 default
bash scripts/run_isaac_demo.sh --gui --opponent --opponent-trajectory $NEW1_ROOT/dataset/opponent_gru.json
```

**Motion blur.** `render_learned_demo.py` averages `--blur-samples` (3) sub-frame renders over
`--exposure-ms` (8): the camera, the opponents and their wheels move within the exposure, so
spoked rims stop strobing at 30 fps and fast passes smear the way the real OAK-D frames do.
Costs one render per sample; the LiDAR is not blurred.

## The LiDAR view in the clips

Every clip from `render_learned_demo.py` shows the LiDAR the way the sensor samples it: the
scan is rendered along **recorded Livox Mid-360 return directions** (`--livox-bag`, default the
session-142014 `all_data_0.db3`, 62 scan patterns over 6 s) at the recorded ~10 Hz scan times,
from the ego pose and object state of that instant, with an 8 m range cut. The right column
shows the same scan twice: a 3D isometric and a bird's-eye orthographic view in the sensor
frame, 1 m grid, height coloured (cyan -1 m to gold +3 m), returns on inserted objects in
orange. The old dense cubemap perspective panel is gone (`--lidar-sampling cubemap` keeps the
dense sampling if a bag is unavailable). The directions are sampling only: they come from
another session's successful returns, not a full firing pattern, and geometry comes from the
New1 LiDAR map. `scans/NNNN.npz` keeps every rendered scan with its ray directions and poses.

## Looking like part of the scene

Mesh-derived objects are razor sharp next to a splat scene that is soft by nature. Two knobs
close the gap: the camera renderer scales every inserted gaussian's footprint by
`object_soften` (default 1.3; server flag `--object-soften`, demo `--object-soften`, per-object
`meta["soften"]`). In a 1.0 / 1.4 / 1.8 / 2.4 sweep on New1, 1.4 matches the scene's softness
and 1.8 already bloats the wheels. And the sub-frame motion blur above. The LiDAR layer keeps
the exact geometry.

## Step time

`scripts/benchmark_sim_step.py` times one simulation step in-process and through the file
queue the Isaac clients use (medians, RTX, New1 flattened maps, ego at frame 600, two F1TENTH
cars plus four cones as objects, GPU otherwise idle):

| component | ms |
| --- | --- |
| camera render, scene only | 4 |
| camera + 6 objects (articulated) | 35 |
| LiDAR Livox pattern (15k rays), scene only | 5 |
| LiDAR + 6 objects, full assets (stride 1) | 97 |
| LiDAR + 6 objects, object stride 4 (default) | 40 |
| GRU step / MS-NN step (CPU) | 0.2 / 3 |
| in-process step: camera + Livox LiDAR + GRU, 6 objects | 75 |
| queue camera round trip, png (old default, level 6) / bmp (new default) | 163 / 39 |
| queue LiDAR round trip, compressed npz + stride 1 (old) / uncompressed + stride 4 | 218 / 48 to 60 |
| queue step, old defaults / new defaults | ~300 / ~100 |

What changed for the queue path: the camera server writes BMP (`--image-format bmp|png|jpg`;
BMP is lossless, PNG encode+decode alone cost 160 ms per 1280x800 frame), the LiDAR server
writes uncompressed npz (compressing 130k points cost ~100 ms), both servers and clients poll
every 5 ms instead of 50, the clients save their frame dumps as JPEG, and the LiDAR layer uses
`ObjectSplat.proxy(4)` (every 4th gaussian, 2x footprint: 57 vs 59 returns on the car, cones
161 vs 162) via `--object-lidar-stride`. `03_lidar_render_server.py --livox-bag <bag>` samples
along recorded Livox return directions, cycling through the recorded patterns.

## Limits and next steps

* Legacy mesh-derived objects (`f1tenth_legacy`, `cone_lowpoly`) have flat CAD colours: no wear, no
  reflections, no view-dependent shading, and the AutoDRIVE model is not the lab's exact car
  (different mast and LiDAR). The scene-cut `cone` was seen from the track side only, so its far
  side has floaters and it looks blurry up close. A walk-around capture of the real F1TENTH (D)
  gives the photoreal version.
  `cone_photoscan` preserves photographed texture and wear, but has no learned
  view-dependent appearance. The short demo shades car surface Gaussians with a
  baked lighting with ray-traced contact occlusion. The detailed asset adds distinct
  component colors and fine surface Gaussians; it remains CAD-derived. Rebuild/run
  commands and before/after views are in the [README opponent guide](../README.md#opponent-vehicles-as-gaussian-splats).
* Wheels, steering and body attitude are animated in the pseudo sensors, the Isaac viewport prim
  is still rigid (per-part USD export is the missing piece). The contact shadow is a fixed soft
  patch, not a light-dependent shadow; reflections of the object on the floor are not rendered.
* LiDAR returns on the object come from the same cubemap ray casting as the map, so the
  Livox scan pattern (`render_along_rays`) and range noise apply equally; intensity is the
  object's luminance, not a calibrated reflectivity.
* Motion: the opponent follows the recorded path. Driving it from a second dynamics model or
  a planner only needs a different `T_wo` per frame (`pose_from_camera_frame` is one way to
  build it; any 4x4 works).
