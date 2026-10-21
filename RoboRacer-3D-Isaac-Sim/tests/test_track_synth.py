from pathlib import Path
import numpy as np
import pytest
from track_synth.extract import extract
from track_synth.meshes import sweep_tube, stacked_hose_layers
from track_synth.collect import build_parser, episodes, load_scene, select_model

MAP = Path(__file__).resolve().parents[1] / 'reference/iros2026/IROS2026.yaml'


@pytest.fixture(scope='module')
def geometry():
    return extract(MAP)


def closed_length(points):
    return np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1).sum()


def test_map_geometry(geometry):
    assert geometry.bounds == pytest.approx((0, 0, 51.8, 31.55))
    assert 190 < closed_length(geometry.centerline) < 230
    assert np.linalg.norm(np.diff(np.vstack([geometry.centerline, geometry.centerline[:1]]), axis=0), axis=1).max() < .15
    assert geometry.half_width.min() > .7 and np.median(geometry.half_width) == pytest.approx(1.1, abs=.15)
    assert len(geometry.holes) == 1


def test_centerline_inside_free_space(geometry):
    from matplotlib.path import Path as P
    assert P(geometry.outer).contains_points(geometry.centerline).all()
    assert not any(P(h).contains_points(geometry.centerline).any() for h in geometry.holes)


def test_sweep_tube_is_consistent():
    path = np.c_[np.linspace(0, 2, 21), np.zeros(21)]
    mesh = sweep_tube(path, .15, .15, .13, sides=12)
    assert mesh['counts'].sum() == len(mesh['indices']) == len(mesh['uvs'])
    assert mesh['indices'].max() < len(mesh['points']) == len(mesh['normals'])
    assert np.allclose(np.linalg.norm(mesh['normals'], axis=1), 1)
    assert mesh['points'][:, 2].min() == pytest.approx(.02, abs=1e-6)


def test_hose_layers_stack_without_penetration():
    path = np.c_[np.linspace(0, 5, 51), np.zeros(51)]
    (lower, upper), straps = stacked_hose_layers(path, np.random.default_rng(0), layers=2, sides=12, upper_fraction=1.)
    low, up = lower['points'][:-2].reshape(51, 12, 3), upper['points'][:-2].reshape(51, 12, 3)
    assert (up[..., 2].min(1) >= low[..., 2].min(1) + .03).all()        # upper hose rests on the lower one
    assert (up[..., 2].min(1) <= low[..., 2].max(1) + .01).all()        # ... and touches it (no floating)
    assert 0.25 < upper['points'][:, 2].max() < 0.55                    # two soft 30 cm hoses
    assert 2 <= len(straps) <= 5                        # straps every ~1.8 m


def test_upper_layer_only_on_part_of_the_wall():
    path = np.c_[np.linspace(0, 30, 301), np.zeros(301)]
    (lower, upper), _ = stacked_hose_layers(path, np.random.default_rng(1), layers=2, sides=12, upper_fraction=.4)
    ring = upper['points'][:-2].reshape(301, 12, 3)
    height = ring[..., 2].max(1) - ring[..., 2].min(1)
    assert .25 < (height > .05).mean() < .75           # stacked hose present on some, not all, of the wall


def test_usd_scene_builds(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd
    from track_synth.build import build
    geometry, usd = build(MAP, tmp_path, seed=0, sides=8, raceline=False)
    stage = Usd.Stage.Open(str(usd))
    hoses = [p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Track/Hose')]
    assert len(hoses) == 2 * len(geometry.walls)
    assert stage.GetPrimAtPath('/World/Hall/Floor').IsValid()
    assert (tmp_path / 'scenario.json').exists() and (tmp_path / 'textures/hose_normal.png').exists()


def test_autopilot_laps_without_contact(tmp_path):
    pytest.importorskip('pxr')
    from track_synth.build import build
    build(MAP, tmp_path, seed=0, sides=8, raceline=False)
    args = build_parser().parse_args(['--scene', str(tmp_path), '--kinematic', '--driver', 'centerline',
                                      '--episodes', '2', '--seconds', '20', '--lead-in', '0'])
    scene = load_scene(tmp_path); last = {}
    for ep, setup, snap, due in episodes(scene, select_model(args), args):
        last[ep] = snap
    assert set(last) == {0, 1}
    assert all(s.status == 'running' and s.time_s == pytest.approx(20, abs=.05) for s in last.values())


def test_camera_height_and_pitch_match_car02(tmp_path):
    pytest.importorskip('pxr')
    from track_synth.build import build
    from track_synth.collect import camera_pose, scaled_intrinsics
    build(MAP, tmp_path, seed=0, sides=8, raceline=False)
    scene = load_scene(tmp_path)
    pose = camera_pose(scene, dict(x=1., y=2., yaw=0.))
    assert pose[2, 3] == pytest.approx(.18, abs=.005)                      # measured camera height above the floor
    assert np.degrees(np.arcsin(pose[2, 2])) == pytest.approx(5.3, abs=.3)  # calibrated: optical axis ~5.3 deg up
    assert camera_pose(scene, dict(x=0., y=0., yaw=0.), camera_height=.3)[2, 3] == pytest.approx(.3)
    big = scaled_intrinsics(scene['camera'], 2)
    assert (big['width'], big['height']) == (1280, 800) and big['fx'] == pytest.approx(2 * scene['camera']['fx'])


def test_panorama_faces_land_in_latlong_convention():
    from track_synth.splat_panorama import FACES, equirect_from_faces
    colours = {name: np.full((32, 32, 3), 40 * (i + 1), np.uint8) for i, name in enumerate(FACES)}
    pano = equirect_from_faces(colours, 256)
    h, w = pano.shape[:2]
    assert (pano[h // 2, w // 2] == colours['front'][0, 0]).all()      # image centre = forward
    assert (pano[h // 2, w // 4] == colours['left'][0, 0]).all()       # +lon (left half) = left
    assert (pano[h // 2, 3 * w // 4] == colours['right'][0, 0]).all()
    assert (pano[0, w // 3] == colours['up'][0, 0]).all() and (pano[-1, w // 3] == colours['down'][0, 0]).all()


def test_build_with_panorama_and_real_floor(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdLux
    from PIL import Image
    from track_synth.build import build, REFERENCE
    pano = tmp_path / 'pano.png'; Image.new('RGB', (64, 32), (120, 120, 120)).save(pano)
    out = tmp_path / 'scene'
    build(MAP, out, seed=0, sides=8, floor_texture=REFERENCE / 'floor_real.jpg', hdri=pano, raceline=False)
    stage = Usd.Stage.Open(str(out / 'track.usd'))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath('/World/Lighting/HallPanorama'))
    assert dome.GetTextureFileAttr().Get().path == './textures/hall_pano.png'
    assert not stage.GetPrimAtPath('/World/Hall/Wall0').IsValid()        # placeholder hall replaced
    domes = [p for p in stage.Traverse() if p.GetTypeName() == 'DomeLight']
    assert len(domes) == 1                                                # RTX uses one environment only
    assert stage.GetPrimAtPath('/World/Lighting/FillSun0').GetTypeName() == 'DistantLight'
    assert (out / 'textures/floor.jpg').exists() and (out / 'textures/hall_pano.png').exists()


@pytest.fixture(scope='module')
def raceline_scene(tmp_path_factory):
    pytest.importorskip('pxr')
    from track_synth.build import build
    out = tmp_path_factory.mktemp('raceline_scene')
    build(MAP, out, seed=0, sides=8, line_mus=(.2,))
    return out


def test_raceline_is_faster_than_centerline_and_drivable(raceline_scene):
    scene = load_scene(raceline_scene)
    info = scene['raceline_info']
    assert info['lap_time_s'] < .97 * info['centerline_lap_time_s']
    assert info['feasible'] and info['kappa_max'] < .9                      # below the steering limit tan(.29)/.3225
    assert scene['raceline_speed'].max() <= info['limits']['v_max'] + 1e-9 and scene['raceline_speed'].min() > 1.
    assert scene['raceline_clearance'].min() > .2                           # car + safety margin to the hose surface


def test_raceline_driver_laps_without_contact_and_skips_lead_in(raceline_scene):
    scene = load_scene(raceline_scene)
    args = build_parser().parse_args(['--scene', str(raceline_scene), '--kinematic', '--episodes', '1',
                                      '--seconds', '30', '--lead-in', '2', '--speed-scale-min', '1', '--speed-scale-max', '1'])
    frames = []; last = None; speeds = []
    for ep, setup, snap, due in episodes(scene, select_model(args), args):
        last = snap; speeds.append(snap.vehicles['ego'].vx)
        if 'camera' in due:
            frames.append(snap.time_s)
    assert last.status == 'running'
    assert min(frames) >= 2.0 - 1e-9                                        # no camera frames during the lead-in
    assert 29 * 20 <= len(frames) <= 30 * 20 + 2                            # --seconds counts after the lead-in
    assert max(speeds) > 2.2                                                # runs near the line's top speed


REF = Path(__file__).resolve().parents[1] / 'reference'


def test_vehicle_constants_match_the_reference_files():
    import re
    import yaml
    from track_synth import vehicle
    from racing import geometry
    # camera intrinsics (calibration/camera_intrinsics.yaml, rgb 1280x800)
    rgb = yaml.safe_load((REF / 'calibration/camera_intrinsics.yaml').read_text())['rgb']['camera_matrix']
    for key in ('fx', 'fy', 'cx', 'cy'):
        assert vehicle.INTRINSICS_1280[key] == pytest.approx(rgb[key])
    # camera mount: calibrated base_link -> optical frame (x, y and orientation)
    ext = yaml.safe_load((REF / 'calibration/extrinsics_candidate_checked_v2.yaml').read_text())
    mount = next(v for k, v in _walk(ext) if k == 'base_link_to_oak_rgb_camera_optical_frame')
    assert vehicle.CAMERA_XY == pytest.approx((mount['translation']['x'], mount['translation']['y']))
    assert vehicle.CAMERA_QUATERNION == pytest.approx(tuple(mount['rotation'][c] for c in 'xyzw'))
    # geometry and steering (vehicle_parameters.txt) vs the racing runtime
    text = (REF / 'vehicle/vehicle_parameters.txt').read_text()
    assert float(re.search(r'Radstand\s+L\s+=\s+([\d.]+)', text).group(1)) == vehicle.WHEELBASE
    assert float(re.search(r'Breite \(über Reifen\)\s+=\s+([\d.]+)', text).group(1)) == vehicle.WIDTH == geometry.WIDTH
    assert float(re.search(r'Länge  \(über Räder\)\s+=\s+([\d.]+)', text).group(1)) == vehicle.LENGTH == geometry.LENGTH
    assert float(re.search(r'steering_range: ±([\d.]+) rad', text).group(1)) == vehicle.MAX_STEERING
    # validated controller and GRU speed range (closed_loop/10_v2_main_demo.py, 16_raceline_retest...)
    demo = (REF.parent / 'simulation/closed_loop/10_v2_main_demo.py').read_text()
    assert float(re.search(r'ld_min=([\d.]+), ld_max=([\d.]+), ld_curve_k=([\d.]+)', demo).group(1)) == vehicle.LD_MIN
    retest = (REF.parent / 'simulation/closed_loop/16_raceline_retest_correct_checkpoint.py').read_text()
    assert (float(re.search(r'train_speed_min=([\d.]+)', retest).group(1)),
            float(re.search(r'train_speed_max=([\d.]+)', retest).group(1))) == (vehicle.SPEED_MIN, vehicle.SPEED_MAX)


def _walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)


def test_acceleration_limits_match_the_car02_reference_raceline():
    import json
    from track_synth import vehicle
    from track_synth.raceline import curvature
    from scipy.interpolate import splprep, splev
    ref = json.loads((REF / 'raceline_ftm_halle/reftraj_FTM_Halle_min_time.json').read_text())
    assert (min(ref['ref_acc']), max(ref['ref_acc'])) == pytest.approx((-vehicle.A_DEC, vehicle.A_ACC), abs=.01)
    x, y, v = (np.array(ref[k][:-1]) for k in ('pos_x', 'pos_y', 'ref_v'))
    tck, _ = splprep([x, y], per=1, s=.002)
    fine = np.linspace(0, 1, 2000, endpoint=False)
    path = np.c_[tuple(splev(fine, tck))]
    lateral = np.interp(fine, np.linspace(0, 1, len(x), endpoint=False), v, period=1) ** 2 * np.abs(curvature(path, .1))
    assert lateral.max() == pytest.approx(vehicle.A_LAT, abs=.05)


def test_collision_polygons_sit_on_the_hose_surface(raceline_scene):
    from track_synth.build import HOSE_RADIUS, WALL_HALF_LINE
    from track_synth.collect import hose_clearance
    from matplotlib.path import Path as P
    scene = load_scene(raceline_scene)
    plain = extract(MAP)
    # polygon edge ~ hose surface: the free-space edge is WALL_HALF_LINE from the hose centre, so the polygon
    # sits (HOSE_RADIUS - WALL_HALF_LINE) inside it
    outer = np.array(scene['outer_boundary'])
    shrink = np.min(np.linalg.norm(plain.outer[None, :, :] - outer[::5, None, :], axis=2), axis=1)
    assert np.median(shrink) == pytest.approx(HOSE_RADIUS - WALL_HALF_LINE, abs=.03)
    # a car standing on the polygon edge touches the hose surface (clearance ~ 0), one on the raceline does not
    from racing.core import VehicleState
    on_line = VehicleState(x=float(scene['raceline'][0, 0]), y=float(scene['raceline'][0, 1]), yaw=0.)
    assert hose_clearance(scene, on_line) > .15


def test_raceline_keeps_the_car_clear_of_the_hoses(raceline_scene):
    scene = load_scene(raceline_scene)
    from track_synth.build import CLEARANCE
    assert scene['raceline_clearance'].min() >= CLEARANCE + .13 - .02      # hose surface to line >= half width + clearance


def test_stale_scene_gets_a_rebuild_hint(tmp_path):
    import json
    (tmp_path / 'scenario.json').write_text(json.dumps(dict(usd='track.usd', reference_path=[[0, 0]], half_width=[1.])))
    with pytest.raises(SystemExit, match='rebuild'):
        load_scene(tmp_path)
