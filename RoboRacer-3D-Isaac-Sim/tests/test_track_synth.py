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
    lower, upper = stacked_hose_layers(path, np.random.default_rng(0), layers=2, sides=12)
    assert upper['points'][:, 2].min() >= lower['points'][:, 2].min() + .2
    assert 0.4 < upper['points'][:, 2].max() < 0.55   # two 30 cm hoses, flattened by their own weight


def test_usd_scene_builds(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd
    from track_synth.build import build
    geometry, usd = build(MAP, tmp_path, seed=0, sides=8)
    stage = Usd.Stage.Open(str(usd))
    hoses = [p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Track/Hose')]
    assert len(hoses) == 2 * len(geometry.walls)
    assert stage.GetPrimAtPath('/World/Hall/Floor').IsValid()
    assert (tmp_path / 'scenario.json').exists() and (tmp_path / 'textures/hose_normal.png').exists()


def test_autopilot_laps_without_contact(tmp_path):
    pytest.importorskip('pxr')
    from track_synth.build import build
    build(MAP, tmp_path, seed=0, sides=8)
    args = build_parser().parse_args(['--scene', str(tmp_path), '--kinematic', '--episodes', '2', '--seconds', '20'])
    scene = load_scene(tmp_path); last = {}
    for ep, setup, snap, due in episodes(scene, select_model(args), args):
        last[ep] = snap
    assert set(last) == {0, 1}
    assert all(s.status == 'running' and s.time_s == pytest.approx(20, abs=.05) for s in last.values())


def test_camera_height_and_pitch_match_car02(tmp_path):
    pytest.importorskip('pxr')
    from track_synth.build import build
    from track_synth.collect import camera_pose, scaled_intrinsics
    build(MAP, tmp_path, seed=0, sides=8)
    scene = load_scene(tmp_path)
    pose = camera_pose(scene, dict(x=1., y=2., yaw=0.))
    assert pose[2, 3] == pytest.approx(.25, abs=.01)                       # optical centre above the floor
    assert np.degrees(np.arcsin(pose[2, 2])) == pytest.approx(3., abs=.5)  # optical axis pitched ~3 deg up
    assert camera_pose(scene, dict(x=0., y=0., yaw=0.), camera_height=.3)[2, 3] == pytest.approx(.3)
    big = scaled_intrinsics(scene['camera'], 2)
    assert (big['width'], big['height']) == (1280, 800) and big['fx'] == pytest.approx(2 * scene['camera']['fx'])
