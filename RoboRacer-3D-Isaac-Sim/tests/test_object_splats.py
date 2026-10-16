"""Object-splat maths: SH rotation, pose composition, frame recovery. CPU only, no Isaac."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("n_coeffs", [1, 4, 9, 16])
def test_sh_rotation_matches_rotated_view_directions(n_coeffs):
    rng = np.random.default_rng(1)
    R = _random_rotation(2)
    sh = rng.normal(size=(50, n_coeffs, 3))
    d_world = rng.normal(size=(50, 3)); d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    # colour seen from world direction d must equal the object-frame colour at R^T d
    got = osp.sh_eval(osp.rotate_sh(sh, R), d_world)
    want = osp.sh_eval(sh, d_world @ R)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_sh_rotation_is_orthogonal_and_block_diagonal():
    D = osp.sh_rotation(_random_rotation(3))
    np.testing.assert_allclose(D @ D.T, np.eye(16), atol=1e-9)
    for lo, hi in ((0, 1), (1, 4), (4, 9), (9, 16)):
        off = D[lo:hi].copy(); off[:, lo:hi] = 0
        assert np.abs(off).max() < 1e-9
    np.testing.assert_allclose(osp.sh_rotation(np.eye(3)), np.eye(16), atol=1e-9)


def test_quat_from_matrix_roundtrip():
    for seed in range(5):
        R = _random_rotation(seed)
        q = osp.quat_from_matrix(R)
        w, x, y, z = q
        R2 = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        np.testing.assert_allclose(R2, R, atol=1e-9)


def test_posed_applies_similarity_to_means_scales_quats_and_sh():
    rng = np.random.default_rng(4)
    n = 20
    obj = osp.ObjectSplat(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 4)),
                          rng.normal(size=n), rng.normal(size=(n, 16, 3)))
    R, s, t = _random_rotation(5), 2.5, np.array([1.0, -2.0, 0.5])
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    p = {k: v.cpu().numpy() for k, v in obj.posed(T, device="cpu").items()}
    np.testing.assert_allclose(p["means"], obj.means @ (s * R).T + t, atol=1e-5)
    np.testing.assert_allclose(p["scales_log"], obj.scales_log + np.log(s), atol=1e-5)
    # rotated covariance axes: rotate the first principal axis of each gaussian
    def axes(q):
        w, x, y, z = q.T
        return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1)
    np.testing.assert_allclose(axes(p["quats"]), axes(obj.quats) @ R.T, atol=1e-5)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    np.testing.assert_allclose(osp.sh_eval(p["sh"], d), osp.sh_eval(obj.sh, d @ R), atol=1e-4)
    p2 = obj.posed(T, device="cpu", opaque_min=0.95)["opacities"].numpy()
    assert p2.min() >= np.log(0.95 / 0.05) - 1e-5


def test_decompose_similarity_rejects_shear():
    T = np.eye(4); T[0, 1] = 0.3
    with pytest.raises(ValueError):
        osp.decompose_similarity(T)


def test_pose_from_camera_frame_heading_and_ground():
    c2w = np.eye(4)                      # OpenGL camera looking along -z_world? no: rotate so it looks along +x
    # camera looking along world +x: cam -z = +x  -> cam z = -x; cam y = +z (up); cam x = -y... build from columns
    c2w[:3, :3] = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    c2w[:3, 3] = [1.0, 2.0, 0.5]
    T = osp.pose_from_camera_frame(c2w, ground_offset=0.2, lateral=0.3, ahead=1.0)
    np.testing.assert_allclose(T[:3, 0], [1, 0, 0], atol=1e-12)          # +x forward
    np.testing.assert_allclose(T[:3, 1], [0, 1, 0], atol=1e-12)          # +y left
    np.testing.assert_allclose(T[:3, 3], [2.0, 2.3, 0.3], atol=1e-12)


def test_similarity_from_frames_recovers_scale_rotation_translation():
    R, s, t = _random_rotation(7), 3.9868, np.array([0.3, -1.2, 2.0])
    rng = np.random.default_rng(8)
    frames = []
    for _ in range(6):
        c2w = np.eye(4); c2w[:3, :3] = _random_rotation(int(rng.integers(100))); c2w[:3, 3] = rng.normal(size=3)
        iso = np.eye(4); iso[:3, :3] = R @ c2w[:3, :3]; iso[:3, 3] = s * R @ c2w[:3, 3] + t
        frames.append({"transform_matrix": iso.tolist(), "transform_matrix_source": c2w.tolist()})
    S = osp.similarity_from_frames(frames)
    np.testing.assert_allclose(S[:3, :3], s * R, atol=1e-9)
    np.testing.assert_allclose(S[:3, 3], t, atol=1e-9)
    assert osp.similarity_from_frames([{"transform_matrix": np.eye(4).tolist()}]) is None


def test_extract_object_recentres_and_rotates(tmp_path):
    rng = np.random.default_rng(9)
    # a scene in training units: ground at z=-0.7 (world) plus a small object at (2, 1)
    world_from_train = np.diag([2.0, 2.0, 2.0, 1.0])       # train -> world: scale 2, no rotation
    ground = np.column_stack([rng.uniform(-3, 3, 6000), rng.uniform(-3, 3, 6000), np.full(6000, -0.7)])
    obj_w = np.array([2.0, 1.0, -0.7]) + rng.uniform([-0.1, -0.1, 0.02], [0.1, 0.1, 0.3], size=(60, 3))
    means_w = np.vstack([ground, obj_w])
    n = len(means_w)
    g = {"means": means_w / 2.0, "scales_log": np.full((n, 3), np.log(0.01)), "quats": np.tile([1.0, 0, 0, 0], (n, 1)),
         "opacities": np.full(n, 3.0), "sh": rng.normal(size=(n, 16, 3))}
    floor = osp.estimate_floor_z(means_w, np.full(n, 0.9), center=[2.0, 1.0, -0.5])
    assert abs(floor + 0.7) < 0.03
    obj = osp.extract_object(g, world_from_train, center=[2.0, 1.0, -0.5], size=[0.4, 0.4, 0.6], yaw_deg=90.0, floor_z=floor)
    assert obj.n == 60
    lo, hi = obj.bbox()
    assert lo[2] >= 0.0 and hi[2] < 0.35 and abs(lo[0] + hi[0]) < 0.05 and abs(lo[1] + hi[1]) < 0.05
    # yaw 90: world +y becomes object +x
    np.testing.assert_allclose(obj.means[:, 0], (obj_w[:, 1] - 1.0), atol=1e-6)
    np.testing.assert_allclose(obj.scales_log[0], np.log(0.02), atol=1e-6)   # scale carried into metres
    p = obj.save(tmp_path / "o.npz")
    obj2 = osp.ObjectSplat.load(p)
    assert obj2.n == 60 and obj2.meta["n"] == 60


def test_trajectory_up_recovers_tilted_floor_normal():
    rng = np.random.default_rng(11)
    R = _random_rotation(12)                       # tilted world
    frames = []
    for k in range(200):
        a = 2 * np.pi * k / 200
        p_level = np.array([3 * np.cos(a), 2 * np.sin(a), 0.15])
        c2w = np.eye(4); c2w[:3, :3] = R @ np.column_stack([[-np.sin(a), np.cos(a), 0], [0, 0, 1], [-np.cos(a), -np.sin(a), 0]])
        c2w[:3, 3] = R @ p_level + rng.normal(scale=0.005, size=3)
        frames.append({"transform_matrix": c2w.tolist()})
    up = osp.trajectory_up(frames)
    np.testing.assert_allclose(up, R[:, 2], atol=0.01)
    T = osp.pose_from_camera_frame(frames[0]["transform_matrix"], ground_offset=0.15, up=up)
    np.testing.assert_allclose(T[:3, 2], R[:, 2], atol=0.01)                     # object stands on the floor normal
    assert abs(np.dot(T[:3, 0], up)) < 0.02                                       # heading in the floor plane
    np.testing.assert_allclose(np.linalg.inv(R) @ T[:3, 3], [3, 0, 0], atol=0.02)  # origin on the floor under the camera
    assert np.allclose(osp.trajectory_up(frames[:5]), [0, 0, 1])


def test_frame_ahead_by_path_distance_and_by_frames():
    frames = [{"transform_matrix": [[1, 0, 0, 0.1 * k], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]} for k in range(50)]
    cum = osp.path_arclength(frames)
    np.testing.assert_allclose(cum[10], 1.0)
    assert osp.frame_ahead(0, cum, 2.5, None) == 25
    assert osp.frame_ahead(40, cum, 2.5, None) == 49          # clamped at the end
    assert osp.frame_ahead(3, cum, None, 7) == 10


def test_frame_from_axes_maps_source_axes_to_object_frame():
    R = osp.frame_from_axes("X", "Y")             # Y-up asset, +X forward (F1TENTH USD)
    np.testing.assert_allclose(R @ [1, 0, 0], [1, 0, 0])   # forward -> +x
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1])   # up -> +z
    np.testing.assert_allclose(R @ [0, 0, -1], [0, 1, 0])  # -Z (left wheel side) -> +y left
    with pytest.raises(ValueError):
        osp.frame_from_axes("X", "-X")


def test_gaussians_from_mesh_covers_area_and_lies_in_plane():
    # unit square in the xy plane at z=0.2, red
    V = np.array([[0, 0, .2], [1, 0, .2], [1, 1, .2], [0, 1, .2]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    g = osp.gaussians_from_mesh(V, F, np.tile([1.0, 0.0, 0.0], (2, 1)), spacing=0.02, overlap=2.0)
    n = len(g["means"])
    assert abs(n - 2.0 * 1.0 / 0.02 ** 2) < 0.1 * n                   # overlap * area / spacing^2
    np.testing.assert_allclose(g["means"][:, 2], 0.2, atol=1e-9)
    assert g["means"][:, :2].min() >= 0 and g["means"][:, :2].max() <= 1
    # thin axis is the face normal (z): rotate local z by the quaternion
    w, x, y, z = g["quats"].T
    nz = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], 1)
    np.testing.assert_allclose(np.abs(nz[:, 2]), 1.0, atol=1e-6)
    assert np.allclose(np.exp(g["scales_log"][:, 2]), 0.005)
    rgb = 0.5 + osp.SH_C0 * g["sh"][:, 0]
    np.testing.assert_allclose(rgb, [[1, 0, 0]] * n, atol=1e-6)
    # tiny faces: a million-face mesh must not force a gaussian per face
    t = np.array([[0, 0, 0], [1e-4, 0, 0], [0, 1e-4, 0]], float)
    g2 = osp.gaussians_from_mesh(t, np.array([[0, 1, 2]]), np.ones((1, 3)) * 0.5, spacing=0.004)
    assert len(g2["means"]) <= 1


def test_splat_ply_roundtrip_and_object_frame(tmp_path):
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(3)
    n = 30
    means = rng.normal(size=(n, 3)); means[:, 1] += 5          # Y-up asset floating at y~5
    sh = rng.normal(size=(n, 16, 3)); scales = rng.normal(size=(n, 3)); quats = rng.normal(size=(n, 4)); opac = rng.normal(size=n)
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4")] + [(f"f_dc_{i}", "f4") for i in range(3)] + \
           [(f"f_rest_{i}", "f4") for i in range(45)] + [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    arr = np.empty(n, dtype=cols)
    arr["x"], arr["y"], arr["z"] = means.T
    for c in range(3):
        arr[f"f_dc_{c}"] = sh[:, 0, c]
        for k in range(15):
            arr[f"f_rest_{c * 15 + k}"] = sh[:, k + 1, c]              # channel-major, as 3DGS writes it
        arr[f"scale_{c}"] = scales[:, c]
    arr["opacity"] = opac
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]
    p = tmp_path / "s.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(p))
    g = osp.load_splat_ply(p)
    np.testing.assert_allclose(g["sh"], sh, atol=1e-6)
    np.testing.assert_allclose(g["means"], means, atol=1e-6)
    obj = osp.gaussians_to_object(g, osp.frame_from_axes("-Z", "Y"), scale=2.0)
    lo, hi = obj.bbox()
    assert abs(lo[2]) < 1e-5 and abs(lo[0] + hi[0]) < 1e-4 and abs(lo[1] + hi[1]) < 1e-4
    np.testing.assert_allclose(hi[2] - lo[2], 2.0 * (means[:, 1].max() - means[:, 1].min()), atol=1e-4)
    # colours seen from a direction agree with the source SH at the un-rotated direction
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    R = osp.frame_from_axes("-Z", "Y")
    np.testing.assert_allclose(osp.sh_eval(obj.sh, d), osp.sh_eval(sh, d @ R), atol=1e-4)


def test_floor_plane_fit_rejects_outliers_and_poses_objects_on_it(tmp_path):
    rng = np.random.default_rng(21)
    n = np.array([0.1, -0.05, 1.0]); n /= np.linalg.norm(n)
    p0 = np.array([0.5, -1.0, -0.6])
    # floor samples with 5 mm noise plus 10 % outliers (cones, floaters) up to 40 cm off
    xy = rng.uniform(-4, 4, size=(2000, 2))
    P = np.c_[xy, np.zeros(2000)]
    P = P - np.outer(P @ n, n) + p0 + np.outer(rng.normal(scale=0.005, size=2000), n)
    out = rng.random(2000) < 0.1
    P[out] += np.outer(rng.uniform(0.05, 0.4, out.sum()), n)
    plane = osp.FloorPlane.fit(P, up_hint=[0, 0, 1])
    assert abs(np.dot(plane.normal, n)) > 0.99999
    assert abs(plane.height(p0)) < 0.003
    assert plane.meta["inlier_rms_m"] < 0.01 and plane.meta["n_inliers"] > 1700
    # a camera 0.15 m above the floor with 5 cm height jitter: the object must not inherit it
    heights = []
    for k in range(20):
        c2w = np.eye(4); c2w[:3, :3] = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        c2w[:3, 3] = p0 + (0.15 + rng.normal(scale=0.05)) * n + [0.1 * k, 0, 0]
        T = osp.pose_from_camera_frame(c2w, floor=plane, lateral=0.3, ahead=1.0)
        heights.append(plane.height(T[:3, 3]))
        np.testing.assert_allclose(T[:3, 2], plane.normal, atol=1e-9)
        assert abs(np.dot(T[:3, 0], plane.normal)) < 1e-9
    assert np.abs(heights).max() < 1e-9
    plane.clearance = 0.02
    T = osp.pose_from_camera_frame(c2w, floor=plane)
    assert abs(plane.height(T[:3, 3]) - 0.02) < 1e-9
    plane.clearance = 0.0
    # round trip through json and a similarity transform
    S = np.eye(4); S[:3, :3] = 2.0 * _random_rotation(22); S[:3, 3] = [1, 2, 3]
    q = plane.transformed(S)
    np.testing.assert_allclose(q.height((S @ np.append(P[~out][0], 1))[:3]), 0.0, atol=0.03)
    np.testing.assert_allclose(q.normal, S[:3, :3] @ plane.normal / 2.0, atol=1e-9)
    plane.clearance = 0.02
    assert abs(plane.transformed(S).clearance - 0.04) < 1e-12
    p = plane.save(tmp_path / "floor.json")
    assert abs(osp.FloorPlane.load(p).clearance - 0.02) < 1e-12



def _car_with_wheels():
    rng = np.random.default_rng(31)
    n = 400
    means = rng.uniform([-0.25, -0.15, 0.0], [0.25, 0.15, 0.2], size=(n, 3))
    parts = np.zeros(n, np.int16)
    # a wheel: gaussians on a ring of radius 0.05 around the axle at (0.2, 0.15, 0.05)
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    wheel = np.c_[0.2 + 0.05 * np.cos(ang), np.full(60, 0.15), 0.05 + 0.05 * np.sin(ang)]
    means = np.vstack([means, wheel]); parts = np.concatenate([parts, np.ones(60, np.int16)])
    N = len(means)
    obj = osp.ObjectSplat(means, np.full((N, 3), np.log(0.005)), np.tile([1.0, 0, 0, 0], (N, 1)), np.full(N, 3.0),
                          rng.normal(size=(N, 16, 3)), {"name": "car"}, parts=parts)
    obj.meta["parts"] = osp.wheel_parts_meta(obj.means, obj.parts, ["body", "wheel_fl"])
    return obj


def test_wheel_part_meta_and_articulation_rotates_only_the_wheel():
    obj = _car_with_wheels()
    w = obj.part_meta[1]
    assert w["kind"] == "wheel" and w["steer"] is True
    np.testing.assert_allclose(w["centre"], [0.2, 0.15, 0.05], atol=1e-6)
    assert abs(w["radius"] - 0.05) < 1e-6
    tf = obj.articulated(wheel_angle=np.pi / 2)
    p = {k: v.cpu().numpy() for k, v in obj.posed(np.eye(4), device="cpu", part_transforms=tf).items()}
    body = obj.parts == 0
    np.testing.assert_allclose(p["means"][body], obj.means[body], atol=1e-6)         # body untouched
    ring = p["means"][~body] - np.array(w["centre"])
    np.testing.assert_allclose(np.linalg.norm(ring[:, [0, 2]], axis=1), 0.05, atol=1e-5)  # still on the ring
    # a point at the top of the wheel moved to the front (rotation about +y by 90 deg)
    top = np.argmin(np.abs(obj.means[~body, 0] - 0.2) + np.abs(obj.means[~body, 2] - 0.10))
    np.testing.assert_allclose(ring[top], [0.05, 0.0, 0.0], atol=1e-4)
    # SH of the wheel follow the rotation, body SH unchanged
    d = np.array([[0.3, 0.5, 0.8]]) / np.linalg.norm([0.3, 0.5, 0.8])
    Rw = osp.rot_axis([0, 1, 0], np.pi / 2)
    np.testing.assert_allclose(osp.sh_eval(p["sh"][~body][:1], d), osp.sh_eval(obj.sh[~body][:1], d @ Rw), atol=1e-4)
    np.testing.assert_allclose(p["sh"][body], obj.sh[body], atol=1e-6)
    # steering yaws the wheel about the vertical through its centre
    tf = obj.articulated(steer_angle=0.3)
    q = obj.posed(np.eye(4), device="cpu", part_transforms=tf)["means"].numpy()
    c = np.array(w["centre"])
    np.testing.assert_allclose(q[~body] - c, (obj.means[~body] - c) @ osp.rot_z(0.3).T, atol=1e-5)
    # pitch tilts the body about the ground origin
    tf = obj.articulated(pitch=0.1)
    q = obj.posed(np.eye(4), device="cpu", part_transforms=tf)["means"].numpy()
    np.testing.assert_allclose(q[body], obj.means[body] @ osp.rot_y(0.1).T, atol=1e-5)


def test_contact_shadow_is_visual_only_and_round_trips(tmp_path):
    obj = _car_with_wheels()
    sh = osp.contact_shadow(0.5, 0.3)
    assert sh.n > 200 and np.all(sh.means[:, 2] > 0) and sh.opacities.max() < 3.0
    full = obj.with_part(sh, {"name": "shadow", "kind": "shadow", "lidar": False})
    assert full.n == obj.n + sh.n and len(full.part_meta) == 3
    assert full.lidar_mask().sum() == obj.n
    p = full.posed(np.eye(4), device="cpu", for_lidar=True)
    assert len(p["means"]) == obj.n
    assert len(full.lidar_colors(3, device="cpu")) == obj.n
    q = full.posed(np.eye(4), device="cpu", part_transforms=full.articulated(pitch=0.2))["means"].numpy()
    np.testing.assert_allclose(q[full.parts == 2], full.means[full.parts == 2], atol=1e-6)   # shadow stays on the floor
    path = full.save(tmp_path / "car.npz")
    back = osp.ObjectSplat.load(path)
    assert back.parts is not None and back.parts.max() == 2 and back.part_meta[1]["kind"] == "wheel"


def test_driving_state_and_attitude():
    t = np.linspace(0, 10, 501)
    # circle of radius 2 m at 1 m/s
    P = np.c_[2 * np.cos(t / 2), 2 * np.sin(t / 2), np.zeros_like(t)]
    st = osp.driving_state_from_path(P, t, wheelbase=0.33)
    np.testing.assert_allclose(st["v"][20:-20], 1.0, atol=0.02)
    np.testing.assert_allclose(st["yaw_rate"][20:-20], 0.5, atol=0.02)
    np.testing.assert_allclose(st["steer"][20:-20], np.arctan(0.33 * 0.5), atol=0.01)
    np.testing.assert_allclose(st["a_y"][20:-20], 0.5, atol=0.02)
    pitch, roll = osp.attitude_from_accel(a_x=-3.0, a_y=2.0)
    assert pitch > 0 and roll > 0 and abs(np.degrees(pitch)) <= 4.0
    assert osp.classify_part("Wheel_3_solid2") == "wheel_fl" and osp.classify_part("Chassis_1_solid1") == "body"
    assert osp.classify_part("Rear_Wheel_Hub_2_solid1") == "wheel_rr" and osp.classify_part("Wheel_Shaft_2_solid1") == "wheel_rr"
