"""Verify native USD reference normalization without launching Isaac Sim."""
import numpy as np
import pytest

pytest.importorskip('pxr.Usd', reason='USD bindings required for asset regression')
from pxr import Gf, Usd, UsdGeom

from racing.assets import add_vehicle, mesh_triangles


def test_y_up_centimeter_vehicle_mesh_matches_rendered_normalization(tmp_path):
    source_path = tmp_path / 'vehicle.usda'
    source = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.SetStageUpAxis(source, 'Y')
    UsdGeom.SetStageMetersPerUnit(source, .01)
    model = UsdGeom.Xform.Define(source, '/Vehicle')
    source.SetDefaultPrim(model.GetPrim())
    mount = UsdGeom.Xform.Define(source, '/Vehicle/Mount')
    mount.AddTranslateOp().Set(Gf.Vec3d(7, 11, 13))
    mesh = UsdGeom.Mesh.Define(source, '/Vehicle/Mount/Body')
    mesh.CreatePointsAttr([(x, y, z) for x in (100., 200.)
                          for y in (20., 60.) for z in (-25., 25.)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([
        0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1,
        2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3,
    ])
    source.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    actor, pose = add_vehicle(stage, '/World/Car', source_path)
    triangles = mesh_triangles(stage, actor)
    vertices = triangles.reshape(-1, 3)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    np.testing.assert_allclose(hi-lo, [.42, .21, .168], atol=1e-6)
    np.testing.assert_allclose((hi+lo)[:2] / 2, [0, 0], atol=1e-6)
    assert lo[2] == pytest.approx(0., abs=1e-6)
    assert len(triangles) == 12

    moved = Gf.Matrix4d(1.)
    moved.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 75.))
    moved.SetTranslateOnly(Gf.Vec3d(4, -2, 0))
    pose.Set(moved)
    np.testing.assert_allclose(mesh_triangles(stage, actor), triangles, atol=1e-6)
