from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pxr.Usd')
from pxr import Gf, Usd, UsdGeom, UsdLux

from racing.visual_check import run_visual_check, run_occlusion_check, run_material_check


def fixture():
    stage = Usd.Stage.CreateInMemory()
    actor = UsdGeom.Xform.Define(stage, '/World/Opponent')
    op = actor.AddTransformOp()
    original = Gf.Matrix4d(1.)
    original.SetTranslateOnly(Gf.Vec3d(4, 5, 0))
    op.Set(original)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(750.)
    return stage, actor, op, original, light


class Annotator:
    def __init__(self, actor, light, fail_on=None):
        self.actor, self.light, self.fail_on = actor, light, fail_on
        self.calls = 0

    def get_data(self):
        self.calls += 1
        if self.calls == self.fail_on:
            return np.empty((0,))
        frame = np.full((40, 40, 4), 20, np.uint8)
        if self.actor.ComputeVisibility() != 'invisible':
            frame[10:26, 10:26, :3] = min(255, 60+20*self.light.GetIntensityAttr().Get())
        return frame


def test_visual_check_measures_opponent_and_restores_scene(tmp_path):
    stage, actor, op, original, light = fixture()
    steps = []
    rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw: steps.append(kw)))
    app = SimpleNamespace(update=lambda: None)
    result = run_visual_check(stage, app, rep, Annotator(actor, light), tmp_path,
                              light, actor.GetPrim(), op, {'x': 1, 'y': 2, 'yaw': .4})
    assert result['passed']
    assert result['difference_pixels'] == 256
    assert result['baseline_noise_pixels'] == 0
    assert set(result['lighting']) == {'1', '10', '100'}
    for name in ['native_initial.png', 'no_opponent.png', 'opponent_visible.png',
                 'opponent_difference.png', 'ego_light_1.png', 'ego_light_10.png',
                 'ego_light_100.png', 'visual_check.json']:
        assert (tmp_path/name).is_file()
    np.testing.assert_allclose(op.Get(), original)
    assert light.GetIntensityAttr().Get() == 750.
    assert actor.ComputeVisibility() == 'inherited'
    assert all(step['delta_time'] == 0 for step in steps)


def test_capture_failure_restores_scene(tmp_path):
    stage, actor, op, original, light = fixture()
    rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw: None))
    with pytest.raises(RuntimeError, match='RGB'):
        run_visual_check(stage, SimpleNamespace(update=lambda: None), rep,
                         Annotator(actor, light, fail_on=3), tmp_path,
                         light, actor.GetPrim(), op, {'x': 1, 'y': 2, 'yaw': .4})
    np.testing.assert_allclose(op.Get(), original)
    assert light.GetIntensityAttr().Get() == 750.
    assert actor.ComputeVisibility() == 'inherited'


class OcclusionAnnotator:
    def __init__(self, stage, actor, op, broken=None):
        self.stage, self.actor, self.op, self.broken = stage, actor, op, broken

    def get_data(self):
        wall = next((p for p in self.stage.Traverse() if p.GetName().startswith('OcclusionCheckWall')), None)
        wall_visible = wall and UsdGeom.Imageable(wall).ComputeVisibility() != 'invisible'
        frame = np.full((40, 40, 4), 20, np.uint8)
        if wall_visible:
            frame[:, :, :3] = [10, 20, 217]
        actor_visible = self.actor.ComputeVisibility() != 'invisible'
        front = np.asarray(self.op.Get())[3, 0] < .8
        if self.broken == 'overlay':
            front = True
        if self.broken == 'wall_always_front':
            front = False
        if actor_visible and (not wall_visible or front):
            frame[10:26, 10:26, :3] = [80, 80, 80]
        return frame


@pytest.mark.parametrize('broken, expected', [(None, True), ('overlay', False), ('wall_always_front', False)])
def test_occlusion_check_requires_both_depth_orders_and_restores_fields(tmp_path, broken, expected):
    stage, actor, op, original, light = fixture()
    field = stage.DefinePrim('/World/Track/Field', 'ParticleField3DGaussianSplat')
    visibility = UsdGeom.Imageable(field).CreateVisibilityAttr('inherited')
    app = SimpleNamespace(update=lambda: None)
    rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw: None))
    report = run_occlusion_check(stage, app, rep, OcclusionAnnotator(stage, actor, op, broken),
        tmp_path, light, actor.GetPrim(), op, {'x': 0, 'y': 0, 'yaw': 0})
    assert report['passed'] == expected
    assert report['unoccluded_car_pixels'] == 256
    assert report['wall_blue_coverage_of_car'] == 1.
    assert (tmp_path/'occlusion_car_behind_wall.png').is_file()
    assert (tmp_path/'occlusion_car_in_front.png').is_file()
    assert (tmp_path/'occlusion_mesh_only.png').is_file()
    assert (tmp_path/'occlusion_check.json').is_file()
    assert not stage.GetPrimAtPath(report['wall_prim'])
    assert visibility.Get() == 'inherited'
    np.testing.assert_allclose(op.Get(), original)
    assert light.GetIntensityAttr().Get() == 750.


def test_occlusion_check_authors_native_schema_and_restores_on_failure(tmp_path):
    stage, actor, op, original, light = fixture()
    rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw: None))
    inspected = []

    class BrokenCapture:
        def get_data(self):
            wall = next((p for p in stage.Traverse() if p.GetName().startswith('OcclusionCheckWall')), None)
            if wall:
                assert wall.GetTypeName() == 'ParticleField3DGaussianSplat'
                assert np.asarray(wall.GetAttribute('scales').Get())[0].tolist() == pytest.approx([.02, .1, .1])
                assert wall.GetAttribute('radiance:sphericalHarmonicsDegree').Get() == 0
                assert wall.GetAttribute('radiance:sphericalHarmonicsCoefficients').GetMetadata('elementSize') == 1
                inspected.append(True)
                return np.empty((0,))
            return np.full((40, 40, 4), 20, np.uint8)

    with pytest.raises(RuntimeError, match='RGB'):
        run_occlusion_check(stage, SimpleNamespace(update=lambda: None), rep, BrokenCapture(),
            tmp_path, light, actor.GetPrim(), op, {'x': 0, 'y': 0, 'yaw': 0})
    assert inspected
    assert not any(p.GetName().startswith('OcclusionCheckWall') for p in stage.Traverse())
    np.testing.assert_allclose(op.Get(), original)
    assert light.GetIntensityAttr().Get() == 750.


def test_occlusion_check_does_not_accept_noise_as_depth_evidence(tmp_path):
    stage, actor, op, original, light = fixture()
    app = SimpleNamespace(update=lambda: None)
    rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw: None))

    class Noisy(OcclusionAnnotator):
        calls = 0

        def get_data(self):
            self.calls += 1
            frame = super().get_data()
            if self.calls == 3:
                frame[10:26, 10:26, :3] = [80, 80, 80]
            return frame

    report = run_occlusion_check(stage, app, rep, Noisy(stage, actor, op), tmp_path,
        light, actor.GetPrim(), op, {'x': 0, 'y': 0, 'yaw': 0})
    assert report['noise_pixels_in_car_footprint'] == 256
    assert not report['passed']


@pytest.mark.parametrize('prior_binding, fail', [(False,False),(True,False),(True,True),(False,'cube_only')])
def test_material_comparison_restores_exact_bindings_and_metadata(tmp_path,prior_binding,fail):
    from pxr import UsdShade
    stage,actor,op,original,light=fixture()
    actor.CreateVisibilityAttr('inherited')
    field=stage.DefinePrim('/World/Track/Field','ParticleField3DGaussianSplat')
    UsdGeom.Imageable(field).CreateVisibilityAttr('inherited')
    if prior_binding:
        material=UsdShade.Material.Define(stage,'/World/OriginalMaterial')
        api=UsdShade.MaterialBindingAPI.Apply(actor.GetPrim())
        api.Bind(material,UsdShade.Tokens.weakerThanDescendants)
        api.GetDirectBindingRel().SetCustomData({'original':'preserve'})
    before=stage.GetRootLayer().ExportToString()
    rep=SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw:None))
    checked=[]
    cube_checked=[]

    class MaterialAnnotator:
        def get_data(self):
            result=np.full((40,40,4),20,np.uint8)
            cubes=[p for p in stage.Traverse() if p.GetName().startswith('MaterialCheckCube')]
            if cubes:
                bound=UsdShade.MaterialBindingAPI(cubes[0]).ComputeBoundMaterial()[0]
                assert bound.GetPrim().GetName().startswith('MaterialCheckMat')
                cube_checked.append(True)
                if fail=='cube_only' and UsdGeom.Imageable(field).ComputeVisibility()=='invisible':
                    return np.empty((0,))
            material=UsdShade.MaterialBindingAPI(actor.GetPrim()).ComputeBoundMaterial()[0]
            if material and material.GetPrim().GetName().startswith('MaterialCheckMat'):
                shader=UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath())+'/Surface'))
                assert shader.GetInput('opacity').Get()==1.
                assert shader.GetInput('roughness').Get()==1.
                assert shader.GetInput('metallic').Get()==0.
                assert shader.GetInput('ior').Get()==1.
                checked.append(True)
                if fail is True:return np.empty((0,))
                result[10:26,10:26,:3]=40
            else:
                result[10:26,10:26,:3]=120
            return result

    call=lambda:run_material_check(stage,SimpleNamespace(update=lambda:None),rep,MaterialAnnotator(),
        tmp_path,light,actor.GetPrim(),op,{'x':0,'y':0,'yaw':0})
    if fail:
        with pytest.raises(RuntimeError,match='RGB'):call()
    else:
        report=call()
        assert report['difference_pixels']==256
        assert (tmp_path/'original_car.png').is_file()
        assert (tmp_path/'matte_car.png').is_file()
        assert (tmp_path/'material_check.json').is_file()
        assert (tmp_path/'cube_scene.png').is_file()
        assert (tmp_path/'cube_only.png').is_file()
        assert cube_checked
    assert checked
    assert stage.GetRootLayer().ExportToString()==before
