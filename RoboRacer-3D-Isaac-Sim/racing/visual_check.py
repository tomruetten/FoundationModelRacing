"""Bounded native-render appearance checks, run inside the existing Isaac app.

The appearance check compares opponent visibility and lighting. A separate
controlled Gaussian-wall fixture checks both depth orders; neither validates
learned-track calibration or visibility at every point on the race course.
"""
import json
from pathlib import Path

import numpy as np

from .config import pose_matrix


def run_visual_check(stage, app, rep, annotator, output, light,
                     opponent_prim, opponent_op, ego_world_dict):
    """Capture a stationary opponent 1.3 m ahead, restoring all scene edits.

    Returns the same measurements written to ``visual_check.json``. A failed
    appearance check is reported as ``passed=False``; malformed RGB output is
    a runtime error. Neither rendering nor this check advances simulation time.
    """
    from PIL import Image
    from pxr import Gf, UsdGeom

    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    intensity_attr=light.GetIntensityAttr()
    visibility_attr=UsdGeom.Imageable(opponent_prim).GetVisibilityAttr()
    original_intensity=intensity_attr.Get()
    original_pose=opponent_op.Get()
    original_visibility=visibility_attr.Get()
    visibility_authored=visibility_attr.HasAuthoredValueOpinion()
    ahead=pose_matrix(ego_world_dict)
    ahead[:3,3] += ahead[:3,0]*1.3
    threshold=16

    def capture():
        # Allow Hydra changes and temporal rendering history to settle. The
        # final annotator array is copied before another render can reuse it.
        for _ in range(4):
            app.update()
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
        frame=np.asarray(annotator.get_data())
        if frame.ndim!=3 or frame.shape[2]<3 or not frame.shape[0] or not frame.shape[1] or frame.dtype!=np.uint8:
            raise RuntimeError('Visual check requires a nonempty uint8 native RGB frame')
        return frame[:,:,:3].copy()

    def save(name,frame):
        Image.fromarray(frame).save(output/name)

    def difference(first,second):
        if first.shape!=second.shape:
            raise RuntimeError('Native RGB shape changed during visual check')
        delta=np.abs(first.astype(np.int16)-second.astype(np.int16))
        return delta,np.max(delta,axis=2)>threshold

    report={'pixel_difference_threshold':threshold,'opponent_distance_ahead_m':1.3,
            'native_gaussian_fields':sum(p.GetTypeName()=='ParticleField3DGaussianSplat' for p in stage.Traverse()),
            'lighting':{},'scope':'Native opponent visibility and lighting only; Gaussian/mesh depth occlusion and calibration are not proven.'}
    try:
        save('native_initial.png',capture())
        opponent_op.Set(Gf.Matrix4d(ahead.T.tolist()))
        for intensity in (1,10,100):
            intensity_attr.Set(float(intensity))
            visibility_attr.Set(UsdGeom.Tokens.invisible)
            hidden=capture()
            if intensity==1:
                hidden_repeat=capture()
                _,noise=difference(hidden,hidden_repeat)
                report['baseline_noise_pixels']=int(noise.sum())
                hidden=hidden_repeat
            visibility_attr.Set(UsdGeom.Tokens.inherited)
            visible=capture()
            delta,mask=difference(hidden,visible)
            count=int(mask.sum())
            pixels=visible[mask]
            report['lighting'][str(intensity)]={
                'difference_pixels':count,
                'changed_region_mean_rgb':pixels.mean(axis=0).tolist() if count else None,
                'changed_region_white_fraction':float(np.all(pixels>=250,axis=1).mean()) if count else None,
            }
            save(f'ego_light_{intensity}.png',visible)
            save(f'ego_light_{intensity}_hidden.png',hidden)
            if intensity==1:
                save('no_opponent.png',hidden)
                save('opponent_visible.png',visible)
                save('opponent_difference.png',np.clip(delta*4,0,255).astype(np.uint8))
                report['difference_pixels']=count
                report['minimum_changed_pixels']=max(64,int(mask.size*.001))
                report['passed']=bool(count>report['minimum_changed_pixels'] and
                                      count>report['baseline_noise_pixels']*2)
        (output/'visual_check.json').write_text(json.dumps(report,indent=2))
        return report
    finally:
        opponent_op.Set(original_pose)
        intensity_attr.Set(original_intensity)
        if visibility_authored:
            visibility_attr.Set(original_visibility)
        else:
            visibility_attr.Clear()
        for _ in range(2):
            app.update()


def run_material_check(stage, app, rep, annotator, output, light,
                       opponent_prim, opponent_op, ego_world_dict):
    """Compare original and opaque matte materials with the full track visible.

    A root binding stronger than descendants overrides the authored materials
    for all render purposes. Exact local binding specs and applied-schema list
    edits are saved/restored, including relationship targets and metadata.
    """
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    material_path='/World/MaterialCheckMat'
    suffix=0
    while stage.GetPrimAtPath(material_path):
        suffix+=1
        material_path=f'/World/MaterialCheckMat_{suffix}'
    cube_path='/World/MaterialCheckCube'
    suffix=0
    while stage.GetPrimAtPath(cube_path):
        suffix+=1
        cube_path=f'/World/MaterialCheckCube_{suffix}'
    edit_target=stage.GetEditTarget()
    layer=edit_target.GetLayer()
    spec_path=edit_target.MapToSpecPath(opponent_prim.GetPath())
    spec=layer.GetPrimAtPath(spec_path)
    old_api=spec.GetInfo('apiSchemas') if spec and spec.HasInfo('apiSchemas') else None
    bindings=[name for name in spec.properties.keys() if name.startswith('material:binding')] if spec else []
    visibility_local=bool(spec and 'visibility' in spec.properties)
    backup=Sdf.Layer.CreateAnonymous('material-binding-backup')
    backup_path=Sdf.Path('/Bindings')
    Sdf.CreatePrimInLayer(backup,backup_path)
    for name in bindings:
        Sdf.CopySpec(layer,spec_path.AppendProperty(name),backup,backup_path.AppendProperty(name))
    original_pose=opponent_op.Get()
    visibility=UsdGeom.Imageable(opponent_prim).GetVisibilityAttr()
    original_visibility=visibility.Get()
    saved_fields=[]

    def capture(name):
        for _ in range(8):
            app.update()
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
        image=np.asarray(annotator.get_data())
        if image.ndim!=3 or image.shape[2]<3 or not image.shape[0] or not image.shape[1] or image.dtype!=np.uint8:
            raise RuntimeError('Material check requires a nonempty uint8 native RGB frame')
        image=image[:,:,:3].copy()
        Image.fromarray(image).save(output/name)
        return image

    try:
        pose=pose_matrix(ego_world_dict)
        pose[:3,3]+=pose[:3,0]*1.3
        opponent_op.Set(Gf.Matrix4d(pose.T.tolist()))
        visibility.Set(UsdGeom.Tokens.inherited)
        original=capture('original_car.png')
        material=UsdShade.Material.Define(stage,material_path)
        shader=UsdShade.Shader.Define(stage,material_path+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        for name,value in [('roughness',1.),('metallic',0.),('opacity',1.),('ior',1.),('clearcoat',0.)]:
            shader.CreateInput(name,Sdf.ValueTypeNames.Float).Set(value)
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08,.08,.08))
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
        shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        api=UsdShade.MaterialBindingAPI.Apply(opponent_prim)
        for purpose in (UsdShade.Tokens.allPurpose,UsdShade.Tokens.full,UsdShade.Tokens.preview):
            api.Bind(material,UsdShade.Tokens.strongerThanDescendants,purpose)
        matte=capture('matte_car.png')
        if matte.shape!=original.shape:
            raise RuntimeError('Native RGB shape changed during material check')
        difference=np.abs(matte.astype(np.int16)-original.astype(np.int16))
        count=int((np.max(difference,axis=2)>16).sum())
        Image.fromarray(np.clip(difference*4,0,255).astype(np.uint8)).save(output/'material_difference.png')
        report={'difference_pixels':count,'pixel_difference_threshold':16,
                'light_intensity':float(light.GetIntensityAttr().Get()),'opponent_distance_ahead_m':1.3,
                'material':{'shader':'UsdPreviewSurface','diffuseColor':[.08,.08,.08],
                            'roughness':1.,'metallic':0.,'opacity':1.,'ior':1.,'specularColor':[0,0,0]},
                'scope':'Material isolation comparison only; inspect original_car.png and matte_car.png for reflections or residual Gaussian compositing.'}
        # A directly bound primitive removes the vehicle's reference/material
        # hierarchy from the experiment while preserving camera and location.
        visibility.Set(UsdGeom.Tokens.invisible)
        full_background=capture('cube_scene_background.png')
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(.8,.02,.8))
        cube=UsdGeom.Cube.Define(stage,cube_path)
        cube.CreateSizeAttr(1.)
        cube_pose=pose.copy()
        cube_pose[:3,3]+=np.array([0.,0.,.187/2])
        cube_pose[:3,:3]=cube_pose[:3,:3]@np.diag([.42,.23,.187])
        cube.AddTransformOp().Set(Gf.Matrix4d(cube_pose.T.tolist()))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,UsdShade.Tokens.strongerThanDescendants)
        cube_scene=capture('cube_scene.png')
        for prim in stage.Traverse():
            if prim.GetTypeName()!='ParticleField3DGaussianSplat':continue
            field_path=edit_target.MapToSpecPath(prim.GetPath())
            field_spec=layer.GetPrimAtPath(field_path)
            local=bool(field_spec and 'visibility' in field_spec.properties)
            saved_path=Sdf.Path(f'/Field{len(saved_fields)}')
            Sdf.CreatePrimInLayer(backup,saved_path)
            if local:
                Sdf.CopySpec(layer,field_path.AppendProperty('visibility'),backup,saved_path.AppendProperty('visibility'))
            saved_fields.append((prim,field_path,local,saved_path))
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        cube_only=capture('cube_only.png')
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        empty=capture('cube_only_background.png')
        footprint=np.max(np.abs(cube_only.astype(np.int16)-empty.astype(np.int16)),axis=2)>16
        scene_delta=np.abs(cube_scene.astype(np.int16)-full_background.astype(np.int16))
        scene_only_delta=np.abs(cube_scene.astype(np.int16)-cube_only.astype(np.int16))
        footprint_count=int(footprint.sum())
        report['cube']={
            'dimensions_m':[.42,.23,.187],'diffuseColor':[.8,.02,.8],
            'footprint_pixels':footprint_count,
            'scene_changed_pixels_in_footprint':int(((scene_delta.max(axis=2)>16)&footprint).sum()),
            'scene_vs_only_changed_pixels_in_footprint':int(((scene_only_delta.max(axis=2)>16)&footprint).sum()),
            'scene_mean_rgb_in_footprint':cube_scene[footprint].mean(axis=0).tolist() if footprint_count else None,
            'only_mean_rgb_in_footprint':cube_only[footprint].mean(axis=0).tolist() if footprint_count else None,
            'scope':'Directly bound opaque cube with/without native field; color differences alone do not establish physical opacity.'}
        (output/'material_check.json').write_text(json.dumps(report,indent=2))
        return report
    finally:
        for prim,field_path,local,saved_path in reversed(saved_fields):
            prim.RemoveProperty('visibility')
            if local:
                Sdf.CopySpec(backup,saved_path.AppendProperty('visibility'),layer,field_path.AppendProperty('visibility'))
        if stage.GetPrimAtPath(cube_path):stage.RemovePrim(cube_path)
        spec=layer.GetPrimAtPath(spec_path)
        if spec:
            for name in list(spec.properties.keys()):
                if name.startswith('material:binding'):
                    opponent_prim.RemoveProperty(name)
            for name in bindings:
                Sdf.CopySpec(backup,backup_path.AppendProperty(name),layer,spec_path.AppendProperty(name))
            if old_api is None:
                spec.ClearInfo('apiSchemas')
            else:
                spec.SetInfo('apiSchemas',old_api)
        if stage.GetPrimAtPath(material_path):stage.RemovePrim(material_path)
        opponent_op.Set(original_pose)
        if visibility_local:
            visibility.Set(original_visibility)
        else:
            opponent_prim.RemoveProperty('visibility')
        for _ in range(2):app.update()


def _gaussian_wall(stage, path, ego):
    """Author the exact native converter schema, with linear scale/opacity."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    base=pose_matrix(ego)
    lateral,height=np.meshgrid(np.linspace(-1,1,26),np.linspace(-1,1,26))
    local=np.column_stack((np.full(lateral.size,.8),lateral.ravel(),height.ravel()))
    positions=(local@base[:3,:3].T+base[:3,3]).astype(np.float32)
    count=len(positions)
    prim=stage.DefinePrim(path,'ParticleField3DGaussianSplat')

    def attr(name, kind, value):
        result=prim.CreateAttribute(name,kind,custom=False)
        result.Set(value)
        return result

    attr('positions',Sdf.ValueTypeNames.Point3fArray,Vt.Vec3fArray.FromNumpy(positions))
    attr('scales',Sdf.ValueTypeNames.Float3Array,Vt.Vec3fArray([Gf.Vec3f(.02,.1,.1)]*count))
    yaw=float(ego['yaw'])
    quaternion=Gf.Quatf(float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2)))
    attr('orientations',Sdf.ValueTypeNames.QuatfArray,Vt.QuatfArray([quaternion]*count))
    attr('opacities',Sdf.ValueTypeNames.FloatArray,Vt.FloatArray([1.]*count))
    attr('radiance:sphericalHarmonicsDegree',Sdf.ValueTypeNames.Int,0)
    color=np.array([.04,.08,.85])
    coefficient=(color-.5)/.28209479177387814
    sh=attr('radiance:sphericalHarmonicsCoefficients',Sdf.ValueTypeNames.Float3Array,
            Vt.Vec3fArray([Gf.Vec3f(*coefficient.tolist())]*count))
    primvar=UsdGeom.Primvar(sh)
    primvar.SetElementSize(1)
    primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar('displayColor',Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray([Gf.Vec3f(*color.tolist())]*count))
    attr('extent',Sdf.ValueTypeNames.Float3Array,
         Vt.Vec3fArray([Gf.Vec3f(*(positions.min(0)-.3).tolist()),Gf.Vec3f(*(positions.max(0)+.3).tolist())]))
    return prim


def run_occlusion_check(stage, app, rep, annotator, output, light,
                        opponent_prim, opponent_op, ego_world_dict):
    """Test a mesh on both sides of a temporary native blue Gaussian wall.

    Existing learned Gaussian fields are hidden for this controlled fixture,
    so their calibration/large splats cannot invalidate the depth experiment.
    The initial mesh-only image also isolates material opacity from Gaussian
    compositing. All field visibility, actor pose and lighting are restored.
    """
    from PIL import Image
    from pxr import Gf, UsdGeom

    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    wall_path='/World/OcclusionCheckWall'
    suffix=0
    while stage.GetPrimAtPath(wall_path):
        suffix+=1
        wall_path=f'/World/OcclusionCheckWall_{suffix}'
    original_pose=opponent_op.Get()
    intensity=light.GetIntensityAttr()
    original_intensity=intensity.Get()
    visibility=UsdGeom.Imageable(opponent_prim).GetVisibilityAttr()
    saved_visibility=[]
    threshold=16
    base=pose_matrix(ego_world_dict)

    def remember(attr):
        saved_visibility.append((attr,attr.HasAuthoredValueOpinion(),attr.Get()))

    def restore(attr,authored,value):
        if authored:
            attr.Set(value)
        else:
            attr.Clear()

    def car_distance(distance):
        transform=base.copy()
        transform[:3,3]+=base[:3,0]*distance
        opponent_op.Set(Gf.Matrix4d(transform.T.tolist()))

    def capture(name):
        for _ in range(6):
            app.update()
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
        data=np.asarray(annotator.get_data())
        if data.ndim!=3 or data.shape[2]<3 or not data.shape[0] or not data.shape[1] or data.dtype!=np.uint8:
            raise RuntimeError('Occlusion check requires a nonempty uint8 native RGB frame')
        data=data[:,:,:3].copy()
        Image.fromarray(data).save(output/name)
        return data

    def delta(a,b):
        if a.shape!=b.shape:
            raise RuntimeError('Native RGB shape changed during occlusion check')
        return np.abs(a.astype(np.int16)-b.astype(np.int16))

    def changed(a,b):
        return np.max(delta(a,b),axis=2)>threshold

    remember(visibility)
    try:
        for prim in stage.Traverse():
            if prim.GetTypeName()=='ParticleField3DGaussianSplat':
                attr=UsdGeom.Imageable(prim).GetVisibilityAttr()
                remember(attr)
                attr.Set(UsdGeom.Tokens.invisible)
        intensity.Set(1.)
        car_distance(1.3)
        visibility.Set(UsdGeom.Tokens.inherited)
        clear_car=capture('occlusion_mesh_only.png')
        visibility.Set(UsdGeom.Tokens.invisible)
        clear_background=capture('occlusion_no_wall_no_car.png')
        car_mask=changed(clear_car,clear_background)
        wall=_gaussian_wall(stage,wall_path,ego_world_dict)
        wall_visibility=UsdGeom.Imageable(wall).GetVisibilityAttr()
        wall_only=capture('occlusion_wall_only.png')
        wall_repeat=capture('occlusion_wall_noise_repeat.png')
        noise=changed(wall_only,wall_repeat)
        visibility.Set(UsdGeom.Tokens.inherited)
        behind=capture('occlusion_car_behind_wall.png')
        behind_changed=changed(wall_repeat,behind)
        car_distance(.6)
        front=capture('occlusion_car_in_front.png')
        wall_visibility.Set(UsdGeom.Tokens.invisible)
        clear_front=capture('occlusion_front_without_wall.png')
        front_mask=changed(clear_front,clear_background)
        front_changed=changed(front,wall_repeat)
        wall_rgb=wall_repeat.astype(np.int16)
        blue=(wall_rgb[:,:,2]>wall_rgb[:,:,0]+40)&(wall_rgb[:,:,2]>wall_rgb[:,:,1]+40)&(wall_rgb[:,:,2]>90)
        car_pixels=int(car_mask.sum())
        front_pixels=int(front_mask.sum())
        noise_pixels=int((noise&car_mask).sum())
        behind_pixels=int((behind_changed&car_mask).sum())
        front_signal=int((front_changed&front_mask&blue).sum())
        coverage=float((blue&car_mask).sum()/max(car_pixels,1))
        minimum=max(64,int(car_mask.size*.001))
        allowed=max(8,int(.02*car_pixels),3*noise_pixels)
        report={
            'wall_prim':wall_path,'wall_gaussians':676,
            'wall_distance_ahead_m':.8,'car_behind_distance_m':1.3,'car_front_distance_m':.6,
            'pixel_difference_threshold':threshold,'minimum_car_pixels':minimum,
            'unoccluded_car_pixels':car_pixels,'wall_blue_pixels':int(blue.sum()),
            'wall_blue_coverage_of_car':coverage,'noise_pixels_in_car_footprint':noise_pixels,
            'noise_baseline_passed':bool(noise_pixels<=max(8,int(.02*car_pixels))),
            'behind_car_residual_pixels':behind_pixels,'allowed_behind_residual_pixels':allowed,
            'front_car_footprint_pixels':front_pixels,'front_car_replaces_wall_pixels':front_signal,
            'behind_passed':bool(car_pixels>minimum and coverage>=.95 and behind_pixels<=allowed),
            'front_passed':bool(front_pixels>minimum and front_signal>max(minimum,.5*front_pixels)),
            'scope':'Controlled native Gaussian/mesh depth ordering at two poses; learned-track depth and calibration are not validated.'}
        report['passed']=report['noise_baseline_passed'] and report['behind_passed'] and report['front_passed']
        Image.fromarray(np.clip(delta(wall_repeat,behind)*4,0,255).astype(np.uint8)).save(output/'occlusion_behind_difference.png')
        Image.fromarray(np.clip(delta(wall_repeat,front)*4,0,255).astype(np.uint8)).save(output/'occlusion_front_difference.png')
        (output/'occlusion_check.json').write_text(json.dumps(report,indent=2))
        return report
    finally:
        if stage.GetPrimAtPath(wall_path):
            stage.RemovePrim(wall_path)
        opponent_op.Set(original_pose)
        intensity.Set(original_intensity)
        for attr,authored,value in reversed(saved_visibility):
            restore(attr,authored,value)
        for _ in range(2):
            app.update()
