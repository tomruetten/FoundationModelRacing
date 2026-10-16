"""Native Gaussian USD authoring with optional radiance-preserving world baking."""
from pathlib import Path
import json
import numpy as np


def _file_stamp(path):
    stat=Path(path).stat()
    return {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def _camera_provenance(scenario):
    checkpoint=Path(scenario['assets']['camera_checkpoint']).resolve(strict=True)
    return {'version':1,'checkpoint':str(checkpoint),'checkpoint_stamp':_file_stamp(checkpoint),
            'camera_to_world':np.asarray(scenario['camera_to_world'],dtype=float).tolist(),
            'camera_field':{'max_scale_m':scenario.get('camera_field',{}).get('max_scale_m'),
                            'bake_world':scenario.get('camera_field',{}).get('bake_world',False)},
            'exclude_world_boxes':scenario.get('exclude_world_boxes',[])}


def _camera_cache_matches(output, provenance):
    output=Path(output)
    field=output.with_name(output.stem+'_field.usd')
    try:
        metadata=json.loads(output.with_suffix('.json').read_text())
        return (metadata.get('provenance')==provenance and
                metadata.get('artifacts')=={'wrapper':_file_stamp(output),'field':_file_stamp(field)})
    except (OSError, ValueError, AttributeError):
        return False


def _camera_scale_mask(log_scales, camera_to_world, maximum):
    """Reject oversized splats by their largest physical Gaussian scale axis.

    Checkpoint scales are logarithms in the training frame. The wrapper's
    positive Sim(3) scale converts them to metres; SH and orientations remain
    untouched. This is a representation filter, not semantic actor removal.
    """
    log_scales=np.asarray(log_scales)
    if maximum is None:
        return np.ones(len(log_scales),dtype=bool)
    if isinstance(maximum,bool) or not isinstance(maximum,(int,float)) or not np.isfinite(maximum) or maximum<=0:
        raise ValueError('camera_field.max_scale_m must be positive and finite')
    scale=np.cbrt(np.linalg.det(np.asarray(camera_to_world)[:3,:3]))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('camera_to_world must have positive finite scale')
    return np.max(log_scales,axis=1)+np.log(scale)<=np.log(maximum)


def _sh_basis(directions, degree):
    """Real SH degree 0..3 in gsplat/Nerfstudio coefficient order.

    NumPy port of gsplat.cuda._torch_impl._eval_sh_bases_fast; CPU-only,
    allowing asset preparation in Isaac's Python without installing gsplat.
    """
    if degree not in (0,1,2,3):
        raise ValueError('World baking supports SH degrees 0 through 3')
    directions=np.asarray(directions,dtype=np.float64)
    x,y,z=directions.T
    result=np.empty((len(directions),(degree+1)**2),dtype=np.float64)
    result[:,0]=.2820947917738781
    if degree==0:return result
    result[:,1]=-.48860251190292*y
    result[:,2]=.48860251190292*z
    result[:,3]=-.48860251190292*x
    if degree==1:return result
    z2=z*z; c1=x*x-y*y; s1=2*x*y
    result[:,4]=.5462742152960395*s1
    result[:,5]=-1.092548430592079*z*y
    result[:,6]=.9461746957575601*z2-.3153915652525201
    result[:,7]=-1.092548430592079*z*x
    result[:,8]=.5462742152960395*c1
    if degree==2:return result
    result[:,9]=-.5900435899266435*(x*s1+y*c1)
    result[:,10]=1.445305721320277*z*s1
    result[:,11]=(-2.285228997322329*z2+.4570457994644658)*y
    result[:,12]=z*(1.865881662950577*z2-1.119528997770346)
    result[:,13]=(-2.285228997322329*z2+.4570457994644658)*x
    result[:,14]=1.445305721320277*z*c1
    result[:,15]=-.5900435899266435*(x*c1-y*s1)
    return result


def _bake_camera_world(means,log_scales,quats,dc,rest,camera_to_world):
    """Apply positive Sim(3), rotating both covariance and degree 0..3 SH.

    A deterministic 200-direction sphere determines the SH change of basis
    B(world) M = B(R.T world). It preserves directional radiance numerically
    to floating-point precision; this is not a view-dependent approximation.
    """
    transform=np.asarray(camera_to_world,dtype=np.float64)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError('World bake requires a finite affine transform')
    scale=float(np.cbrt(np.linalg.det(transform[:3,:3])))
    if scale<=0 or not np.isfinite(scale):raise ValueError('World bake requires positive scale')
    rotation=transform[:3,:3]/scale
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6):
        raise ValueError('World bake requires a proper rotation and uniform scale')
    basis_count=rest.shape[1]+1
    degree=int(round(np.sqrt(basis_count)))-1
    if degree not in (0,1,2,3) or (degree+1)**2!=basis_count:
        raise ValueError('World baking supports SH degrees 0 through 3')
    indices=np.arange(200,dtype=np.float64)
    z=1-2*(indices+.5)/200
    theta=indices*np.pi*(3-np.sqrt(5))
    radial=np.sqrt(1-z*z)
    directions=np.column_stack((radial*np.cos(theta),radial*np.sin(theta),z))
    change=np.linalg.lstsq(_sh_basis(directions,degree),
                           _sh_basis(directions@rotation,degree),rcond=None)[0]
    change[np.abs(change)<1e-12]=0.
    change[0,0]=1.
    coefficients=np.concatenate((dc[:,None,:],rest),axis=1)
    rotated=np.einsum('ij,njc->nic',change.astype(np.float32),coefficients,optimize=True)
    # Davenport eigenvector gives the rotation quaternion as xyzw; checkpoint
    # quaternions use wxyz. Hamilton left multiplication rotates each covariance.
    r=rotation
    k=np.array([[r[0,0]-r[1,1]-r[2,2],r[0,1]+r[1,0],r[0,2]+r[2,0],r[2,1]-r[1,2]],
                [r[0,1]+r[1,0],r[1,1]-r[0,0]-r[2,2],r[1,2]+r[2,1],r[0,2]-r[2,0]],
                [r[0,2]+r[2,0],r[1,2]+r[2,1],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
                [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],np.trace(r)]])/3
    _,vectors=np.linalg.eigh(k)
    rq=vectors[:,-1][[3,0,1,2]]
    q=np.asarray(quats,dtype=np.float64)
    norms=np.linalg.norm(q,axis=1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norms<1e-12):
        raise ValueError('Cannot bake a zero or nonfinite Gaussian quaternion')
    q=q/norms
    world_q=np.column_stack((rq[0]*q[:,0]-q[:,1:]@rq[1:],
        rq[0]*q[:,1:]+q[:,:1]*rq[1:]+np.cross(rq[1:],q[:,1:])))
    world_q/=np.linalg.norm(world_q,axis=1,keepdims=True)
    return ((means@transform[:3,:3].T+transform[:3,3]).astype(np.float32),
            (log_scales+np.log(scale)).astype(np.float32),world_q.astype(np.float32),
            rotated[:,0].astype(np.float32),rotated[:,1:].astype(np.float32))


def prepare_camera(scenario, force=False):
    output=Path(scenario['assets']['camera_usd'])
    if scenario.get('exclude_world_boxes'):
        raise ValueError('Exclusion boxes require matching removal from both camera and LiDAR; not yet supported')
    provenance=_camera_provenance(scenario)
    if not force and _camera_cache_matches(output,provenance): return output
    import glob
    import sys
    import isaacsim
    import torch
    from pxr import Usd, UsdGeom, Gf
    converters=glob.glob(str(Path(isaacsim.__file__).parent/'extscache/omni.kit.converter.gsplat-*/pip_prebundle'))
    if not converters: raise RuntimeError('Native Gaussian converter requires Isaac Sim 6')
    sys.path.insert(0,converters[0])
    from usd_convert_gsplat import GaussianSplatData, write_gaussian_splat_usd
    checkpoint=scenario['assets']['camera_checkpoint']
    params=torch.load(checkpoint,map_location='cpu',weights_only=False)['pipeline']
    def get(name): return params['_model.gauss_params.'+name].detach().numpy()
    means=get('means'); opacity=get('opacities').reshape(-1)
    keep=np.isfinite(means).all(axis=1) & (opacity>-12)
    scales=get('scales')
    scale_mask=_camera_scale_mask(scales,scenario['camera_to_world'],
                                  scenario.get('camera_field',{}).get('max_scale_m'))
    keep &= scale_mask
    raw=output.with_name(output.stem+'_field.usd'); output.parent.mkdir(parents=True,exist_ok=True)
    rest=get('features_rest')[keep]
    positions,scales,rotations,dc=means[keep],scales[keep],get('quats')[keep],get('features_dc')[keep]
    parent_transform=np.asarray(scenario['camera_to_world'])
    if scenario.get('camera_field',{}).get('bake_world',False):
        positions,scales,rotations,dc,rest=_bake_camera_world(positions,scales,rotations,dc,rest,parent_transform)
        parent_transform=np.eye(4)
    data=GaussianSplatData(positions=positions,scales=scales,
        rotations=rotations,opacities=opacity[keep],f_dc=dc,
        f_rest=np.ascontiguousarray(rest.transpose(0,2,1)).reshape(len(rest),-1),count=int(keep.sum()))
    write_gaussian_splat_usd(data,str(raw),source_file=checkpoint,prim_name='Field',up_axis='Z')
    stage=Usd.Stage.CreateNew(str(output)); world_prim=UsdGeom.Xform.Define(stage,'/World')
    stage.SetDefaultPrim(world_prim.GetPrim()); UsdGeom.SetStageUpAxis(stage,'Z'); UsdGeom.SetStageMetersPerUnit(stage,1.)
    track=UsdGeom.Xform.Define(stage,'/World/Track')
    track.AddTransformOp().Set(Gf.Matrix4d(parent_transform.T.tolist()))
    stage.DefinePrim('/World/Track/Field').GetReferences().AddReference(raw.name)
    stage.GetRootLayer().Save()
    metadata={'provenance':provenance,'gaussians':int(keep.sum()),
              'source_gaussians':len(means),'filtered_by_scale':int((~scale_mask).sum()),
              'artifacts':{'wrapper':_file_stamp(output),'field':_file_stamp(raw)}}
    sidecar=output.with_suffix('.json')
    temporary=sidecar.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata,indent=2))
    temporary.replace(sidecar)
    return output


def add_vehicle(stage, path, asset):
    """Reference detailed mesh, normalize uniformly to 42cm length and ground it."""
    from pxr import Usd, UsdGeom, Gf
    root=UsdGeom.Xform.Define(stage,path)
    pose_op=root.AddTransformOp()
    model=stage.DefinePrim(path+'/Model','Xform'); model.GetReferences().AddReference(str(asset))
    xf=UsdGeom.Xformable(model); xf.ClearXformOpOrder()
    # Referenced file is Y-up, centimeters. Explicit mapping is required because
    # USD references do not automatically reconcile unit/up-axis metadata.
    source=Usd.Stage.Open(str(asset)); up=UsdGeom.GetStageUpAxis(source)
    rotation=Gf.Matrix4d(1.)
    if up=='Y': rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),90.))
    op=xf.AddTransformOp(); op.Set(rotation)
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(model).ComputeAlignedRange()
    lo=np.array(bounds.GetMin()); hi=np.array(bounds.GetMax())
    scale=.42/(hi[0]-lo[0])
    matrix=np.asarray(rotation).T.copy(); matrix[:3,:3]*=scale
    matrix[:3,3]=-scale*np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
    op.Set(Gf.Matrix4d(matrix.T.tolist()))
    return root.GetPrim(),pose_op


def mesh_triangles(stage, actor):
    """Flatten rendered USD meshes in actor coordinates for the LiDAR BVH."""
    from pxr import Usd, UsdGeom
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(actor)).T)
    result=[]
    for prim in Usd.PrimRange(actor):
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh=UsdGeom.Mesh(prim)
        if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible': continue
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        if not len(points): continue
        transform=inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points=points@transform[:3,:3].T+transform[:3,3]
        counts=mesh.GetFaceVertexCountsAttr().Get(); indices=mesh.GetFaceVertexIndicesAttr().Get(); start=0
        for count in counts:
            face=indices[start:start+count]; start+=count
            for j in range(1,count-1):result.append(points[[face[0],face[j],face[j+1]]])
    if not result: raise ValueError('Vehicle asset contains no visible mesh triangles')
    return np.asarray(result,dtype=np.float32)


if __name__=='__main__':
    import argparse
    from .config import load_scenario
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario');parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    print(prepare_camera(load_scenario(args.scenario,check_assets=False),args.force),flush=True)
