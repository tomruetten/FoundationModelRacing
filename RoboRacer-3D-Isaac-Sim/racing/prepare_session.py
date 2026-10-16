"""Reproduce session 142014's provisional camera/LiDAR metric alignment.

Pairs COLMAP camera centers with timestamp-interpolated FAST-LIO odometry and
known camera lever arm. Uses the complete normalized training frame, including
Nerfstudio's dataparser scale. The residual is retained: these captures do not
support an exact global alignment. Corridor boundaries are a driving proxy,
not measured track walls. All inputs are read-only; only --output is written.
"""
import argparse
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/media/alienware_felix/T7/roboracer_data/27.08新赛道和Dynamik')


def fit_similarity(source, target):
    """Least-squares proper Sim(3), with per-correspondence metric residuals."""
    source=np.asarray(source,dtype=float);target=np.asarray(target,dtype=float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<4:
        raise ValueError('Need at least four matched XYZ centers')
    x=source-source.mean(0);y=target-target.mean(0)
    if np.linalg.matrix_rank(x)<2: raise ValueError('Trajectory must span at least a plane')
    u,d,v=np.linalg.svd(y.T@x/len(x))
    if np.linalg.matrix_rank(x)==3 and np.linalg.det(u@v)<0:
        raise ValueError('Correspondences imply a reflection, not a proper Sim3')
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@v)
    rotation=u@correction@v
    scale=np.sum(d*np.diag(correction))/np.mean(np.sum(x*x,axis=1))
    transform=np.eye(4);transform[:3,:3]=scale*rotation
    transform[:3,3]=target.mean(0)-scale*rotation@source.mean(0)
    residual=np.linalg.norm(source@transform[:3,:3].T+transform[:3,3]-target,axis=1)
    return transform,residual


def closed_route(positions, spacing=.12):
    """First return near start after >=10m; smooth measured closure discrepancy.

    Fails if no credible return exists. The return must approach within 0.6m
    with forward direction agreeing, avoiding premature out-and-back closure.
    """
    xy=np.asarray(positions)[:,:2]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    distance=np.linalg.norm(xy-xy[0],axis=1)
    forward=xy[3]-xy[0];forward/=np.linalg.norm(forward)
    tangent=np.gradient(xy,axis=0)
    alignment=tangent@forward/np.maximum(np.linalg.norm(tangent,axis=1),1e-9)
    candidates=np.flatnonzero((arc>10)&(distance<.6)&(alignment>.7))
    if not len(candidates): raise ValueError('No supported first lap closure in trajectory')
    first=int(candidates[0]);window=np.arange(first,min(first+20,len(xy)))
    end=int(window[np.argmin(distance[window])])
    points=np.vstack((xy[:end+1],xy[0]))
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    sample=np.linspace(0,length[-1],max(100,int(length[-1]/spacing)),endpoint=False)
    route=np.column_stack([np.interp(sample,length,points[:,j]) for j in range(2)])
    # 0.3m periodic smoothing handles the measured seam without an endpoint stop.
    route=gaussian_filter1d(route,sigma=.3/spacing,axis=0,mode='wrap')
    return route,end


def _load_odometry(path):
    reader=REPO/'data/local/pipeline_scripts_1707/cdr_reader.py'
    spec=importlib.util.spec_from_file_location('_race_cdr',reader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as conn:
        rows=conn.execute("SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
                          "WHERE t.name='/Odometry' ORDER BY m.timestamp").fetchall()
    parsed=[module.parse_odometry_pose(bytes(row[0])) for row in rows]
    times=np.array([row[0] for row in parsed]);positions=np.array([row[1] for row in parsed])
    quaternions=np.array([row[2] for row in parsed])
    times,indices=np.unique(times,return_index=True)
    return times,positions[indices],quaternions[indices]


def prepare(root, vehicle_usd, output):
    from shapely.geometry import LineString
    original=root/'3dgs session_20260827_142014'
    noflare=root/'3dgs session_20260827_142014_noflare'
    training=noflare/'outputs/colmap_dataset_full/splatfacto/2026-08-28_234736'
    transforms_path=noflare/'dataset/transforms_colmap_full.json'
    manifest_path=original/'dataset/frame_manifest_full.json'
    odometry_path=original/'odometry_bag/recording/recording_0.db3'
    training_camera=json.loads(transforms_path.read_text())
    frames=sorted(training_camera['frames'],key=lambda f:f['file_path'])
    lookup={row['output_frame']:row['unix_timestamp_sec']
            for row in json.loads(manifest_path.read_text())['frames']}
    timestamps=np.array([lookup[Path(frame['file_path']).name] for frame in frames])
    raw=np.array([frame['transform_matrix'] for frame in frames])
    dp=json.loads((training/'dataparser_transforms.json').read_text());dpt=np.array(dp['transform'])
    source=dp['scale']*(raw[:,:3,3]@dpt[:,:3].T+dpt[:,3])
    times,positions,quaternions=_load_odometry(odometry_path)
    valid=(timestamps>=times[0])&(timestamps<=times[-1])
    timestamps=timestamps[valid];source=source[valid]
    rotation=Slerp(times,Rotation.from_quat(quaternions))(timestamps).as_matrix()
    lidar=np.column_stack([np.interp(timestamps,times,positions[:,j]) for j in range(3)])
    calibration_path=REPO/'data/local/calibration_summary/camera_lidar_extrinsic_T_cl/T_cl_2026-07-14.json'
    calibration=json.loads(calibration_path.read_text())
    camera_offset=np.array(calibration['cam_origin_in_lidar'])
    target=lidar+np.einsum('nij,j->ni',rotation,camera_offset)
    transform,residual=fit_similarity(source,target)
    base=lidar-np.einsum('nij,j->ni',rotation,[.27,0,.11])
    path,end=closed_route(base)
    path_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    opponent_index=int(np.argmin(abs(path_arc-1.5)))
    corridor=LineString(np.vstack((path,path[0]))).buffer(.8).simplify(.025,preserve_topology=True)
    if corridor.geom_type!='Polygon' or len(corridor.interiors)!=1:
        raise ValueError('First lap corridor is not a simple closed driving loop')
    def spawn(index):
        direction=path[(index+1)%len(path)]-path[index]
        return {'x':float(path[index,0]),'y':float(path[index,1]),
                'yaw':float(np.arctan2(direction[1],direction[0]))}
    # Camera mount uses ROS optical RDF; runtime converts to OpenGL for USD.
    camera_rotation=np.array(calibration['T_cl_result']['rotation_matrix']).T
    output=Path(output).resolve()
    def repo_asset(path):
        return os.path.relpath(path,output.parent)
    scale_x=640/training_camera['w'];scale_y=400/training_camera['h']
    result={
        'name':'Session 142014 — provisional camera/LiDAR alignment',
        'assets':{
            'camera_checkpoint':str(training/'nerfstudio_models/step-000079999.ckpt'),
            'camera_usd':repo_asset(REPO/'outputs/racing142014/track.usd'),
            'lidar_checkpoint':str(original/'lidar_3dgs_experiments/lidar_3dgs_v8_5cmvoxel/best.pt'),
            'dynamics_checkpoint':repo_asset(REPO/'dynamics/deployed/trained_models/gru_run30_best.pth'),
            'vehicle_usd':str(Path(vehicle_usd).resolve())},
        'camera':{'width':640,'height':400,'fx':training_camera['fl_x']*scale_x,
                  'fy':training_camera['fl_y']*scale_y,'cx':training_camera['cx']*scale_x,
                  'cy':training_camera['cy']*scale_y},
        'camera_intrinsics_source':{'path':str(transforms_path),
                                    'training_width':training_camera['w'],
                                    'training_height':training_camera['h']},
        'camera_to_world':transform.tolist(),
        'world_base_z':float(np.median(base[:end+1,2])),
        'reference_path':path.tolist(),
        'outer_boundary':np.asarray(corridor.exterior.coords)[:-1].tolist(),
        'inner_boundaries':[np.asarray(ring.coords)[:-1].tolist() for ring in corridor.interiors],
        'spawns':{'ego':spawn(0),'opponent':spawn(opponent_index)},
        'rates':{'dynamics':50,'camera':20,'lidar':10,'viewport':30},
        'sensors':{'camera':{'translation':(camera_offset+[.27,0,.11]).tolist(),
                             'quaternion':Rotation.from_matrix(camera_rotation).as_quat().tolist()},
                   'lidar':{'translation':[.27,0,.11],'quaternion':[0,0,0,1]}},
        'calibration':{'status':'provisional','method':'timestamp-paired camera centers; proper least-squares Sim3',
                       'correspondences':len(source),'scale_m_per_training_unit':float(np.cbrt(np.linalg.det(transform[:3,:3]))),
                       'mean_residual_m':float(residual.mean()),'median_residual_m':float(np.median(residual)),
                       'p95_residual_m':float(np.quantile(residual,.95)),'max_residual_m':float(residual.max()),
                       'time_offset_s':0.,'transforms':str(transforms_path),'manifest':str(manifest_path),
                       'odometry':str(odometry_path),'extrinsic':str(calibration_path),
                       'limitation':'Global rigid alignment cannot explain the observed reconstruction/odometry disagreement.'},
        'route_provenance':{'source':'interpolated FAST-LIO base trajectory, first closed lap',
                            'first_timestamp':float(timestamps[0]),'last_timestamp':float(timestamps[end]),
                            'sample_end_index':end,'raw_closure_gap_m':float(np.linalg.norm(base[end,:2]-base[0,:2])),
                            'periodic_smoothing_sigma_m':.3,'corridor_halfwidth_m':.8,
                            'boundary_kind':'synthetic corridor proxy; not measured physical walls'},
    }
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--vehicle-usd',required=True)
    parser.add_argument('--output',type=Path,default=REPO/'configs/racing_142014.json')
    args=parser.parse_args();result=prepare(args.session_root,args.vehicle_usd,args.output)
    print(json.dumps({'output':str(args.output),'calibration':result['calibration'],
                      'route':result['route_provenance']},indent=2))


if __name__=='__main__': main()
