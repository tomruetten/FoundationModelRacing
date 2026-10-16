"""Scenario validation and frame/rate conventions, independent of Isaac."""
import json
import math
import os
from pathlib import Path
import numpy as np


def pose_matrix(state):
    values = state if isinstance(state, dict) else vars(state)
    yaw = values['yaw']; c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c,-s,0,values['x']],[s,c,0,values['y']],
                     [0,0,1,values.get('z',0.)],[0,0,0,1]], dtype=float)


def quaternion_matrix(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('Quaternion must be finite nonzero xyzw')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def sensor_pose(state, mount):
    transform=np.eye(4)
    transform[:3,:3]=quaternion_matrix(mount['quaternion'])
    transform[:3,3]=mount['translation']
    return pose_matrix(state)@transform


class Schedule:
    """Integer phase accumulator: 20Hz and 30Hz remain exact on a 50Hz clock.

    Actual sample intervals are quantized to dynamics ticks (camera 40/60ms).
    Each sample must carry its actual snapshot time, never an invented 50ms.
    """
    def __init__(self, dynamics_hz, rates):
        self.base=int(dynamics_hz); self.rates=dict(rates)
        if self.base!=dynamics_hz or self.base<=0 or any(
            int(r)!=r or not 0<r<=self.base for r in self.rates.values()):
            raise ValueError('Sensor rate must be an integer in (0,dynamics rate]')
        self.reset()

    def reset(self):
        self.previous=0

    def due(self, step):
        if step<self.previous: raise ValueError('Reset schedule before resetting time')
        due={k for k,r in self.rates.items() if step*r//self.base>self.previous*r//self.base}
        self.previous=step
        return due


def load_scenario(path, check_assets=True):
    path=Path(path).resolve(); data=json.loads(path.read_text())
    required={'name','assets','reference_path','spawns','rates','sensors'}
    if required-data.keys(): raise ValueError(f'Missing scenario fields: {required-data.keys()}')
    rates=data['rates']; Schedule(rates['dynamics'],{k:rates[k] for k in ('camera','lidar','viewport')})
    points=np.asarray(data['reference_path'],dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('reference_path must contain at least three finite XY points')
    if set(data['spawns'])!={'ego','opponent'}: raise ValueError('Scenario needs ego and opponent spawns')
    for name,state in data['spawns'].items():
        if not isinstance(state,dict) or not {'x','y','yaw'}<=state.keys():
            raise ValueError(f'{name} spawn must include x, y, and yaw')
        if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
               for value in state.values()):
            raise ValueError(f'{name} spawn values must be finite numbers')
    if not isinstance(data['sensors'],dict) or not {'camera','lidar'}<=data['sensors'].keys():
        raise ValueError('Scenario needs camera and lidar sensors')
    for mount in data['sensors'].values():
        if not isinstance(mount,dict) or not {'quaternion','translation'}<=mount.keys():
            raise ValueError('Sensor mount needs quaternion and translation')
        quaternion_matrix(mount['quaternion'])
        if np.asarray(mount['translation']).shape!=(3,) or not np.isfinite(mount['translation']).all():
            raise ValueError('Sensor translation must be a finite XYZ vector')
    if data.get('exclude_world_boxes'):
        raise ValueError('Exclusion boxes require matching removal from both camera and LiDAR; not yet supported')
    camera_field=data.get('camera_field',{})
    if not isinstance(camera_field,dict):
        raise ValueError('camera_field must be an object with optional max_scale_m')
    if not isinstance(camera_field.get('bake_world',False),bool):
        raise ValueError('camera_field.bake_world must be a boolean')
    maximum=camera_field.get('max_scale_m')
    if maximum is not None and (isinstance(maximum,bool) or not isinstance(maximum,(int,float))
                               or not math.isfinite(maximum) or maximum<=0):
        raise ValueError('camera_field.max_scale_m must be positive and finite')
    for key,value in data['assets'].items():
        if value is None: continue
        expanded=Path(os.path.expandvars(os.path.expanduser(value)))
        data['assets'][key]=str(expanded if expanded.is_absolute() else path.parent/expanded)
        if check_assets and not Path(data['assets'][key]).is_file():
            raise FileNotFoundError(f'{key}: {data["assets"][key]}')
    transform=np.asarray(data.get('camera_to_world',np.eye(4)),dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError('camera_to_world must be a finite affine 4x4 transform')
    linear=transform[:3,:3]; scale=np.linalg.det(linear)**(1/3) if np.linalg.det(linear)>0 else 0
    if scale<=0 or not np.allclose(linear.T@linear,np.eye(3)*scale**2,atol=1e-5):
        raise ValueError('camera_to_world must be a positive uniform-scale rigid transform')
    data['camera_to_world']=transform.tolist()
    return data
