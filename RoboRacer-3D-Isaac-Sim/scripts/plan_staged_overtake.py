#!/usr/bin/env python3
"""Plan a continuous, explicitly staged pass along the reconstructed New1 track."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulation/isaac_sim'))
from object_splats import FloorPlane


def smoothstep(t,a,b):
    x=np.clip((t-a)/(b-a),0,1)
    return x*x*x*(10+x*(-15+6*x))


def plan(duration=19.,dt=.01,start_frame=560,ego_half=(.265,.146),opponent_half=(.23,.104),ego_wheelbase=.33,opponent_wheelbase=.27):
    """ego_half / opponent_half: half length and half width of the vehicles' footprints [m]
    (from the object splats' bounding boxes), used for the lane split and the footprint clearance test."""
    data=ROOT/'data/local/New1/dataset'
    f=json.loads((data/'transforms.json').read_text())['frames'];poses=np.array([x['transform_matrix'] for x in f])
    floor=FloorPlane.load(data/'floor_plane.json');up=floor.normal
    ex=-poses[start_frame,:3,2];ex-=ex@up*up;ex/=np.linalg.norm(ex)
    ey=np.cross(up,ex);basis=np.stack([ex,ey,up],axis=1)
    path=(floor.project(poses[:,:3,3])-floor.point)@basis
    raw_s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))]
    # Smooth on a uniform distance grid, so slow/paused recording does not distort the line.
    ss=np.arange(0,raw_s[-1],.04)
    xy=np.column_stack([np.interp(ss,raw_s,path[:,k]) for k in (0,1)])
    xy=gaussian_filter1d(xy,6.0,axis=0)
    curve=CubicSpline(ss,xy)
    time=np.arange(0,duration+dt/2,dt);s=np.zeros(len(time));s[0]=raw_s[start_frame]
    speed=1.45
    for i in range(1,len(time)):
        d1=curve(s[i-1],1);d2=curve(s[i-1],2)
        curvature=abs(np.cross(d1,d2))/max(np.linalg.norm(d1)**3,1e-6)
        pass_boost=.35*smoothstep(time[i],2.7,3.8)*(1-smoothstep(time[i],7.3,8.3))
        target=min(1.5+pass_boost,np.sqrt(2.4/max(curvature,.01)))
        speed+=np.clip(target-speed,-1.5*dt,1.2*dt)
        s[i]=s[i-1]+speed*dt/max(np.linalg.norm(d1),.5)
    assert s[-1]+2<ss[-1]
    gap=1.25-2.5*smoothstep(time,3.0,7.2)
    lane=smoothstep(time,2.2,3.8)*(1-smoothstep(time,7.8,9.4))
    result={}
    # lane split: ego moves 0.24 m left, the opponent 0.18 m right plus whatever it is wider than the reference car
    opp_lane=.18+max(0.,opponent_half[1]-.104)
    for name,progress,lateral,wheelbase in [('ego',s,.24*lane,ego_wheelbase),('opponent',s+gap,-opp_lane*lane,opponent_wheelbase)]:
        center=curve(progress);tangent=curve(progress,1);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        left=np.c_[-tangent[:,1],tangent[:,0]];pos=center+lateral[:,None]*left
        vel=np.gradient(pos,dt,axis=0);v=np.linalg.norm(vel,axis=1)
        heading=np.unwrap(np.arctan2(vel[:,1],vel[:,0]));yr=np.gradient(heading,dt)
        steer=np.arctan(wheelbase*yr/np.maximum(v,.1))
        odo=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
        world=floor.point+np.c_[pos,np.full(len(pos),floor.clearance)]@basis.T
        result[name]=dict(position=world.tolist(),heading=heading.tolist(),speed=v.tolist(),
                          yaw_rate=yr.tolist(),steer=steer.tolist(),a_x=np.gradient(v,dt).tolist(),
                          a_y=(v*yr).tolist(),odometer=odo.tolist(),progress=progress.tolist(),lateral=lateral.tolist())
    # Separating-axis test on oriented footprints in the levelled track frame.
    clearance=[]
    for i in range(len(time)):
        corners=[];axes=[]
        for name,half in [('ego',np.asarray(ego_half)),('opponent',np.asarray(opponent_half))]:
            heading=result[name]['heading'][i];R=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
            pos=(np.array(result[name]['position'][i])-floor.point)@basis
            corners.append(np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*half@R.T+pos[:2]);axes.extend(R.T)
        sep=[]
        for axis in axes:
            aa,bb=corners[0]@axis,corners[1]@axis
            sep.append(max(bb.min()-aa.max(),aa.min()-bb.max()))
        clearance.append(max(sep))
    report=dict(min_separating_axis_gap_m=float(min(clearance)),start_gap_m=float(gap[0]),end_gap_m=float(gap[-1]),
                pass_time_s=float(np.interp(0,-gap,time)),travel_m=result['ego']['odometer'][-1])
    position=np.array(result['ego']['position']);return_distance=np.linalg.norm(position-position[0],axis=1)
    candidates=np.where(time>13)[0];lap_index=candidates[np.argmin(return_distance[candidates])]
    report.update(lap_return_time_s=float(time[lap_index]),lap_return_distance_m=float(return_distance[lap_index]))
    assert return_distance[lap_index]<.5,report
    for name in result:
        r=result[name];report[name]=dict(min_speed=min(r['speed']),max_speed=max(r['speed']),
            max_acceleration=max(abs(np.array(r['a_x']))),max_lateral_acceleration=max(abs(np.array(r['a_y']))),
            max_steer_deg=float(np.degrees(max(abs(np.array(r['steer']))))))
    assert report['min_separating_axis_gap_m']>.07,report
    for name in result:
        assert report[name]['min_speed']>.1 and report[name]['max_speed']<2.5,report
        assert report[name]['max_acceleration']<4 and report[name]['max_lateral_acceleration']<5,report
        assert report[name]['max_steer_deg']<40,report
    return dict(time=time.tolist(),basis=basis.tolist(),floor_point=floor.point.tolist(),floor_normal=up.tolist(),
                start_frame=start_frame,actors=result,validation=report,track_xy=xy.tolist(),track_s=ss.tolist(),
                kind='Staged kinematic trajectory with validated bounds; not a reactive racing policy')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'outputs/staged_overtake/trajectory.json')
    args=ap.parse_args();p=plan();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(p));print(json.dumps(p['validation'],indent=2))
