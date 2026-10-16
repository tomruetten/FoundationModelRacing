#!/usr/bin/env python3
"""Five-second Porsche replay with switchable CC0 hall props and shared LiDAR.

Requires the New1 checkpoints, built demo assets, and cached Livox ray templates
from render_livox_demo.py. Playback rate is not a rendering-speed measurement.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
from render_learned_demo import ROOT, font, SplatCameraRenderer, LidarRenderer, ObjectSplat
from object_splats import FloorPlane, trajectory_up, pose_from_camera_frame, contact_shadow
from render_livox_demo import panel, encode


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,1000*(time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/porsche_hall_demo_hd')
    ap.add_argument('--templates',type=Path,default=ROOT/'outputs/learned_demo_5s_livox_black/scans')
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--benchmark-only',action='store_true',help='Only measure the paired fixed-pose renders')
    ap.add_argument('--switch-at',type=float,default=1.5)
    ap.add_argument('--obstacle-quality',choices=['standard','high'],default='high')
    ap.add_argument('--resolution-scale',type=int,choices=[1,2],default=2,
                    help='2 renders camera at 2560x1600 and exports a 3840x2160 video')
    args=ap.parse_args()
    if args.preview:args.out=args.out/'preview'
    for folder in ('frames','scans'):(args.out/folder).mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/local/New1'
    frames=json.loads((data/'dataset/transforms.json').read_text())['frames']
    poses=np.array([f['transform_matrix'] for f in frames])
    times=np.array([int(Path(f['file_path']).stem.split('_')[-1])/30 for f in frames])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1))]
    rotations=Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))
    floor=FloorPlane.load(data/'dataset/floor_plane.json');up=trajectory_up(frames)
    start=times[560]
    def at(t):
        t=np.clip(t,times[0],times[-1]);p=np.eye(4);p[:3,:3]=rotations(t).as_matrix()
        p[:3,3]=[np.interp(t,times,poses[:,k,3]) for k in range(3)]
        return p
    def placement(t,gap,side=0,height=0):
        distance=np.interp(t,times,arc)+gap
        p=pose_from_camera_frame(at(np.interp(distance,arc,times)),lateral=side,up=up,floor=floor)
        p[:3,3]+=height*p[:3,2]
        return p
    assets={n:ObjectSplat.load(ROOT/'data/local/objects'/(n+('_hd' if args.obstacle_quality=='high' and n not in ('porsche_911','f1tenth') else '')+'.npz')) for n in
            ('porsche_911','f1tenth','cone_photoscan','wooden_pallet','wooden_crate','plastic_crate')}
    car=assets['porsche_911'];car=car.with_part(contact_shadow(.42,.19,darkness=.38),dict(name='contact_shadow',kind='shadow',lidar=False))
    assets['porsche_911']=car
    # These are static hall furnishings, not objects carried by the camera.
    props=[('wooden_pallet',2.5,-.95,0),('wooden_pallet',2.5,-.95,.154),
           ('wooden_crate',2.5,-.95,.308),('plastic_crate',3.1,.9,0),
           ('plastic_crate',3.1,.9,.254),('wooden_crate',5.2,-.75,0),
           ('wooden_pallet',9.2,.9,0),('plastic_crate',9.2,.9,.154),
           ('plastic_crate',9.2,.9,.408)]
    static=[(assets[n],placement(start,gap,side,height),None) for n,gap,side,height in props]
    cones=[(assets['cone_photoscan'],placement(start,gap,side),None) for gap,side in [(1.4,-.5),(2.5,.55),(3.8,-.5),(5,.55)]]
    def objects_at(t,changed):
        elapsed=t-start;gap=1.05+.12*elapsed;side=.10*np.sin(elapsed*1.1)
        travelled=np.interp(t,times,arc)-np.interp(start,times,arc)+gap-1.05
        angle=travelled/car.part_meta[1]['radius']
        result=[(car,placement(t,gap,side),car.articulated(wheel_angle=angle)),
                (assets['f1tenth'],placement(t,2.7,-.12),None)]+cones
        return result+(static if changed else [])
    rcl=Rotation.from_quat([-.47431095,.49897995,-.50794995,.51772095]).as_matrix().T
    tcl=np.eye(4);tcl[:3,:3]=rcl;tcl[:3,3]=-rcl@np.array([.02267,.06987,-.04441])
    tcl=np.diag([1.,-1.,-1.,1.])@tcl
    cam=SplatCameraRenderer(data/'checkpoints/camera_3dgs/config.yml')
    for attribute in ('width','height','fx','fy','cx','cy'):
        setattr(cam,attribute,getattr(cam,attribute)*args.resolution_scale)
    lidar=LidarRenderer(data/'checkpoints/lidar_3dgs/best.pt')
    templates=[np.load(p) for p in sorted(args.templates.glob('*.npz'))]
    stamps=np.array([float(p['pattern_stamp']) for p in templates]);stamps-=stamps[0]
    assert len(templates)>=50
    cam.render(at(start),objects=objects_at(start,False)) # warmup excluded from metrics
    # Paired fixed-pose benchmark: same pose and rays, differing only in hall props.
    benchmark={};comparison=Image.new('RGB',(2560,880),(9,16,26))
    bp=at(start+2);bl=bp@tcl;dirs=templates[20]['ray_dirs']
    for changed in (False,True):
        obs=objects_at(start+2,changed);camera_ms=[];lidar_ms=[]
        for repeat in range(4):
            matched_rgb,cm=timed(lambda:cam.render(bp,objects=obs))
            _,lm=timed(lambda:lidar.render_along_rays(bl[:3,:3],bl[:3,3],dirs,use_raydrop=False,alpha_thresh=.99,objects=obs))
            if repeat:camera_ms.append(cm);lidar_ms.append(lm)
        benchmark['with_props' if changed else 'without_props']={
            'camera_ms':camera_ms,'lidar_ms':lidar_ms,'camera_median_ms':float(np.median(camera_ms)),
            'lidar_median_ms':float(np.median(lidar_ms)),'object_instances':len(obs)}
        comparison.paste(Image.fromarray(matched_rgb).resize((1280,800),Image.Resampling.LANCZOS),(1280 if changed else 0,70))
        Image.fromarray(matched_rgb).save(args.out/('matched_after_native.png' if changed else 'matched_before_native.png'))
        ImageDraw.Draw(comparison).text((20+(1280 if changed else 0),18),
            'Same pose / props inserted' if changed else 'Same pose / original hall',font=font(28,True),fill='white')
    comparison.save(args.out/'matched_before_after.png')
    (args.out/'benchmark.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        obstacle_quality=args.obstacle_quality,resolution_scale=args.resolution_scale,
        measurements=benchmark,method='CUDA-synchronised wall time including compositing and transfers; same pose and rays; 4 repeats, first discarded; excludes encoding'),indent=2))
    if args.benchmark_only:return
    scan_cache={};records=[]
    for i in ([0,44,45,75,149] if args.preview else range(150)):
        elapsed=i/30;t=start+elapsed;changed=elapsed>=args.switch_at
        obs=objects_at(t,changed);pose=at(t)
        rgb,cm=timed(lambda:cam.render(pose,objects=obs))
        j=max(0,int(np.searchsorted(stamps,elapsed,side='right')-1))
        # A scene switch triggers a fresh scan immediately; retain its actual pose/time.
        key=(j,changed)
        if key not in scan_cache:
            scan_elapsed=max(float(stamps[j]),args.switch_at if changed else 0)
            scan_t=start+scan_elapsed;wl=at(scan_t)@tcl;scanobs=objects_at(scan_t,changed)
            dirs=templates[j]['ray_dirs']
            (xyz,intensity),lm=timed(lambda:lidar.render_along_rays(wl[:3,:3],wl[:3,3],dirs,use_raydrop=False,alpha_thresh=.99,objects=scanobs))
            valid=np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)<=8)
            xyz,intensity=xyz[valid],intensity[valid]
            assert len(xyz)>0
            np.savez_compressed(args.out/'scans'/f'{i:04d}.npz',xyz=xyz,intensity=intensity,ray_dirs=dirs,
                                lidar_to_world=wl,object_to_world=np.array([o[1] for o in scanobs]),
                                object_names=np.array([o[0].meta['name'] for o in scanobs]),
                                time_s=scan_elapsed,hall_changed=changed)
            scan_cache[key]=(panel(xyz),panel(xyz,True),len(xyz),lm)
        p3,pb,count,lm=scan_cache[key]
        canvas=Image.new('RGB',(1920,1080),(9,16,26));canvas.paste(Image.fromarray(rgb).resize((1152,720)),(0,150))
        canvas.paste(p3,(1152,110));canvas.paste(pb,(1152,560));d=ImageDraw.Draw(canvas)
        d.text((28,20),'RoboRacer / Miniature Porsche challenge',font=font(32,True),fill='white')
        d.text((30,67),'Learned camera + LiDAR maps | one set of object placements | no scene retraining',font=font(21),fill=(175,195,215))
        d.rounded_rectangle((20,165,550,205),radius=7,fill=(9,16,26))
        d.text((32,171),'HALL PROPS ADDED' if changed else 'ORIGINAL HALL / NEW OPPONENT',font=font(22,True),fill=(90,230,190))
        d.text((28,900),f'{elapsed:.2f} / 5.00 s | Porsche 911 Carrera 4S, 46 cm | {count:,} LiDAR returns',font=font(22),fill='white')
        d.text((28,945),'Pallet stack + wooden crates + storage crates' if changed else 'Hall changes at 1.50 s',font=font(23,True),fill=(90,230,190))
        d.text((28,994),'Trajectory replay | synthetic LiDAR with August Livox ray templates; known New1 artifacts remain',font=font(18),fill=(255,190,110))
        d.text((28,1030),'Porsche: Lionsharp Studios, CC BY 4.0, modified | Props: CC0 | 30 fps playback, offline rendering',font=font(18),fill=(165,185,205))
        if args.resolution_scale==2:
            canvas=canvas.resize((3840,2160),Image.Resampling.LANCZOS)
            # Preserve the newly rendered camera pixels; do not upscale a 720p panel.
            canvas.paste(Image.fromarray(rgb).resize((2304,1440),Image.Resampling.LANCZOS),(0,300))
            hi=ImageDraw.Draw(canvas)
            hi.rounded_rectangle((40,330,1100,410),radius=14,fill=(9,16,26))
            hi.text((64,342),'HALL PROPS ADDED' if changed else 'ORIGINAL HALL / NEW OPPONENT',font=font(44,True),fill=(90,230,190))
        canvas.save(args.out/'frames'/f'{i:04d}.png')
        records.append(dict(frame=i,time_s=elapsed,hall_changed=changed,camera_ms=cm,scan=j,returns=count,
                            camera_object_to_world=[o[1].tolist() for o in obs]))
        if i%15==0 or args.preview:print('frame',i,'props',changed,'camera_ms',round(cm,1),'returns',count,flush=True)
    provenance=dict(gpu=torch.cuda.get_device_name(),benchmark=benchmark,frames=records,props=props,
                    obstacle_quality=args.obstacle_quality,resolution_scale=args.resolution_scale,
                    template_source=str(args.templates),switch_at_s=args.switch_at,
                    timing='CUDA synchronised wall time, model render + compositing + transfers; excludes file I/O and encoding. Four repeats per condition, first discarded. Same pose and rays.',
                    motion='Scripted trajectory replay with wheel rotation, not closed-loop racing or obstacle avoidance.',
                    lidar='New1 LiDAR checkpoint + shared object Gaussians. Raw-return directions from another session. No accuracy claim.',
                    appearance='Mesh-derived Gaussians with baked lighting; no physically correct moving reflections.',
                    asset_provenance={k:v.meta for k,v in assets.items()})
    (args.out/('preview_metrics.json' if args.preview else 'metrics.json')).write_text(json.dumps(provenance,indent=2))
    if not args.preview:encode(args.out/'frames',args.out/'porsche_hall_change_5s.mp4')


if __name__=='__main__':main()
