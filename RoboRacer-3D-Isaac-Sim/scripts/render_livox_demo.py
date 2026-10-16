#!/usr/bin/env python3
"""Re-render a saved learned demo along recorded Livox return directions.

The August recording supplies angular sampling only, not New1 geometry or truth.
Requires an existing render_learned_demo.py output and a ROS2 PointCloud2 bag.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.eval_lidar import iter_scans
from render_learned_demo import font, LidarRenderer, ObjectSplat
import subprocess


def panel(points, bev=False, size=(768, 430)):
    w, h = size
    im = Image.new('RGB', size, (12, 22, 34))
    d = ImageDraw.Draw(im)
    # Orthographic views: no perspective foreshortening or hidden height filter.
    scale = min((w-70)/16, (h-85)/16)
    if bev:
        basis = np.array([[0, -1, 0], [-1, 0, 0.]])
    else:
        basis = np.array([[.7071, -.7071, 0], [-.35355, -.35355, -.8660254]])
    def project(p):
        return np.asarray(p) @ basis.T * scale + [w/2, (h+35)/2]
    for k in range(-8, 9):
        for line in ([[k,-8,0],[k,8,0]], [[-8,k,0],[8,k,0]]):
            d.line([tuple(x) for x in project(line)], fill=(30,44,57))
    xy = project(points).astype(int)
    good = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=52)&(xy[:,1]<h-24)
    # Height in metres, identical fixed colour scale for raw and synthetic.
    z = np.clip((points[:,2]+1)/4, 0, 1)
    colors = np.column_stack((60+195*z, 210-100*z, 250-195*z)).astype('uint8')
    a = np.array(im)
    a[xy[good,1], xy[good,0]] = colors[good]
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.text((20,12), 'BEV / sensor frame' if bev else '3D / sensor frame', font=font(22,True), fill='white')
    for p, label, color in [([2,0,0],'+X',(255,120,100)),([0,2,0],'+Y',(90,230,130))]:
        origin, end = project([[0,0,0],p])
        d.line([tuple(origin),tuple(end)], fill=color,width=2)
        d.text(tuple(end),label,font=font(13),fill=color)
    d.text((20,h-23),'Grid 1 m | range <= 8 m | height: cyan -1 m to gold +3 m',font=font(14),fill=(165,185,205))
    return im


def encode(folder, target):
    subprocess.run(['ffmpeg','-y','-v','error','-framerate','30','-i',str(folder/'%04d.png'),
                    '-frames:v','150','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(target)],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=ROOT/'outputs/learned_demo_5s_detailed')
    ap.add_argument('--bag',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/learned_demo_5s_livox')
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args()
    for folder in ('frames','raw_frames','scans','raw_scans'):
        (args.out/folder).mkdir(parents=True,exist_ok=True)
    scans=[]
    for stamp,xyz,intensity in iter_scans(args.bag):
        lengths=np.linalg.norm(xyz,axis=1)
        good=np.isfinite(xyz).all(axis=1)&(lengths>.1)
        xyz=xyz[good]; intensity=intensity[good] if intensity is not None else np.zeros(len(xyz))
        if not scans and not len(xyz):
            continue
        scans.append((stamp,xyz,intensity))
        if stamp-scans[0][0]>=5:
            break
    assert len(scans)>1 and scans[-1][0]-scans[0][0]>=5, 'Need five seconds of raw scans'
    stamps=np.array([x[0]-scans[0][0] for x in scans])
    lidar=LidarRenderer(ROOT/'data/local/New1/checkpoints/lidar_3dgs/best.pt')
    car=ObjectSplat.load(ROOT/'data/local/objects/f1tenth.npz')
    cone=ObjectSplat.load(ROOT/'data/local/objects/cone_photoscan.npz')
    cached={}; records=[]
    for i in ([0,75,149] if args.preview else range(150)):
        j=max(0,int(np.searchsorted(stamps,i/30,side='right')-1))
        if j not in cached:
            source_i=min(149,round(stamps[j]*30))
            saved=np.load(args.source/'scans'/f'{source_i:04d}.npz')
            wl=saved['lidar_to_world']
            objects=list(zip([car,car,cone,cone,cone,cone],saved['object_to_world']))
            stamp,raw,inten=scans[j]
            dirs=raw/np.linalg.norm(raw,axis=1,keepdims=True)
            pts,values=lidar.render_along_rays(wl[:3,:3],wl[:3,3],dirs,
                                               use_raydrop=False,alpha_thresh=.99,objects=objects)
            good=np.isfinite(pts).all(axis=1)&(np.linalg.norm(pts,axis=1)<=8)
            pts,values=pts[good],values[good]
            raw_good=np.linalg.norm(raw,axis=1)<=8
            np.savez_compressed(args.out/'scans'/f'{j:04d}.npz',xyz=pts,intensity=values,
                                ray_dirs=dirs,lidar_to_world=wl,source_frame=source_i,
                                pattern_stamp=stamp,object_to_world=saved['object_to_world'])
            np.savez_compressed(args.out/'raw_scans'/f'{j:04d}.npz',xyz=raw,intensity=inten,stamp=stamp)
            cached[j]=(panel(pts),panel(pts,True),panel(raw[raw_good]),panel(raw[raw_good],True),len(pts))
            print(f'scan {j}: rays={len(dirs)} synthetic={len(pts)}',flush=True)
        p3,pb,r3,rb,n=cached[j]
        canvas=Image.new('RGB',(1920,1080),(9,16,26))
        camera=Image.open(args.source/'frames'/f'{i:04d}.png').crop((0,100,1152,820))
        canvas.paste(camera,(0,150)); canvas.paste(p3,(1152,110));canvas.paste(pb,(1152,560))
        d=ImageDraw.Draw(canvas)
        d.text((28,20),'RoboRacer / Camera + Livox-pattern LiDAR',font=font(32,True),fill='white')
        d.text((30,67),'SYNTHETIC New1 geometry | recorded Livox return directions from another session',font=font(21),fill=(175,195,215))
        d.text((28,905),f'{i/30:.2f} / 5.00 s   |   {n:,} LiDAR returns   |   scan {j:02d} at {stamps[j]:.3f} s',font=font(23),fill='white')
        d.text((28,954),'2 Gaussian opponents + 4 photo-scanned cones | each scan shown in both views',font=font(21),fill=(90,230,190))
        d.text((28,1000),'Diagnostic render: New1 streak artifacts remain; this is not a validated raw-sensor reconstruction.',font=font(20),fill=(255,190,110))
        d.text((28,1035),'No scan accumulation / no motion distortion simulation / recorded return pattern is not a complete firing pattern',font=font(17),fill=(165,185,205))
        canvas.save(args.out/'frames'/f'{i:04d}.png')
        rawcanvas=Image.new('RGB',(1536,600),(9,16,26));rd=ImageDraw.Draw(rawcanvas)
        rawcanvas.paste(r3,(0,95));rawcanvas.paste(rb,(768,95))
        rd.text((25,15),'RAW LIVOX REFERENCE / 2026-08-27, session 142014',font=font(27,True),fill='white')
        rd.text((25,55),'Different track and time from New1. Recorded XYZ; no Gaussian rendering or inserted objects.',font=font(20),fill=(175,195,215))
        rd.text((25,550),f'{i/30:.2f} / 5.00 s | scan {j:02d} | same 8 m range and height colour scale as synthetic views',font=font(20),fill='white')
        rawcanvas.save(args.out/'raw_frames'/f'{i:04d}.png')
        records.append(dict(frame=i,scan=j,scan_time=float(stamps[j]),returns=n))
    (args.out/'provenance.json').write_text(json.dumps(dict(bag=str(args.bag),source=str(args.source),
        sampling='Consecutive recorded valid return directions, not a full firing pattern. Different session.',
        geometry='New1 LiDAR best.pt with shared object splats; known artifacts remain',
        scan_intervals_s=np.diff(stamps).tolist(),frames=records),indent=2))
    if not args.preview:
        encode(args.out/'frames',args.out/'roboracer_camera_lidar_3d_bev_5s.mp4')
        encode(args.out/'raw_frames',args.out/'raw_livox_reference_5s.mp4')


if __name__=='__main__':
    main()
