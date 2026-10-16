#!/usr/bin/env python3
"""Render a grounded, motion-blurred staged ego overtake around the New1 track."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from scipy.io import wavfile
from scipy.signal import butter,sosfilt
from plan_staged_overtake import plan,ROOT
from render_learned_demo import SplatCameraRenderer,LidarRenderer,ObjectSplat,font
from object_splats import contact_shadow,attitude_from_accel
from render_livox_demo import panel


class Race:
    def __init__(self,p):
        self.p=p;self.ts=np.array(p['time']);self.basis=np.array(p['basis']);self.up=self.basis[:,2]
        self.columns={name:{k:np.array(v) for k,v in actor.items()} for name,actor in p['actors'].items()}
    def state(self,name,t):
        r=self.columns[name];s={k:float(np.interp(t,self.ts,v)) for k,v in r.items() if v.ndim==1}
        xyz=np.array([np.interp(t,self.ts,r['position'][:,k]) for k in range(3)])
        pose=np.eye(4);pose[:3,:3]=self.basis@Rotation.from_euler('z',s['heading']).as_matrix();pose[:3,3]=xyz
        return pose,s
    def camera(self,t,rear=False):
        pose,s=self.state('ego',t);f,left,up=pose[:3,:3].T
        camera=np.eye(4)
        side_view=rear and 4.2<=t<=5.9
        forward=-left if side_view else (-f if rear else f)
        camera[:3,:3]=np.stack([-np.cross(up,forward),up,-forward],axis=1)
        camera[:3,3]=pose[:3,3]+(.17 if rear else .155)*up
        camera[:3,3]+=(-.17*left if side_view else (-.29 if rear else .24)*f)
        # Small mount response, not arbitrary camera shake.
        pitch=np.clip(-.002*s['a_x'],-.006,.006)
        roll=np.clip(-.002*s['a_y'],-.008,.008)
        camera[:3,:3]=camera[:3,:3]@Rotation.from_euler('xz',[(-.32 if side_view else -.025)+pitch,roll]).as_matrix()
        return camera


def audio_track(race,path,duration):
    fs=44100;t=np.arange(int(duration*fs))/fs;rng=np.random.default_rng(31)
    v=np.interp(t,race.ts,race.columns['ego']['speed']);vo=np.interp(t,race.ts,race.columns['opponent']['speed'])
    # Quiet synthetic electric-RC drivetrain and tyre/air noise, explicitly not recorded audio.
    phase=2*np.pi*np.cumsum(190+180*v)/fs
    motor=.027*np.sin(phase)+.009*np.sin(2*phase)+.004*np.sin(4*phase)
    noise=sosfilt(butter(2,1800,fs=fs,output='sos'),rng.normal(size=len(t)))
    noise*=.018+.008*v
    a=np.array(race.columns['ego']['position']);b=np.array(race.columns['opponent']['position'])
    distance=np.interp(t,race.ts,np.linalg.norm(b-a,axis=1));opp_phase=2*np.pi*np.cumsum(210+180*vo)/fs
    opponent=.025*np.sin(opp_phase)/(1+distance)
    stereo=np.c_[motor+noise+opponent*.8,motor+noise+opponent]
    fade=np.minimum(np.clip(t/.3,0,1),np.clip((duration-t)/.4,0,1))
    wavfile.write(path,fs,(np.clip(stereo*fade[:,None],-1,1)*32767).astype(np.int16))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/staged_overtake')
    ap.add_argument('--preview',action='store_true');ap.add_argument('--blur-samples',type=int,choices=[1,3,5],default=3)
    ap.add_argument('--exposure-ms',type=float,default=8)
    ap.add_argument('--ego-asset',default='f1tenth.npz',help='object splat in data/local/objects for the ego car')
    ap.add_argument('--opponent-asset',default='porsche_911.npz',help='object splat for the opponent (e.g. f1tenth.npz for a second F1TENTH)')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    objdir=ROOT/'data/local/objects'
    def half_size(file):
        lo,hi=ObjectSplat.load(objdir/file).bbox();return (float(hi[0]-lo[0])/2,float(hi[1]-lo[1])/2)
    def wheelbase(file):
        m=ObjectSplat.load(objdir/file);xs=[q['centre'][0] for q in m.part_meta if q.get('kind')=='wheel']
        return float(max(xs)-min(xs)) if len(xs)>=2 else .3
    p=plan(ego_half=half_size(args.ego_asset),opponent_half=half_size(args.opponent_asset),
           ego_wheelbase=wheelbase(args.ego_asset),opponent_wheelbase=wheelbase(args.opponent_asset))
    print('plan validation:',json.dumps(p['validation'],indent=None)[:400],flush=True)
    race=Race(p);duration=p['time'][-1];nframes=round(duration*30)
    target=args.out/'preview' if args.preview else args.out;target.mkdir(exist_ok=True)
    (target/'scans').mkdir(exist_ok=True)
    data=ROOT/'data/local/New1'
    models={name:ObjectSplat.load(objdir/file) for name,file in [('ego',args.ego_asset),('opponent',args.opponent_asset),('pallet','wooden_pallet_hd.npz'),('crate','wooden_crate_hd.npz')]}
    for name in ('ego','opponent'):
        car=models[name]
        if not any(q['kind']=='shadow' for q in car.part_meta):
            lo,hi=car.bbox()   # shadow footprint from the asset itself
            models[name]=car.with_part(contact_shadow(float(hi[0]-lo[0])*.9,float(hi[1]-lo[1])*.8,darkness=.32),dict(name='contact_shadow',kind='shadow',lidar=False))
    print('ego:',models['ego'],'| opponent:',models['opponent'],flush=True)
    opp_is_porsche='porsche' in args.opponent_asset.lower()
    opp_label='PORSCHE' if opp_is_porsche else 'F1TENTH #2'
    title='RoboRacer / Staged Porsche overtake' if opp_is_porsche else 'RoboRacer / Staged F1TENTH overtake'
    credits=('Porsche: Lionsharp Studios, CC BY 4.0, modified. ' if opp_is_porsche else 'Both cars: AutoDRIVE F1TENTH CAD as Gaussian splats. ')+'Hall props: CC0. Offline 30 fps playback.'
    # Keep clutter off the passing straight; furnishings stay fixed throughout the lap.
    prop_pose,_=race.state('ego',12.)
    prop_pose[:3,3]+=1.3*prop_pose[:3,1]
    crate_pose=prop_pose.copy();crate_pose[:3,3]+=.154*race.up
    props=[(models['pallet'],prop_pose,None),(models['crate'],crate_pose,None)]
    prop_clearance=[]
    for model,pose,_ in props:
        lo,hi=model.bbox();prop_radius=np.linalg.norm((hi-lo)[:2]/2)
        for name in ('ego','opponent'):
            lo,hi=models[name].bbox();car_radius=np.linalg.norm((hi-lo)[:2]/2)
            xy=(race.columns[name]['position']-pose[:3,3])@race.basis
            margin=float(np.min(np.linalg.norm(xy[:,:2],axis=1))-prop_radius-car_radius)
            assert margin>.05,('Static prop too close to planned vehicle',name,margin)
            prop_clearance.append(margin)
    p['validation']['min_conservative_prop_clearance_m']=min(prop_clearance)
    (target/'trajectory.json').write_text(json.dumps(p))
    def moving(name,t):
        pose,s=race.state(name,t);model=models[name]
        wheel=next((q for q in model.part_meta if q['kind']=='wheel'),{'radius':.057})
        pitch,roll=attitude_from_accel(s['a_x'],s['a_y'],pitch_gain=np.radians(.2),roll_gain=np.radians(.3),max_deg=1.)
        art=model.articulated(wheel_angle=s['odometer']/wheel['radius'],steer_angle=s['steer'],pitch=pitch,roll=roll)
        return model,pose,art
    def objects(t,include_ego=True):
        return ([moving('ego',t)] if include_ego else [])+[moving('opponent',t)]+props
    cam=SplatCameraRenderer(data/'checkpoints/camera_3dgs_flat/config.yml');cam.object_soften=1.05
    standard={a:getattr(cam,a) for a in ('width','height','fx','fy','cx','cy')}
    def camera_render(t,rear=False):
        if rear:
            vals=dict(width=768,height=256,fx=350.,fy=350.,cx=384.,cy=128.)
        else:vals={k:v*2 for k,v in standard.items()}
        for k,v in vals.items():setattr(cam,k,v)
        return cam.render(race.camera(t,rear),objects=objects(t))
    lidar=LidarRenderer(data/'checkpoints/lidar_3dgs/best_flat.pt')
    templates=[]
    for f in sorted((ROOT/'outputs/learned_demo_5s_livox_black/scans').glob('*.npz')):
        with np.load(f) as q:templates.append(q['ray_dirs'].copy())
    assert len(templates)>=50
    rcl=Rotation.from_quat([-.47431095,.49897995,-.50794995,.51772095]).as_matrix().T
    tcl=np.eye(4);tcl[:3,:3]=rcl;tcl[:3,3]=-rcl@np.array([.02267,.06987,-.04441]);tcl=np.diag([1.,-1.,-1.,1.])@tcl
    # Preview milestones include the side-by-side phase and the lead visible in rear view.
    milestones=[0,90,135,153,180,240,360,nframes-1]
    frame_ids=milestones if args.preview else range(nframes)
    writer=None
    if not args.preview:
        audio_track(race,target/'synthetic_rc_audio.wav',duration)
        writer=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','30','-i','-',
            '-frames:v',str(nframes),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target/'staged_overtake_4k.mp4')],stdin=subprocess.PIPE)
    linear=np.arange(256)/255.;linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4).astype(np.float32)
    records=[];scan_cache={};start=time.perf_counter()
    # Track minimap uses only the current circuit rather than all recorded laps.
    track=np.array(p['track_xy']);ss=np.array(p['track_s']);s0=p['actors']['ego']['progress'][0]
    track=track[(ss>=s0)&(ss<=s0+28.)]
    lo=track.min(0);extent=np.ptp(track,axis=0);map_scale=min(500/extent[0],270/extent[1])
    def map_xy(world):
        xy=(np.asarray(world)-np.array(p['floor_point']))@race.basis
        return (60+(xy[0]-lo[0])*map_scale,2090-(xy[1]-lo[1])*map_scale)
    try:
        for i in frame_ids:
            t=i/30;acc=np.zeros((1600,2560,3),np.float32)
            offsets=[0.] if args.blur_samples==1 else np.linspace(-args.exposure_ms/2000,args.exposure_ms/2000,args.blur_samples)
            for offset in offsets:
                rgb=camera_render(np.clip(t+offset,0,duration));acc+=linear[rgb]
            acc/=args.blur_samples
            rgb=(255*np.clip(np.where(acc<=.0031308,12.92*acc,1.055*np.maximum(acc,0)**(1/2.4)-.055),0,1)).astype(np.uint8)
            rear=camera_render(t,True)
            j=i//3
            if j not in scan_cache:
                st=j/10;wl=race.camera(st)@tcl;obs=objects(st,False);dirs=templates[j%len(templates)]
                xyz,intensity=lidar.render_along_rays(wl[:3,:3],wl[:3,3],dirs,use_raydrop=False,alpha_thresh=.99,objects=obs)
                ok=np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)<=8);xyz,intensity=xyz[ok],intensity[ok]
                np.savez_compressed(target/'scans'/f'{j:04d}.npz',xyz=xyz,intensity=intensity,ray_dirs=dirs,lidar_to_world=wl,time_s=st,
                                    object_to_world=np.array([e[1] for e in obs]),ego_to_world=race.state('ego',st)[0])
                scan_cache[j]=(panel(xyz),panel(xyz,True),len(xyz))
            p3,pb,count=scan_cache[j]
            canvas=Image.new('RGB',(3840,2160),(9,16,26));canvas.paste(Image.fromarray(rgb),(0,150))
            canvas.paste(p3.resize((1280,716),Image.Resampling.LANCZOS),(2560,150))
            canvas.paste(pb.resize((1280,716),Image.Resampling.LANCZOS),(2560,940))
            # A real rear-facing virtual camera, not a prerecorded picture or composited car.
            canvas.paste(Image.fromarray(rear),(900,205));d=ImageDraw.Draw(canvas)
            d.rectangle((896,170,1672,465),outline=(110,130,145),width=4)
            d.rectangle((900,174,1668,204),fill=(9,16,26));d.text((914,175),'RIGHT SIDE VIEW' if 4.2<=t<=5.9 else 'REAR VIEW',font=font(24,True),fill='white')
            d.text((35,22),title,font=font(54,True),fill='white')
            d.text((38,91),f'Floor-corrected scene  /  steering + wheel spin  /  {args.exposure_ms:g} ms exposure',font=font(30),fill=(165,185,205))
            e,es=race.state('ego',t);o,os=race.state('opponent',t);gap=os['progress']-es['progress']
            phase='CHASE' if t<2.2 else ('MOVE OUT' if t<4.2 else ('PASS' if t<6.2 else ('PULL AHEAD' if t<9.4 else 'CONTINUE LAP')))
            d.text((680,1820),phase,font=font(46,True),fill=(90,230,190))
            d.text((680,1900),f"EGO  {es['speed']:.2f} m/s      {opp_label}  {os['speed']:.2f} m/s",font=font(36),fill='white')
            d.text((680,1970),f"{'Opponent ahead' if gap>0 else 'Ego ahead'}: {abs(gap):.2f} m     |     {t:.2f} / {duration:.0f} s",font=font(32),fill=(180,200,215))
            track_pixels=[(60+(q[0]-lo[0])*map_scale,2090-(q[1]-lo[1])*map_scale) for q in track]
            d.line(track_pixels,fill=(85,105,125),width=7)
            for pose,color,label in [(e,(90,230,190),'E'),(o,(245,245,245),'P')]:
                x,y=map_xy(pose[:3,3]);d.ellipse((x-12,y-12,x+12,y+12),fill=color);d.text((x+15,y-17),label,font=font(25,True),fill=color)
            d.text((2580,1775),f'{count:,} LiDAR returns / 10 Hz',font=font(31),fill='white')
            d.text((680,2060),'Staged motion; synthetic RC audio. LiDAR uses a separate learned map; geometry artifacts remain.',font=font(26),fill=(210,180,130))
            d.text((680,2105),credits,font=font(24),fill=(150,170,190))
            if i in milestones:canvas.save(target/f'{i:04d}.png')
            if writer:writer.stdin.write(np.asarray(canvas).tobytes())
            records.append(dict(frame=i,time_s=t,gap_m=gap,phase=phase,ego_speed=es['speed'],opponent_speed=os['speed']))
            if i%30==0 or args.preview:print('frame',i,'phase',phase,'gap',round(gap,2),'elapsed',round(time.perf_counter()-start,1),flush=True)
    finally:
        if writer:
            writer.stdin.close();code=writer.wait()
            if code:raise RuntimeError('ffmpeg failed: '+str(code))
    if writer:
        # Mux after video completion: a raw-video frame limit can otherwise truncate
        # buffered audio before the final frames, even with a full-length WAV.
        movie=target/'staged_overtake_4k.mp4';muxed=target/'staged_overtake_4k_audio.mp4'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(movie),'-i',str(target/'synthetic_rc_audio.wav'),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k',
            '-t',str(duration),'-movflags','+faststart',str(muxed)],check=True)
        muxed.replace(movie)
    (target/'render_metrics.json').write_text(json.dumps(dict(frames=records,duration_s=duration,validation=p['validation'],
        camera_checkpoint=str(data/'checkpoints/camera_3dgs_flat/config.yml'),lidar_checkpoint=str(data/'checkpoints/lidar_3dgs/best_flat.pt'),
        motion='Staged smooth kinematic paths, not reactive racing AI',rear_view='Rendered from ego rear camera; right-side camera during overlap',
        audio='Synthesised electric motor and rolling noise, not a recording',lidar='August return templates repeated every 5 seconds; ego self returns excluded',
        wall_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
