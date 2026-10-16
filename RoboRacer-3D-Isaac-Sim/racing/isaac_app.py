"""One simulation owner: planar dynamics -> native mesh/splat image -> hybrid LiDAR."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from .config import load_scenario, pose_matrix, sensor_pose, Schedule


class RuntimeCoordinator:
    """Drive arbitration and fault latch, independent of Isaac/ROS imports."""
    def __init__(self, engine, controls):
        self.engine, self.controls = engine, controls
        self.fault = None
        self.waiting_for_ros = controls.mode == 'ros'
        engine.set_control_mode(controls.mode)

    def fail(self, error):
        self.fault = str(error)
        self.controls.paused = True
        return self.engine.pause('runtime failure: ' + self.fault)

    def reset(self, seed, reset_sensors):
        try:
            snapshot = self.engine.reset(seed)
            self.controls.reset(snapshot.episode_id)
            self.waiting_for_ros = self.controls.mode == 'ros'
            reset_sensors(snapshot)
        except Exception as exc:
            self.fail(exc)
            return False
        self.fault = None
        return True

    def step(self, poll_commands=None, clock=time.monotonic):
        from .core import Command
        if self.fault is not None:
            return self.engine.snapshot()
        if self.controls.mode != self.engine.control_mode:
            self.engine.set_control_mode(self.controls.mode)
            self.waiting_for_ros = self.controls.mode == 'ros'
        try:
            commands = poll_commands() if poll_commands else []
            # Receipt timestamps may be produced during poll: read clock after it.
            now = clock()
            if self.controls.paused:
                self.engine.set_control_mode(self.controls.mode)
                if self.controls.mode == 'ros':
                    self.waiting_for_ros = True
                return self.engine.pause('paused by driver')
            for command in commands:
                if (self.engine.control_mode == 'ros' and
                        command['episode_id'] == self.engine.episode_id and
                        0 <= now-command['received_at'] <= self.engine.command_timeout):
                    self.engine.set_command(Command(command['speed'], command['steering']),
                                            'ros', command['received_at'])
                    self.waiting_for_ros = False
            if self.engine.control_mode == 'ros' and self.waiting_for_ros:
                return self.engine.pause('waiting for ROS command')
            if self.engine.control_mode == 'manual':
                command = self.controls.command(self.engine.dt, now)
                self.engine.set_command(Command(command.speed, command.steering), 'manual', now)
            self.engine.resume()
            snapshot = self.engine.step(now)
            if snapshot.status == 'paused' and snapshot.reason != 'command timeout':
                return self.fail(snapshot.reason)
            return snapshot
        except Exception as exc:
            return self.fail(exc)


def _camera(stage,path,intrinsics):
    from pxr import UsdGeom
    cam=UsdGeom.Camera.Define(stage,path)
    w,h=intrinsics['width'],intrinsics['height']; fx,fy=intrinsics['fx'],intrinsics['fy']
    horizontal=20.; vertical=horizontal*h*fx/(w*fy)
    cam.CreateHorizontalApertureAttr(horizontal); cam.CreateVerticalApertureAttr(vertical)
    cam.CreateFocalLengthAttr(horizontal*fx/w)
    cam.CreateHorizontalApertureOffsetAttr((w/2-intrinsics['cx'])*horizontal/w)
    cam.CreateVerticalApertureOffsetAttr((intrinsics['cy']-h/2)*vertical/h)
    cam.CreateClippingRangeAttr((.01,1000.))
    return cam,cam.AddTransformOp()


def run(args):
    scenario=load_scenario(args.scenario,check_assets=False)
    # USD authoring uses a standalone USD build; isolate it from Kit's USD ABI.
    import subprocess
    import sys
    command=[sys.executable,'-m','racing.assets',str(Path(args.scenario).resolve())]
    if args.rebuild_assets: command.append('--force')
    subprocess.run(command,check=True)
    scenario=load_scenario(args.scenario)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless,'width':1280,'height':800,
        'multi_gpu':False,'renderer':'RealTimePathTracing','anti_aliasing':3,
        'extra_args':['--/renderer/multiGpu/enabled=false']})
    worker=bridge=None; keyboard_subscription=None; output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    timing={'dynamics_ms':[],'camera_ms':[],'lidar_ms':[],'tick_ms':[]}
    exit_code=0; visual_report=None
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom,UsdLux,Gf
        from .assets import add_vehicle,mesh_triangles
        from .mesh import VehicleRaycaster
        from .core import RaceEngine,VehicleState,Command
        from .dynamics import GRUDynamics,KinematicDynamics
        from .controls import KeyboardControl
        from .lidar import generate_rays,compose_ray_samples
        from .worker import WorkerClient
        import omni.kit.viewport.utility as viewport
        # Author complete native scene before the first Hydra sync.
        context=omni.usd.get_context(); context.new_stage(); stage=context.get_stage()
        stage.DefinePrim('/World','Xform').GetReferences().AddReference(scenario['assets']['camera_usd'])
        UsdGeom.SetStageUpAxis(stage,'Z'); UsdGeom.SetStageMetersPerUnit(stage,1.)
        settings=carb.settings.get_settings()
        settings.set('/renderer/multiGpu/enabled',False)
        settings.set('/rtx/rendermode','RealTimePathTracing')
        settings.set('/rtx/rtpt/gaussian/skipTonemapping/enabled',True)
        settings.set('/rtx/post/tonemap/op',0)
        settings.set('/rtx/ecoMode/enabled',False)
        accumulation=bool(scenario.get('rendering',{}).get('gaussian_accumulation',True))
        settings.set('/rtx/rtpt/gaussian/accumulatedDepth/enabled',accumulation)
        settings.set('/rtx/rtpt/gaussian/accumulatedAlbedo/enabled',accumulation)
        settings.set('/rtx/rtpt/gaussian/maxGaussiansToAccumulate',48)
        settings.set('/persistent/app/viewport/displayOptions',0)
        field_count=sum(p.GetTypeName()=='ParticleField3DGaussianSplat' for p in stage.Traverse())
        if not field_count: raise RuntimeError('Scene has no native Gaussian field; refusing point-cloud fallback')
        light=UsdLux.DomeLight.Define(stage,'/World/Lighting/Sky')
        light.CreateIntensityAttr(scenario.get('lighting',{}).get('intensity',1.))
        light.CreateColorAttr((1.,.97,.92))
        actors={}; ops={}
        for name in scenario['spawns']:
            actors[name],ops[name]=add_vehicle(stage,'/World/Vehicles/'+name,scenario['assets']['vehicle_usd'])
        raycaster=VehicleRaycaster(mesh_triangles(stage,actors['opponent']))
        intrinsics=scenario.get('camera',{'width':640,'height':400,'fx':283.435,'fy':291.878,'cx':320.97935,'cy':194.30535})
        camera,camera_op=_camera(stage,'/World/EgoCamera',intrinsics)
        spectator,spectator_op=_camera(stage,'/World/RaceCamera',dict(width=1280,height=800,fx=850,fy=850,cx=640,cy=400))
        race_viewport=viewport.get_active_viewport()
        race_viewport.camera_path='/World/RaceCamera'
        race_viewport.updates_enabled=not args.no_live_viewport
        def screenshot(name):
            previous=race_viewport.updates_enabled
            race_viewport.updates_enabled=True
            for _ in range(4):app.update()
            viewport.capture_viewport_to_file(race_viewport,str(output/name))
            for _ in range(3):app.update()
            race_viewport.updates_enabled=previous
        render_product=rep.create.render_product(str(camera.GetPath()),(intrinsics['width'],intrinsics['height']))
        rgb_annotator=rep.AnnotatorRegistry.get_annotator('rgb'); rgb_annotator.attach([render_product])
        # The stage only contains visual Xforms. PhysX never integrates these cars.
        models={name:(KinematicDynamics() if args.diagnostic_kinematic else GRUDynamics(scenario['assets']['dynamics_checkpoint'])) for name in actors}
        engine=RaceEngine(models,scenario['reference_path'],
            {name:VehicleState(**state) for name,state in scenario['spawns'].items()},
            outer_boundary=scenario.get('outer_boundary'),inner_boundaries=scenario.get('inner_boundaries'),
            dt=1/scenario['rates']['dynamics'],opponent_speed=scenario.get('opponent_speed',1.5),
            lookahead=scenario.get('lookahead',.9))
        controls=KeyboardControl(command_timeout=engine.command_timeout); controls.paused=not args.autostart
        controls.mode=args.mode; engine.set_control_mode(args.mode)
        if not args.headless:
            import omni.appwindow
            input_interface=carb.input.acquire_input_interface()
            keyboard=omni.appwindow.get_default_app_window().get_keyboard()
            def on_key(event,*_):
                key=event.input.name
                if event.type==carb.input.KeyboardEventType.KEY_PRESS: controls.press(key)
                elif event.type==carb.input.KeyboardEventType.KEY_RELEASE: controls.release(key)
                return True
            keyboard_subscription=input_interface.subscribe_to_keyboard_events(keyboard,on_key)
        def make_worker():
            return WorkerClient(args.sensor_python,scenario['assets']['lidar_checkpoint'],
                return_head=scenario.get('lidar',{}).get('return_head','none'),
                resolution=scenario.get('lidar',{}).get('resolution',256),max_range=scenario.get('lidar',{}).get('max_range',8.))
        worker=make_worker()
        worker.start(); worker.reset(engine.episode_id)
        if args.ros:
            from .ros_bridge import RosBridgeClient
            bridge=RosBridgeClient(scan_config={} if args.scan else None); bridge.reset(engine.episode_id)
        schedule=Schedule(scenario['rates']['dynamics'],{k:scenario['rates'][k] for k in ('camera','lidar','viewport')})
        coordinator=RuntimeCoordinator(engine,controls)
        hud=None
        if not args.headless:
            import omni.ui as ui
            window=ui.Window('F1Tenth Race',width=420,height=160)
            with window.frame:
                with ui.VStack():
                    hud=ui.Label('',word_wrap=True)
                    ui.Label('W/S speed · A/D steer · SPACE pause/resume\nM manual · T ROS · P route follower · R reset',word_wrap=True)
        def world_state(state):
            return {**asdict(state),'z':scenario.get('world_base_z',0.)}
        def sync(snapshot):
            for name,state in snapshot.vehicles.items():ops[name].Set(Gf.Matrix4d(pose_matrix(world_state(state)).T.tolist()))
            pose=sensor_pose(world_state(snapshot.vehicles['ego']),scenario['sensors']['camera'])
            pose[:3,:3]=pose[:3,:3]@np.diag([1.,-1.,-1.]) # optical RDF -> USD camera
            camera_op.Set(Gf.Matrix4d(pose.T.tolist()))
            state=snapshot.vehicles['ego']; base=pose_matrix(world_state(state))
            eye=base@np.array([-1.3,0.,.65,1.]); target=base@np.array([1.,0.,.15,1.])
            look=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye[:3]),Gf.Vec3d(*target[:3]),Gf.Vec3d(0,0,1)).GetInverse()
            spectator_op.Set(look)
            if hud is not None:
                label='KINEMATIC DIAGNOSTIC' if args.diagnostic_kinematic else 'Learned GRU'
                calibration=scenario.get('calibration',{}).get('status','unverified')
                hud.text=f'{label} | {snapshot.control_mode} | {snapshot.status}\nTime {snapshot.time_s:.2f}s | laps {dict(snapshot.laps)}\n{snapshot.reason}\nAlignment: {calibration}'
        observation_sequence=0; last_rgb=None; last_points=None
        last_observation_wall=0.
        def capture(snapshot,due):
            nonlocal observation_sequence,last_rgb,last_points,last_observation_wall
            rgb=points=intensity=None
            if 'camera' in due:
                t=time.perf_counter()
                # Render the authored pose without advancing an unrelated timeline.
                rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
                data=np.asarray(rgb_annotator.get_data())
                if (data.ndim!=3 or data.shape[:2]!=(intrinsics['height'],intrinsics['width'])
                        or data.shape[2]<3 or data.dtype!=np.uint8):
                    raise RuntimeError('Native camera did not produce a valid RGB frame')
                rgb=data[:,:,:3].copy(); last_rgb=rgb
                if args.save_frames:
                    from PIL import Image as _Image
                    frames_dir=output/'frames'; frames_dir.mkdir(parents=True,exist_ok=True)
                    _Image.fromarray(rgb).save(frames_dir/f'ep{snapshot.episode_id}_{snapshot.step_index:06d}.png')
                timing['camera_ms'].append((time.perf_counter()-t)*1000)
            elif 'viewport' in due and not args.no_live_viewport:
                app.update()
            if 'lidar' in due:
                t=time.perf_counter()
                rays=generate_rays(scenario.get('lidar',{}).get('points',20000),snapshot.step_index)
                lidar_pose=sensor_pose(world_state(snapshot.vehicles['ego']),scenario['sensors']['lidar'])
                static=worker.render(lidar_pose,rays,episode_id=snapshot.episode_id,
                    sequence=observation_sequence,time_s=snapshot.time_s)
                dynamic=raycaster.render(lidar_pose,rays,world_state(snapshot.vehicles['opponent']))
                fused=compose_ray_samples(static,dynamic,seed=args.seed+snapshot.episode_id*1000000+snapshot.step_index)
                mask=fused.hit_mask
                points=(rays[mask]*fused.ranges[mask,None]).astype(np.float32)
                intensity=fused.intensity[mask].astype(np.float32); last_points=points
                timing['lidar_ms'].append((time.perf_counter()-t)*1000)
            if bridge:
                bridge.publish(snapshot.episode_id,observation_sequence,snapshot.time_s,
                    {k:world_state(v) for k,v in snapshot.vehicles.items()},rgb=rgb,points=points,intensity=intensity,
                    camera_intrinsics=intrinsics,sensor_transforms=scenario['sensors'])
            observation_sequence+=1
            last_observation_wall=time.monotonic()

        def reset_sensors(snapshot):
            nonlocal worker,bridge,observation_sequence,last_rgb,last_points
            schedule.reset(); observation_sequence=0; last_rgb=last_points=None
            # Reset also recovers a client that terminated itself after timeout.
            worker.close(); worker=make_worker(); worker.start(); worker.reset(snapshot.episode_id)
            if bridge:
                bridge.close()
                bridge=RosBridgeClient(scan_config={} if args.scan else None)
                bridge.reset(snapshot.episode_id)
            sync(snapshot)
            capture(snapshot,{'camera','lidar','viewport'})

        sync(engine.snapshot())
        for _ in range(args.warmup):app.update()
        try:
            # Controllers need observation zero before they can issue command one.
            capture(engine.snapshot(),{'camera','lidar','viewport'})
        except Exception as exc:
            coordinator.fail(exc)
        if args.visual_check and coordinator.fault is None:
            from .visual_check import run_visual_check,run_occlusion_check,run_material_check
            appearance=run_visual_check(stage,app,rep,rgb_annotator,output,light,actors['opponent'],
                             ops['opponent'],world_state(engine.snapshot().vehicles['ego']))
            occlusion=run_occlusion_check(stage,app,rep,rgb_annotator,output,light,actors['opponent'],
                                ops['opponent'],world_state(engine.snapshot().vehicles['ego']))
            material=run_material_check(stage,app,rep,rgb_annotator,output,light,actors['opponent'],
                                        ops['opponent'],world_state(engine.snapshot().vehicles['ego']))
            visual_report={'appearance':appearance,'occlusion':occlusion,'material':material}
            if not appearance['passed'] or not occlusion['passed']:
                coordinator.fail('Native visual check failed; inspect appearance/occlusion reports')
            sync(engine.snapshot())
            capture(engine.snapshot(),{'camera','lidar','viewport'})
        # Warm-up, asset loading and visual probes are excluded from runtime timing.
        for values in timing.values():values.clear()
        from PIL import Image
        if last_rgb is not None:Image.fromarray(last_rgb).save(output/'ego_initial.png')
        screenshot('race_initial.png')
        start=time.perf_counter(); pace_deadline=start; samples=0; total_simulated=0.
        trace=open(output/'trajectory.jsonl','w')
        try:
            while app.is_running() and (args.steps<=0 or samples<args.steps):
                tick=time.perf_counter()
                if controls.consume_reset():
                    coordinator.reset(args.seed,reset_sensors)
                before=engine.step_index; previous_time=engine.time_s; t=time.perf_counter()
                snapshot=coordinator.step(bridge.poll_commands if bridge else None)
                timing['dynamics_ms'].append((time.perf_counter()-t)*1000)
                sync(snapshot)
                if snapshot.step_index==before:
                    pace_deadline=time.perf_counter()
                    app.update()
                    if args.headless and (snapshot.status=='finished' or coordinator.fault or
                            (snapshot.status=='paused' and not coordinator.waiting_for_ros)):break
                    # DDS discovery or a controller restart may occur after the
                    # first sample. Re-render the stationary pose with a new
                    # observation sequence while simulation time stays frozen.
                    if (bridge and coordinator.waiting_for_ros and coordinator.fault is None
                            and time.monotonic()-last_observation_wall>=.5):
                        try:capture(snapshot,{'camera','lidar','viewport'})
                        except Exception as exc:coordinator.fail(exc)
                    time.sleep(.005); continue
                total_simulated+=snapshot.time_s-previous_time
                due=schedule.due(snapshot.step_index)
                try:
                    capture(snapshot,due)
                except Exception as exc:
                    snapshot=coordinator.fail(exc)
                    print(snapshot.reason,flush=True)
                trace.write(json.dumps({'episode_id':snapshot.episode_id,'step':snapshot.step_index,'time_s':snapshot.time_s,
                    'status':snapshot.status,'reason':snapshot.reason,'laps':dict(snapshot.laps),
                    'vehicles':{k:asdict(v) for k,v in snapshot.vehicles.items()}})+'\n')
                samples+=1; timing['tick_ms'].append((time.perf_counter()-tick)*1000)
                if snapshot.status=='finished' and args.headless:break
                if not args.unpaced:
                    # Pace against the shared clock, allowing cheap dynamics ticks
                    # to recover time spent on a more expensive sensor tick.
                    pace_deadline+=snapshot.time_s-previous_time
                    time.sleep(max(0.,pace_deadline-time.perf_counter()))
        finally:trace.close()
        elapsed=time.perf_counter()-start
        if last_rgb is not None:Image.fromarray(last_rgb).save(output/'ego_final.png')
        if last_points is not None:np.save(output/'lidar_final.npy',last_points)
        screenshot('race_final.png')
        report={'scenario':scenario['name'],'dynamics':'kinematic diagnostic' if args.diagnostic_kinematic else 'GRU',
                'steps':samples,'simulated_seconds':total_simulated,'episode_seconds':engine.time_s,'wall_seconds':elapsed,
                'real_time_factor':total_simulated/max(elapsed,1e-9),'status':engine.status,'reason':engine.reason,
                'laps':dict(engine.laps),'calibration':scenario.get('calibration',{}),
                'nominal_rates':scenario['rates'],'visual_checks':visual_report,
                'renderer':settings.get('/rtx/rendermode'),'live_viewport':not args.no_live_viewport,
                'gaussian_accumulated_depth':settings.get('/rtx/rtpt/gaussian/accumulatedDepth/enabled'),
                'gaussian_accumulated_albedo':settings.get('/rtx/rtpt/gaussian/accumulatedAlbedo/enabled'),
                'effective_camera_hz':len(timing['camera_ms'])/max(elapsed,1e-9),
                'effective_lidar_hz':len(timing['lidar_ms'])/max(elapsed,1e-9),
                'timing':{key:{'count':len(values),'mean_ms':float(np.mean(values)),
                              'p95_ms':float(np.percentile(values,95))} for key,values in timing.items() if values}}
        (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        stage.GetRootLayer().Export(str(output/'race_scene.usda'))
        if coordinator.fault:raise RuntimeError(coordinator.fault)
    except BaseException as exc:
        import traceback
        exit_code=1
        traceback.print_exc()
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'type':type(exc).__name__},indent=2))
    finally:
        if worker:worker.close()
        if bridge:bridge.close()
        if keyboard_subscription is not None:input_interface.unsubscribe_to_keyboard_events(keyboard,keyboard_subscription)
        app.close(exit_code=exit_code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',default='configs/racing_142014.json')
    parser.add_argument('--sensor-python',default=str(Path.home()/'miniconda3/envs/roboracer/bin/python'))
    parser.add_argument('--headless',action='store_true');parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--mode',choices=['manual','ros','autopilot'],default='manual')
    parser.add_argument('--autostart',action='store_true');parser.add_argument('--ros',action='store_true')
    parser.add_argument('--scan',action='store_true');parser.add_argument('--diagnostic-kinematic',action='store_true')
    parser.add_argument('--unpaced',action='store_true');parser.add_argument('--warmup',type=int,default=60)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--output',default='outputs/racing')
    parser.add_argument('--rebuild-assets',action='store_true')
    parser.add_argument('--visual-check',action='store_true')
    parser.add_argument('--save-frames',action='store_true',help='Write every ego camera frame to <output>/frames/ep<episode>_<step>.png (input for Code/benchmark)')
    parser.add_argument('--no-live-viewport',action='store_true',help='Headless training: render spectator only for saved snapshots')
    args=parser.parse_args()
    if args.no_live_viewport and not args.headless:parser.error('--no-live-viewport requires --headless')
    if args.headless and not args.autostart:parser.error('--headless requires --autostart')
    if args.mode=='ros' and not args.ros:parser.error('--mode ros requires --ros')
    run(args)
