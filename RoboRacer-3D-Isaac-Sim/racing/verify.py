"""Verify the configured race with independent CPU GRUs and emit a JSON report.

Example:
    python -m racing.verify --scenario configs/racing_142014.json --laps 3 \
        --max-steps 10000 --output outputs/racing/core_report.json

Camera and LiDAR rendering are outside this check. The configured reference
route and boundary proxies are exercised exactly as used by the race engine.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

from .config import load_scenario
from .core import RaceEngine, VehicleState
from .dynamics import GRUDynamics


def _fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_scenario(scenario_path, laps=3, max_steps=10000, seed=0):
    """Run deterministic reset verification followed by the requested race.

    Only the dynamics checkpoint must exist. Rendering assets may be on an
    unmounted external drive; their absence does not change this CPU check.
    """
    if laps<1 or max_steps<1:
        raise ValueError('laps and max_steps must be positive')
    start=monotonic()
    scenario=load_scenario(scenario_path,check_assets=False)
    checkpoint=scenario['assets']['dynamics_checkpoint']
    models={key:GRUDynamics(checkpoint) for key in scenario['spawns']}
    independent=(len({id(model) for model in models.values()})==len(models)
                 and len({id(model.history) for model in models.values()})==len(models))
    engine=RaceEngine(models=models,reference_path=scenario['reference_path'],
                      spawns={key:VehicleState(**value) for key,value in scenario['spawns'].items()},
                      outer_boundary=scenario.get('outer_boundary'),
                      inner_boundaries=scenario.get('inner_boundaries'),
                      opponent_speed=scenario.get('opponent_speed',1.5),
                      lookahead=scenario.get('lookahead',.9),
                      dt=1/scenario['rates']['dynamics'])
    engine.set_control_mode('autopilot')
    prefix_steps=min(64,max_steps)
    prefixes=[]
    for _ in range(2):
        engine.reset(seed=seed);engine.resume()
        prefix=[]
        for _ in range(prefix_steps):
            snapshot=engine.step()
            prefix.append((dict(snapshot.vehicles),dict(snapshot.laps),snapshot.step_index,
                           snapshot.time_s,snapshot.status,snapshot.collision))
            if snapshot.status!='running':break
        prefixes.append(prefix)
    reset_deterministic=prefixes[0]==prefixes[1]
    engine.reset(seed=seed);engine.resume()
    trajectory=hashlib.sha256()
    for _ in range(max_steps):
        snapshot=engine.step()
        states={key:asdict(value) for key,value in snapshot.vehicles.items()}
        trajectory.update(json.dumps(states,sort_keys=True,separators=(',',':')).encode())
        if snapshot.status!='running' or min(snapshot.laps.values())>=laps:
            break
    complete=min(snapshot.laps.values())>=laps
    success=complete and snapshot.status=='running' and reset_deterministic and independent
    failure=None
    if not independent:failure='model instances or histories are shared'
    elif not reset_deterministic:failure='reset replay differs for the same seed'
    elif snapshot.status!='running':failure=snapshot.reason
    elif not complete:failure='maximum step budget exhausted'
    return {
        'success':success,'failure_reason':failure,
        'scenario':str(Path(scenario_path).resolve()),'scenario_name':scenario['name'],
        'dynamics':'GRUDynamics CPU, independent Run30 history per vehicle',
        'checkpoint':checkpoint,'checkpoint_sha256':_fingerprint(checkpoint),
        'seed':seed,'requested_laps':laps,'max_steps':max_steps,
        'steps':snapshot.step_index,'simulation_seconds':snapshot.time_s,
        'wall_seconds':monotonic()-start,'laps':dict(snapshot.laps),
        'status':snapshot.status,'collision':asdict(snapshot.collision) if snapshot.collision else None,
        'independent_models':independent,'reset_deterministic':reset_deterministic,
        'determinism_prefix_steps':prefix_steps,'trajectory_sha256':trajectory.hexdigest(),
        'final_states':states,'reference_length_m':engine.path_length,
        'calibration':scenario.get('calibration'),
        'boundary_kind':scenario.get('route_provenance',{}).get('boundary_kind','configured boundaries'),
        'scope':'CPU learned dynamics, control, reset, lap counting and configured collision proxies; no rendering validation',
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',type=Path,default=Path('configs/racing_142014.json'))
    parser.add_argument('--laps',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('outputs/racing/core_report.json'))
    args=parser.parse_args(argv)
    try:
        report=verify_scenario(args.scenario,laps=args.laps,max_steps=args.max_steps,seed=args.seed)
    except Exception as exc:
        report={'success':False,'failure_reason':f'{type(exc).__name__}: {exc}',
                'scenario':str(args.scenario),'requested_laps':args.laps,'max_steps':args.max_steps}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
