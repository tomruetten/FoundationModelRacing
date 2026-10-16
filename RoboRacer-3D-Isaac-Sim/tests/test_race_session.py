"""Portable calibration/route checks and the committed session's CPU GRU run."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from racing.prepare_session import fit_similarity, closed_route
from racing.verify import verify_scenario

REPO=Path(__file__).resolve().parents[1]
SCENARIO=REPO/'configs/racing_142014.json'


def test_recover_metric_similarity():
    source=np.random.default_rng(0).normal(size=(100,3))
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    target=3.7*source@rotation.T+[1,2,3]
    transform,residual=fit_similarity(source,target)
    assert np.allclose(transform[:3,:3],3.7*rotation)
    assert np.allclose(transform[:3,3],[1,2,3])
    assert max(residual)<1e-10


def test_reject_reflected_3d_correspondences():
    source=np.random.default_rng(0).normal(size=(100,3))
    target=source*np.array([-1,1,1])
    with pytest.raises(ValueError,match='reflection'):
        fit_similarity(source,target)


def test_first_closed_lap_is_smooth_and_metric():
    angle=np.linspace(0,5*np.pi,501)
    positions=np.c_[4*np.cos(angle),4*np.sin(angle),0*angle]
    path,end=closed_route(positions)
    assert 195<=end<=205
    assert len(path)>100
    assert np.linalg.norm(path[0]-path[-1])<.2
    assert np.ptp(path,axis=0).min()>7.5
    assert np.max(np.linalg.norm(np.roll(path,-1,axis=0)-path,axis=1))<.2


def test_open_route_is_not_invented_into_a_lap():
    with pytest.raises(ValueError,match='No supported first lap'):
        closed_route(np.c_[np.linspace(0,50,501),np.zeros((501,2))])


def test_committed_session_three_laps_with_actual_gru():
    report=verify_scenario(SCENARIO,laps=3,max_steps=10000)
    assert report['success'],report
    assert min(report['laps'].values())>=3
    assert report['reset_deterministic']
    assert report['independent_models']
    assert report['collision'] is None
    assert report['calibration']['status']=='provisional'


def test_cli_writes_failure_artifact_when_step_budget_exhausted(tmp_path):
    output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,'-m','racing.verify','--scenario',str(SCENARIO),
                           '--laps','3','--max-steps','2','--output',str(output)],
                          cwd=REPO,capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    report=json.loads(output.read_text())
    assert report['success'] is False
    assert report['steps']==2
    assert report['failure_reason']=='maximum step budget exhausted'


def test_camera_intrinsics_and_repo_asset_paths_are_explicit():
    scenario=json.loads(SCENARIO.read_text())
    assert scenario['camera']['width']==640 and scenario['camera']['height']==400
    assert scenario['camera']['fx']==pytest.approx(566.8698136397473/2)
    assert scenario['camera']['fy']==pytest.approx(583.7564417421023/2)
    for key in ['camera_usd','dynamics_checkpoint']:
        assert not Path(scenario['assets'][key]).is_absolute()


def test_opponent_starts_about_one_and_half_metres_ahead():
    scenario=json.loads(SCENARIO.read_text())
    path=np.asarray(scenario['reference_path'])
    opponent=scenario['spawns']['opponent']
    index=int(np.argmin(np.linalg.norm(path-[opponent['x'],opponent['y']],axis=1)))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    assert arc[index]==pytest.approx(1.5,abs=.07)


def test_verifier_honors_configured_follower_settings(tmp_path):
    from dataclasses import asdict
    from racing.config import load_scenario
    from racing.core import RaceEngine, VehicleState
    from racing.dynamics import GRUDynamics

    scenario=load_scenario(SCENARIO,check_assets=False)
    scenario.update(opponent_speed=.8,lookahead=.4)
    custom=tmp_path/'custom.json';custom.write_text(json.dumps(scenario))
    report=verify_scenario(custom,laps=3,max_steps=2)
    engine=RaceEngine(models={key:GRUDynamics(scenario['assets']['dynamics_checkpoint'])
                              for key in scenario['spawns']},
                      reference_path=scenario['reference_path'],
                      spawns={key:VehicleState(**value) for key,value in scenario['spawns'].items()},
                      opponent_speed=.8,lookahead=.4,
                      outer_boundary=scenario['outer_boundary'],
                      inner_boundaries=scenario['inner_boundaries'])
    engine.set_control_mode('autopilot');engine.resume()
    engine.step();expected=engine.step()
    assert report['final_states']=={key:asdict(value) for key,value in expected.vehicles.items()}
