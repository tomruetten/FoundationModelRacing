import json
import numpy as np
import pytest
from racing.config import load_scenario, pose_matrix, sensor_pose, Schedule


def scenario():
    return {'name':'test','assets':{},'reference_path':[[0,0],[4,0],[4,4],[0,4]],
            'spawns':{'ego':{'x':0,'y':0,'yaw':0},'opponent':{'x':2,'y':0,'yaw':0}},
            'rates':{'dynamics':50,'camera':20,'lidar':10,'viewport':30},
            'sensors':{'camera':{'translation':[.2,0,.1],'quaternion':[0,0,0,1]},
                       'lidar':{'translation':[.27,0,.11],'quaternion':[0,0,0,1]}}}


def test_mount_rotates_with_vehicle():
    p=sensor_pose({'x':1,'y':2,'yaw':np.pi/2},{'translation':[.27,0,.11],'quaternion':[0,0,0,1]})
    np.testing.assert_allclose(p[:3,3],[1,2.27,.11],atol=1e-8)
    np.testing.assert_allclose(p[:3,0],[0,1,0],atol=1e-8)


def test_schedule_has_exact_rates_on_fixed_clock():
    s=Schedule(50,{'camera':20,'lidar':10,'viewport':30})
    counts={key:0 for key in s.rates}
    for step in range(1,51):
        for key in s.due(step): counts[key]+=1
    assert counts=={'camera':20,'lidar':10,'viewport':30}
    s.reset()
    assert s.due(1)==set()


def test_load_paths_relative_to_config_and_validate(tmp_path):
    d=scenario(); d['assets']['camera_usd']='track.usd'
    p=tmp_path/'s.json'; p.write_text(json.dumps(d))
    s=load_scenario(p,check_assets=False)
    assert s['assets']['camera_usd']==str(tmp_path/'track.usd')
    with pytest.raises(FileNotFoundError): load_scenario(p)
    d['rates']['lidar']=100; p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='rate'):load_scenario(p,check_assets=False)


@pytest.mark.parametrize('change, message', [
    (lambda d: d['sensors'].pop('lidar'), 'camera and lidar'),
    (lambda d: d['sensors']['camera'].pop('quaternion'), 'quaternion'),
    (lambda d: d['spawns']['ego'].pop('yaw'), 'x, y, and yaw'),
    (lambda d: d['spawns']['ego'].update(x=float('nan')), 'finite'),
    (lambda d: d['spawns']['ego'].update(vx='fast'), 'finite'),
    (lambda d: d.update(exclude_world_boxes=[{'min': [0,0,0], 'max': [1,1,1]}]),
     'both camera and LiDAR'),
])
def test_invalid_sensor_spawn_and_one_modality_exclusions_fail_early(tmp_path, change, message):
    d = scenario()
    change(d)
    p = tmp_path / 'scenario.json'
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=message):
        load_scenario(p, check_assets=False)


@pytest.mark.parametrize('maximum', [0,-.1,float('nan'),float('inf'),True,'large'])
def test_camera_field_scale_limit_must_be_positive_finite(tmp_path, maximum):
    d = scenario()
    d['camera_field'] = {'max_scale_m': maximum}
    p = tmp_path/'scenario.json'
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='max_scale_m'):
        load_scenario(p, check_assets=False)


def test_camera_world_bake_requires_explicit_boolean(tmp_path):
    d=scenario(); d['camera_field']={'bake_world':'false'}
    p=tmp_path/'scenario.json'; p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='bake_world'):
        load_scenario(p,check_assets=False)
