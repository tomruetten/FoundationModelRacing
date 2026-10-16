"""Camera-cache provenance checks do not require Isaac or Torch."""
import json
import numpy as np

from racing.assets import _camera_provenance, _camera_cache_matches, _file_stamp, _camera_scale_mask


def test_camera_cache_invalidates_changed_inputs_and_partial_outputs(tmp_path):
    checkpoint = tmp_path / 'camera.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    output = tmp_path / 'camera.usd'
    output.write_text('wrapper')
    field = tmp_path / 'camera_field.usd'
    field.write_text('field')
    scenario = {'assets': {'camera_checkpoint': str(checkpoint), 'camera_usd': str(output)},
                'camera_to_world': np.eye(4).tolist()}
    provenance = _camera_provenance(scenario)
    sidecar = output.with_suffix('.json')
    sidecar.write_text(json.dumps({'provenance': provenance,
        'artifacts': {'wrapper': _file_stamp(output), 'field': _file_stamp(field)}}))
    assert _camera_cache_matches(output, provenance)

    scenario['camera_field'] = {'max_scale_m': .3}
    assert not _camera_cache_matches(output, _camera_provenance(scenario))
    scenario.pop('camera_field')
    scenario['camera_field'] = {'bake_world': True}
    assert not _camera_cache_matches(output, _camera_provenance(scenario))
    scenario.pop('camera_field')

    scenario['camera_to_world'][0][3] = 1
    assert not _camera_cache_matches(output, _camera_provenance(scenario))
    scenario['camera_to_world'][0][3] = 0
    checkpoint.write_bytes(b'new checkpoint')
    assert not _camera_cache_matches(output, _camera_provenance(scenario))
    field.write_text('partially overwritten field')
    assert not _camera_cache_matches(output, provenance)
    field.unlink()
    assert not _camera_cache_matches(output, provenance)


def test_camera_cache_rejects_missing_or_legacy_provenance(tmp_path):
    output = tmp_path / 'camera.usd'
    output.write_text('wrapper')
    assert not _camera_cache_matches(output, {})
    output.with_suffix('.json').write_text('{invalid')
    assert not _camera_cache_matches(output, {})
    output.with_suffix('.json').write_text(json.dumps({'checkpoint': 'legacy'}))
    assert not _camera_cache_matches(output, {})


def test_camera_scale_filter_uses_world_metres_and_largest_axis():
    scales = np.log([[.1,.1,.1],[.1,.2,.1],[.2,.2,.2]])
    transform = np.eye(4)
    transform[:3,:3] *= 2
    np.testing.assert_equal(_camera_scale_mask(scales,transform,.3),[True,False,False])
    np.testing.assert_equal(_camera_scale_mask(scales,np.eye(4),.3),[True,True,True])
    np.testing.assert_equal(_camera_scale_mask(scales,transform,None),[True,True,True])
