from racing.controls import KeyboardControl
from racing.core import RaceEngine, VehicleState
from racing.dynamics import KinematicDynamics
from racing.isaac_app import RuntimeCoordinator


def coordinator(mode='manual'):
    engine = RaceEngine({'ego':KinematicDynamics()}, [[0,0],[10,0],[10,10],[0,10]],
                        {'ego':VehicleState()})
    controls = KeyboardControl()
    controls.mode = mode
    return RuntimeCoordinator(engine, controls)


def test_ros_waits_for_first_command_and_samples_clock_after_poll():
    runtime = coordinator('ros')
    assert runtime.step(lambda: [], lambda: 1.).reason == 'waiting for ROS command'
    assert runtime.waiting_for_ros
    clock = [1.]
    def poll():
        clock[0] = 2.
        return [dict(speed=1., steering=0., received_at=2., episode_id=0)]
    snapshot = runtime.step(poll, lambda: clock[0])
    assert snapshot.step_index == 1
    assert not runtime.waiting_for_ros


def test_poll_failure_stays_visible_and_cannot_resume_without_reset():
    runtime = coordinator('ros')
    def broken():
        raise EOFError('ROS disconnected')
    snapshot = runtime.step(broken)
    assert snapshot.status == 'paused'
    assert 'ROS disconnected' in snapshot.reason
    runtime.controls.paused = False
    assert runtime.step(lambda: []).reason == snapshot.reason
    assert runtime.engine.step_index == 0


def test_reset_recovery_reinitializes_sensors_before_clearing_fault():
    runtime = coordinator()
    runtime.fail('worker timed out')
    observed = []
    def reset_sensors(snapshot):
        observed.append((snapshot.episode_id, snapshot.step_index, snapshot.time_s))
    assert runtime.reset(7, reset_sensors)
    assert observed == [(1, 0, 0.)]
    assert runtime.fault is None and runtime.controls.paused
    def broken_reset(snapshot):
        raise RuntimeError('restart failed')
    assert not runtime.reset(7, broken_reset)
    assert 'restart failed' in runtime.engine.reason


def test_dynamics_failure_becomes_terminal_runtime_fault():
    runtime = coordinator()
    def broken(*args):
        raise RuntimeError('model failure')
    runtime.engine.models['ego'].predict = broken
    snapshot = runtime.step()
    assert snapshot.status == 'paused'
    assert runtime.fault and 'model failure' in runtime.fault
