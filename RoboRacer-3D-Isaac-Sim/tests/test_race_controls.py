import math

from racing.controls import KeyboardControl


def test_manual_keys_release_and_pause():
    control = KeyboardControl()
    control.press('W')
    control.press('A')
    command = control.command(.1, 0)
    assert command.speed > 0 and command.steering > 0
    control.release('A')
    assert control.command(.1, .1).steering == 0
    control.press('SPACE')
    assert control.paused
    assert control.command(.1, .2).speed == 0
    control.release('SPACE')
    control.press('SPACE')
    assert not control.paused
    assert control.command(.1, .3).speed == 0


def test_reset_pauses_and_preserves_control_mode():
    control = KeyboardControl()
    control.press('T')
    control.reset(episode_id=2)
    assert control.paused
    assert control.mode == 'ros'


def test_ros_timeout_uses_actual_new_commands_and_reset_rejects_stale():
    control = KeyboardControl(command_timeout=.5)
    control.reset(episode_id=3)
    control.paused = False
    control.press('T')
    assert control.receive_ros(2, .1, 10, episode_id=3)
    assert control.command(.1, 10.4).speed == 2
    assert control.command(.1, 10.6).speed == 0
    assert not control.receive_ros(4, .1, 10.7, episode_id=2)
    assert not control.receive_ros(math.nan, .1, 10.7, episode_id=3)
    assert control.receive_ros(3, .1, 10.8, episode_id=3)
    assert control.command(.1, 11).speed == 2.5
    control.reset(episode_id=4)
    assert control.command(.1, 11).speed == 0


def test_reset_is_edge_triggered_and_switching_mode_clears_commands():
    control = KeyboardControl()
    control.press('R')
    assert control.consume_reset()
    control.press('R')
    assert not control.consume_reset()
    control.release('R')
    control.press('R')
    assert control.consume_reset()
    control.press('W')
    control.command(.1, 0)
    control.press('T')
    control.press('M')
    assert control.command(.1, .1).speed == 0
