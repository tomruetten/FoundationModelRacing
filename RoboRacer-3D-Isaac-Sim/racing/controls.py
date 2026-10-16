"""Keyboard intent and fail-closed command selection, independent of Carb/ROS."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DriveCommand:
    speed: float = 0.0
    steering: float = 0.0


class KeyboardControl:
    def __init__(self, max_speed=2.5, max_steering=.29, acceleration=1.5,
                 command_timeout=.5, initial_speed=1.5):
        self.max_speed = float(max_speed)
        self.max_steering = float(max_steering)
        self.acceleration = float(acceleration)
        self.command_timeout = float(command_timeout)
        self.initial_speed = float(initial_speed)
        self.mode = 'manual'
        self.paused = False
        self.episode_id = 0
        self._keys = set()
        self._speed = min(self.max_speed, self.initial_speed)
        self._ros = DriveCommand()
        self._ros_time = -math.inf
        self._reset_requested = False

    def press(self, key):
        key = str(key).upper()
        if key in self._keys:
            return
        self._keys.add(key)
        if key == 'SPACE':
            self.paused = not self.paused
            if self.paused:
                self._speed = 0.0
                self._keys.difference_update(('W', 'S', 'A', 'D'))
            self._ros_time = -math.inf
        elif key == 'R':
            self._reset_requested = True
        elif key in ('M', 'T', 'P'):
            mode = {'M': 'manual', 'T': 'ros', 'P': 'autopilot'}[key]
            if mode != self.mode:
                self.mode = mode
                self._keys.clear()
                self._speed = 0.0
                self._ros_time = -math.inf

    def release(self, key):
        self._keys.discard(str(key).upper())

    def consume_reset(self):
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def receive_ros(self, speed, steering, now, episode_id=None):
        if episode_id is not None and episode_id != self.episode_id:
            return False
        if not all(math.isfinite(value) for value in (speed, steering, now)):
            return False
        if now < self._ros_time:
            return False
        self._ros = DriveCommand(max(0., min(self.max_speed, speed)),
                                 max(-self.max_steering, min(self.max_steering, steering)))
        self._ros_time = now
        return True

    def command(self, dt, now):
        if self.paused:
            return DriveCommand()
        if self.mode == 'ros':
            return self._ros if 0 <= now - self._ros_time <= self.command_timeout else DriveCommand()
        if self.mode == 'autopilot':
            return DriveCommand()  # The runtime supplies its diagnostic route follower.
        delta = int('W' in self._keys) - int('S' in self._keys)
        self._speed = max(0., min(self.max_speed, self._speed + delta * self.acceleration * max(0., dt)))
        steering = self.max_steering * (int('A' in self._keys) - int('D' in self._keys))
        return DriveCommand(self._speed, steering)

    def reset(self, episode_id=None):
        self.episode_id = self.episode_id + 1 if episode_id is None else int(episode_id)
        self.paused = True
        self._keys.clear()
        self._speed = 0.0
        self._ros = DriveCommand()
        self._ros_time = -math.inf
        self._reset_requested = False
