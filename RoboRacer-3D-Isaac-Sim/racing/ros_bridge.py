"""ROS Humble worker isolated from Isaac Sim's incompatible Python runtime.

Only the worker imports ROS. Control uses bounded JSON on a mode-0700 Unix
socket directory; image/point arrays live in shared memory for one synchronous
exchange. The receiver acknowledges before the producer reclaims the arrays.
"""
import argparse
from collections import deque
import json
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np

MAX_FRAME_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024


def send_frame(connection, message):
    data = json.dumps(message, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError('ROS bridge control frame exceeds size bound')
    connection.sendall(struct.pack('!I', len(data)) + data)


def _read_exact(connection, count):
    pieces = bytearray()
    while len(pieces) < count:
        data = connection.recv(count - len(pieces))
        if not data:
            raise EOFError('ROS bridge disconnected')
        pieces.extend(data)
    return bytes(pieces)


def recv_frame(connection):
    length, = struct.unpack('!I', _read_exact(connection, 4))
    if length > MAX_FRAME_BYTES:
        raise ValueError('ROS bridge control frame exceeds size bound')
    return json.loads(_read_exact(connection, length))


class SharedPayload:
    def __init__(self, **arrays):
        self.blocks = []
        self.descriptors = {}
        try:
            for name, values in arrays.items():
                if values is None:
                    continue
                dtype = np.uint8 if name == 'rgb' else np.float32
                array = np.ascontiguousarray(values, dtype=dtype)
                if array.nbytes > MAX_ARRAY_BYTES:
                    raise ValueError('ROS bridge array exceeds size bound')
                block = shared_memory.SharedMemory(create=True, size=max(1, array.nbytes))
                self.blocks.append(block)
                np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)[:] = array
                self.descriptors[name] = dict(name=block.name, shape=list(array.shape), dtype=array.dtype.str)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def read(descriptors, independent_worker=False):
        arrays = {}
        for key, descriptor in descriptors.items():
            dtype = np.dtype(descriptor['dtype'])
            shape = tuple(descriptor['shape'])
            if dtype not in (np.dtype('uint8'), np.dtype('float32')):
                raise ValueError('Unexpected sensor dtype')
            if len(shape) > 3 or any(not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError('Invalid sensor shape')
            size = math.prod(shape) * dtype.itemsize
            if size > MAX_ARRAY_BYTES:
                raise ValueError('ROS bridge array exceeds size bound')
            block = shared_memory.SharedMemory(name=descriptor['name'])
            try:
                if size > block.size:
                    raise ValueError('Sensor descriptor exceeds shared memory')
                arrays[key] = np.ndarray(shape, dtype=dtype, buffer=block.buf).copy()
            finally:
                block.close()
                if independent_worker:
                    # Producer owns unlink. Worker is an independent interpreter,
                    # hence its resource tracker must not unlink producer memory.
                    from multiprocessing import resource_tracker
                    resource_tracker.unregister(block._name, 'shared_memory')
        return arrays

    def close(self):
        for block in self.blocks:
            block.close()
            block.unlink()
        self.blocks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FrameGate:
    def __init__(self):
        self.reset(0)

    def reset(self, episode_id):
        self.episode_id = int(episode_id)
        self.sequence = -1
        self.time_s = -math.inf

    def accept(self, episode_id, sequence, time_s):
        if (episode_id != self.episode_id or sequence <= self.sequence
                or not math.isfinite(time_s) or time_s < self.time_s):
            return False
        self.sequence, self.time_s = sequence, time_s
        return True


def optical_rotation():
    """Rotation optical RDF axes to vehicle FLU axes."""
    return np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def points_to_scan(points, bins=1080, z_min=-.05, z_max=.05,
                   range_min=.05, range_max=30., angle_min=-math.pi,
                   angle_max=math.pi):
    if bins <= 0 or angle_max <= angle_min:
        raise ValueError('Invalid scan angular configuration')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    distances = np.linalg.norm(points[:, :2], axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] >= z_min)
             & (points[:, 2] <= z_max) & (distances >= range_min)
             & (distances <= range_max) & (angles >= angle_min) & (angles < angle_max))
    indices = np.floor((angles[valid] - angle_min) * bins / (angle_max - angle_min)).astype(int)
    ranges = np.full(bins, np.inf)
    np.minimum.at(ranges, indices, distances[valid])
    return ranges


class RosBridgeClient:
    def __init__(self, scan_config=None, start_timeout=15., exchange_timeout=2.):
        self.directory = tempfile.TemporaryDirectory(prefix='roboracer-ros-')
        self.connection = None
        self.process = None
        self.commands = deque(maxlen=16)
        self.episode_id = 0
        self.gate = FrameGate()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path = os.path.join(self.directory.name, 'bridge.sock')
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(start_timeout)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
        env.pop('PYTHONHOME', None)
        # Isaac carries extension libraries incompatible with the ROS interpreter.
        env['LD_LIBRARY_PATH'] = '/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu'
        command = ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec /usr/bin/python3 -m racing.ros_bridge --worker "$1"', 'ros-worker', path]
        try:
            self.process = subprocess.Popen(command, env=env)
            self.connection, _ = listener.accept()
            self.connection.settimeout(exchange_timeout)
            send_frame(self.connection, {'op': 'configure', 'scan_config': scan_config})
            response = recv_frame(self.connection)
            if response.get('status') != 'ready':
                raise RuntimeError(f'ROS worker failed: {response}')
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    def _exchange(self, message):
        send_frame(self.connection, message)
        response = recv_frame(self.connection)
        if response.get('error'):
            raise RuntimeError(response['error'])
        for command in response.get('commands', []):
            if command['episode_id'] == self.episode_id:
                self.commands.append(command)
        return response

    def publish(self, episode_id, sequence, time_s, states, rgb=None, points=None,
                intensity=None, camera_intrinsics=None, sensor_transforms=None):
        if not self.gate.accept(episode_id, sequence, time_s):
            return False
        with SharedPayload(rgb=rgb, points=points, intensity=intensity) as payload:
            self._exchange(dict(op='publish', episode_id=episode_id, sequence=sequence,
                                time_s=time_s, states=states, arrays=payload.descriptors,
                                camera_intrinsics=camera_intrinsics,
                                sensor_transforms=sensor_transforms or {}))
        return True

    def poll_commands(self):
        self._exchange({'op': 'poll'})
        result = list(self.commands)
        self.commands.clear()
        return result

    def reset(self, episode_id=None):
        self.episode_id = self.episode_id + 1 if episode_id is None else int(episode_id)
        self.gate.reset(self.episode_id)
        self.commands.clear()
        self._exchange({'op': 'reset', 'episode_id': self.episode_id})

    def close(self):
        if self.connection is not None:
            try:
                send_frame(self.connection, {'op': 'close'})
            except OSError:
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        self.directory.cleanup()


def _worker(socket_path):
    # All ABI-specific imports are confined to the system-Python subprocess.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from ackermann_msgs.msg import AckermannDriveStamped
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
    from tf2_ros import TransformBroadcaster

    def stamp(seconds):
        nanoseconds = int(round(seconds * 1_000_000_000))
        return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)

    rclpy.init()
    node = Node('roboracer_bridge')
    pending = deque(maxlen=16)
    gate = FrameGate()
    publishers = {
        'clock': node.create_publisher(Clock, '/clock', 1),
        'image': node.create_publisher(Image, '/ego/camera/image_raw', qos_profile_sensor_data),
        'info': node.create_publisher(CameraInfo, '/ego/camera/camera_info', qos_profile_sensor_data),
        'points': node.create_publisher(PointCloud2, '/ego/lidar/points', qos_profile_sensor_data),
        'scan': node.create_publisher(LaserScan, '/ego/scan', qos_profile_sensor_data),
        'ego': node.create_publisher(Odometry, '/debug/ego/odom', 1),
        'opponent': node.create_publisher(Odometry, '/debug/opponent/odom', 1),
    }
    broadcaster = TransformBroadcaster(node)

    def receive_drive(message):
        speed, steering = float(message.drive.speed), float(message.drive.steering_angle)
        if not math.isfinite(speed) or not math.isfinite(steering):
            return
        pending.append(dict(speed=speed, steering=steering,
                            received_at=time.monotonic(), episode_id=gate.episode_id))

    subscription = node.create_subscription(AckermannDriveStamped, '/ego/drive', receive_drive, 1)

    def spin_ready():
        # A bounded number prevents a fast external publisher starving simulation.
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.)

    def transform(parent, child, translation, quaternion, timestamp):
        result = TransformStamped()
        result.header.stamp, result.header.frame_id = timestamp, parent
        result.child_frame_id = child
        result.transform.translation.x, result.transform.translation.y, result.transform.translation.z = map(float, translation)
        result.transform.rotation.x, result.transform.rotation.y, result.transform.rotation.z, result.transform.rotation.w = map(float, quaternion)
        return result

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_path)
    connection.settimeout(5.)
    scan_config = None
    try:
        while rclpy.ok():
            spin_ready()
            if not select.select([connection], [], [], .005)[0]:
                continue
            request = recv_frame(connection)
            operation = request.get('op')
            if operation == 'close':
                break
            if operation == 'configure':
                scan_config = request.get('scan_config')
                send_frame(connection, {'status': 'ready'})
                continue
            if operation == 'reset':
                node.destroy_subscription(subscription)
                pending.clear()
                gate.reset(request['episode_id'])
                subscription = node.create_subscription(AckermannDriveStamped, '/ego/drive', receive_drive, 1)
                publishers['clock'].publish(Clock(clock=stamp(0.)))
            elif operation == 'publish' and gate.accept(request['episode_id'], request['sequence'], request['time_s']):
                timestamp = stamp(request['time_s'])
                publishers['clock'].publish(Clock(clock=timestamp))
                transforms = []
                for car, state in request['states'].items():
                    if car not in ('ego', 'opponent'):
                        continue
                    yaw = float(state['yaw'])
                    quaternion = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
                    translation = [state['x'], state['y'], state.get('z', 0.)]
                    transforms.append(transform('map', f'{car}/base_link', translation, quaternion, timestamp))
                    odom = Odometry()
                    odom.header.stamp, odom.header.frame_id = timestamp, 'map'
                    odom.child_frame_id = f'{car}/base_link'
                    odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, translation)
                    odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = quaternion[2:]
                    # Learned dynamics outputs are already in vehicle/body axes;
                    # Odometry twist uses its child_frame_id, ego/base_link.
                    odom.twist.twist.linear.x = float(state.get('vx', 0))
                    odom.twist.twist.linear.y = float(state.get('vy', 0))
                    odom.twist.twist.angular.z = float(state.get('yr', 0))
                    publishers[car].publish(odom)
                sensor_transforms = request.get('sensor_transforms') or {}
                for sensor in ('camera', 'lidar'):
                    config = sensor_transforms.get(sensor, {})
                    # Scenario camera calibration directly maps optical RDF to
                    # body; lidar calibration maps sensor FLU to body. Applying
                    # another optical rotation would corrupt camera alignment.
                    child = 'ego/camera_optical_frame' if sensor == 'camera' else 'ego/lidar_link'
                    default_quaternion = [-.5, .5, -.5, .5] if sensor == 'camera' else [0., 0., 0., 1.]
                    transforms.append(transform('ego/base_link', child,
                                                config.get('translation', [0., 0., .2]),
                                                config.get('quaternion', default_quaternion), timestamp))
                broadcaster.sendTransform(transforms)
                arrays = SharedPayload.read(request.get('arrays', {}), independent_worker=True)
                if 'rgb' in arrays:
                    rgb = arrays['rgb']
                    if rgb.ndim != 3 or rgb.shape[2] != 3:
                        raise ValueError('RGB must be HxWx3')
                    image = Image()
                    image.header.stamp, image.header.frame_id = timestamp, 'ego/camera_optical_frame'
                    image.height, image.width = rgb.shape[:2]
                    image.encoding, image.step, image.is_bigendian = 'rgb8', image.width * 3, 0
                    image.data = rgb.tobytes()
                    publishers['image'].publish(image)
                    intrinsics = request.get('camera_intrinsics') or {}
                    fx = float(intrinsics.get('fx', image.width / 2))
                    fy = float(intrinsics.get('fy', fx))
                    cx = float(intrinsics.get('cx', image.width / 2))
                    cy = float(intrinsics.get('cy', image.height / 2))
                    info = CameraInfo()
                    info.header, info.height, info.width = image.header, image.height, image.width
                    info.distortion_model, info.d = 'plumb_bob', [0.] * 5
                    info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
                    info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
                    info.p = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
                    publishers['info'].publish(info)
                if 'points' in arrays:
                    points = arrays['points'].reshape(-1, 3)
                    intensity = arrays.get('intensity', np.ones(len(points), dtype=np.float32)).reshape(-1)
                    if len(intensity) != len(points):
                        raise ValueError('Intensity count must equal point count')
                    records = np.column_stack((points, intensity)).astype('<f4')
                    cloud = PointCloud2()
                    cloud.header.stamp, cloud.header.frame_id = timestamp, 'ego/lidar_link'
                    cloud.height, cloud.width = 1, len(points)
                    cloud.fields = [PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                                    for index, name in enumerate(('x', 'y', 'z', 'intensity'))]
                    cloud.is_bigendian, cloud.point_step, cloud.row_step = False, 16, len(points) * 16
                    cloud.is_dense, cloud.data = bool(np.isfinite(records).all()), records.tobytes()
                    publishers['points'].publish(cloud)
                    if scan_config is not None:
                        ranges = points_to_scan(points, **scan_config)
                        scan = LaserScan()
                        scan.header = cloud.header
                        scan.angle_min = float(scan_config.get('angle_min', -math.pi))
                        scan.angle_increment = (float(scan_config.get('angle_max', math.pi)) - scan.angle_min) / len(ranges)
                        scan.angle_max = scan.angle_min + (len(ranges) - 1) * scan.angle_increment
                        scan.range_min = float(scan_config.get('range_min', .05))
                        scan.range_max = float(scan_config.get('range_max', 30.))
                        scan.ranges = ranges.astype(np.float32).tolist()
                        publishers['scan'].publish(scan)
            spin_ready()
            send_frame(connection, {'status': 'ok', 'commands': list(pending)})
            pending.clear()
    finally:
        connection.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    _worker(parser.parse_args().worker)
