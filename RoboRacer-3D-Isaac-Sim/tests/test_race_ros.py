import math
import os
import socket
import struct
import subprocess
import textwrap

import numpy as np
import pytest

from racing.ros_bridge import (
    FrameGate, SharedPayload, optical_rotation, points_to_scan, recv_frame, send_frame,
)


def test_frame_transport_and_payload_round_trip():
    first, second = socket.socketpair()
    with first, second:
        send_frame(first, {'episode_id': 1, 'sequence': 5})
        assert recv_frame(second) == {'episode_id': 1, 'sequence': 5}
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    points = np.array([[1, 2, 3]], dtype=np.float32)
    with SharedPayload(rgb=rgb, points=points) as payload:
        arrays = SharedPayload.read(payload.descriptors)
        np.testing.assert_array_equal(arrays['rgb'], rgb)
        np.testing.assert_array_equal(arrays['points'], points)


def test_transport_rejects_oversize_before_reading_body():
    first, second = socket.socketpair()
    with first, second:
        first.sendall(struct.pack('!I', 2 ** 30))
        with pytest.raises(ValueError):
            recv_frame(second)


def test_optical_axes_and_scan_nearest_returns():
    # Optical +Z forward, +X right, +Y down expressed in vehicle FLU.
    np.testing.assert_allclose(optical_rotation() @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(optical_rotation() @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_allclose(optical_rotation() @ [0, 1, 0], [0, 0, -1])
    points = np.array([[2, 0, 0], [1, 0, 0], [3, 0, 1], [0, 2, 0]])
    ranges = points_to_scan(points, bins=4, z_min=-.1, z_max=.1)
    assert ranges[2] == 1
    assert ranges[3] == 2
    assert math.isinf(ranges[0]) and math.isinf(ranges[1])


def test_reset_gate_drops_old_episode_and_repeated_frames():
    gate = FrameGate()
    assert gate.accept(0, 1, .1)
    assert not gate.accept(0, 1, .1)
    assert not gate.accept(0, 2, .05)
    gate.reset(1)
    assert not gate.accept(0, 9, 9)
    assert gate.accept(1, 0, 0)


@pytest.mark.skipif(os.environ.get('ROBORACER_TEST_ROS') != '1',
                    reason='Set ROBORACER_TEST_ROS=1 for installed ROS integration')
def test_ros_odometry_preserves_learned_body_velocities():
    script = textwrap.dedent('''
        import math
        import rclpy
        from nav_msgs.msg import Odometry
        from tf2_msgs.msg import TFMessage
        from racing.ros_bridge import RosBridgeClient
        rclpy.init()
        node = rclpy.create_node('body_velocity_regression')
        received = []
        transforms = []
        subscription = node.create_subscription(Odometry, '/debug/ego/odom', received.append, 1)
        tf_subscription = node.create_subscription(TFMessage, '/tf', transforms.append, 10)
        bridge = RosBridgeClient()
        try:
            state = dict(x=0., y=0., yaw=math.pi/2, vx=1., vy=.2, yr=.1)
            # Inverse optical calibration quaternion, from measured T_camera_lidar.
            camera_mount = dict(translation=[.29267,.06987,.06559],
                                quaternion=[-.474311,.49898,-.50795,.517721])
            for sequence in range(60):
                bridge.publish(0, sequence, sequence*.02, {'ego':state},
                               sensor_transforms={'camera':camera_mount})
                rclpy.spin_once(node, timeout_sec=.05)
                if received and transforms:
                    break
            assert received, 'No ROS odometry received'
            odom = received[-1]
            assert odom.child_frame_id == 'ego/base_link'
            assert abs(odom.twist.twist.linear.x - 1.) < 1e-6, odom.twist
            assert abs(odom.twist.twist.linear.y - .2) < 1e-6, odom.twist
            optical = next(t for msg in transforms for t in msg.transforms
                           if t.child_frame_id == 'ego/camera_optical_frame')
            assert optical.header.frame_id == 'ego/base_link'
            q, p = optical.transform.rotation, optical.transform.translation
            assert [q.x,q.y,q.z,q.w] == camera_mount['quaternion']
            assert [p.x,p.y,p.z] == camera_mount['translation']
        finally:
            bridge.close()
            node.destroy_node()
            rclpy.shutdown()
    ''')
    subprocess.run(['bash', '-c',
                    'source /opt/ros/humble/setup.bash && exec /usr/bin/python3 -'],
                   input=script, text=True, check=True, timeout=30)
