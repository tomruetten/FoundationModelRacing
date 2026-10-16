"""
extract_bag_for_felix.py — Extract our ROS2 bag to the 4 CSV files
that Felix's DataPreparation.py expects.

Run from inside ma_jonathanmohr-main/:
  python3 extract_bag_for_felix.py --bag /path/to/all_data_0.db3 --name session_20260721

Outputs (in data/<name>/):
  <name>_pose.csv     — timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w
  <name>_controls.csv — timestamp, speed_cmd, steering_cmd
  <name>_odom.csv     — timestamp, speed, angular_velocity
  <name>_imu.csv      — timestamp, IMU_linear_acceleration_x/y/z, IMU_angular_velocity_x/y/z

Topic → field mapping (confirmed from CDR byte decoding):
  /vrpn_mocap/Chassis2/pose  → pose CSV
  /ackermann_cmd             → speed_cmd  (drive.speed, offset 28, float32)
  /sensors/servo_position_command → steering_cmd (offset 4, float64)
                                    converted via: (servo - 0.5104) / -1.1735
  /sensors/core (VescStateStamped) → odom speed (offset 76, float64, ERPM→m/s)
  /sensors/imu (VescImuStamped)    → odom angular_velocity (offset 36, float64, deg/s→rad/s)
  /livox/imu (sensor_msgs/Imu)     → IMU accel+gyro (offsets 228/132, float64)

Coordinate notes:
  - MoCap world frame: Z-UP, x=forward, y=left (standard ROS)
  - Livox IMU: mounted with livox box on car; axes may differ from Felix's IMU
    Felix's DataPreparation applies: IMU_x_local = -IMU_y, IMU_y_local = IMU_x
    Adjust LIVOX_SWAP below if needed after visual inspection.
"""

import argparse
import os
import sqlite3
import struct

import numpy as np
import pandas as pd

# Servo to steering angle calibration (from f1tenth_stack/config/vesc.yaml)
SERVO_OFFSET = 0.5104
SERVO_GAIN   = -1.1735

# VESC ERPM to m/s — use the calibrated value from f1tenth_stack/config/vesc.yaml
# (speed_to_erpm_gain: 7614.0) rather than the physical formula, which gave ~5832
# (30% off) due to imprecise wheel_r/gear_ratio constants (Issue 6).
# Note: DD and GRU training use speed_cmd, not odom_speed, so this only affects
# the odom_speed column in the processed CSVs.
ERPM_TO_MS = 1.0 / 7614.0


def read_float64(b, off):
    return struct.unpack_from('<d', b, off)[0]


def read_float32(b, off):
    return struct.unpack_from('<f', b, off)[0]


def _fix_mocap_freeze(df, freeze_threshold=0.50):
    """
    Detect and interpolate MoCap position-freeze artifacts (Issue 3.6).

    Motive outputs the last known position when tracking is lost, producing
    runs of identical rows.  Differentiating frozen positions yields spurious
    velocity/rate spikes that corrupt training labels.

    Strategy:
      1. Mark rows whose (p_x, p_y, p_z) is identical to the previous row.
      2. If frozen fraction > freeze_threshold → raise RuntimeError (session unusable).
      3. Otherwise linearly interpolate p_x/p_y/p_z/q_x/q_y/q_z/q_w across
         each frozen run using the surrounding real samples.
    """
    pos_cols = ['p_x', 'p_y', 'p_z']
    quat_cols = ['q_x', 'q_y', 'q_z', 'q_w']

    frozen = (df[pos_cols].diff().abs().sum(axis=1) == 0)
    frozen.iloc[0] = False  # first row can't be compared
    frac = frozen.mean()

    if frac >= freeze_threshold:
        raise RuntimeError(
            f"MoCap frozen {frac*100:.1f}% of session — unusable for training. "
            "Re-record or exclude this session."
        )

    if frac > 0:
        print(f"  [freeze fix] {frozen.sum()} frozen rows ({frac*100:.2f}%) → interpolating")
        for col in pos_cols + quat_cols:
            df[col] = df[col].where(~frozen).interpolate(method='index')

    return df, frac


def extract_pose(conn):
    """
    /vrpn_mocap/Chassis2/pose  (geometry_msgs/PoseStamped)
    CDR: 4(header)+4(sec)+4(nsec)+4+6(frame_id "world\0")+2(pad) = 24 → position at 28

    Also applies MoCap freeze detection + linear interpolation (Issue 3.6).
    Raises RuntimeError if >50% of pose samples are frozen (session unusable).
    """
    tid = conn.execute("SELECT id FROM topics WHERE name='/vrpn_mocap/Chassis2/pose'").fetchone()[0]
    rows = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)).fetchall()

    ts, px, py, pz, qx, qy, qz, qw = [], [], [], [], [], [], [], []
    for t, data in rows:
        b = bytes(data)
        try:
            x, y, z = struct.unpack_from('<3d', b, 28)
            ox, oy, oz, ow = struct.unpack_from('<4d', b, 52)
            if abs(ow) <= 1.1:
                ts.append(t * 1e-9)
                px.append(x); py.append(y); pz.append(z)
                qx.append(ox); qy.append(oy); qz.append(oz); qw.append(ow)
        except Exception:
            pass

    df = pd.DataFrame({'timestamp': ts, 'p_x': px, 'p_y': py, 'p_z': pz,
                       'q_x': qx, 'q_y': qy, 'q_z': qz, 'q_w': qw})
    df, _ = _fix_mocap_freeze(df)
    return df


def extract_controls(conn):
    """
    speed_cmd:    /ackermann_cmd  drive.speed  (float32 at offset 28)
    steering_cmd: /sensors/servo_position_command  (float64 at offset 4)
                  → (servo - SERVO_OFFSET) / SERVO_GAIN  [radians]
    """
    # speed_cmd
    tid_ack = conn.execute("SELECT id FROM topics WHERE name='/ackermann_cmd'").fetchone()[0]
    rows_ack = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid_ack,)).fetchall()
    ts_ack, spd = [], []
    for t, data in rows_ack:
        b = bytes(data)
        if len(b) >= 32:
            ts_ack.append(t * 1e-9)
            spd.append(read_float32(b, 28))

    # steering_cmd
    tid_srv = conn.execute("SELECT id FROM topics WHERE name='/sensors/servo_position_command'").fetchone()[0]
    rows_srv = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid_srv,)).fetchall()
    ts_srv, steer = [], []
    for t, data in rows_srv:
        b = bytes(data)
        if len(b) >= 12:
            servo = read_float64(b, 4)
            ts_srv.append(t * 1e-9)
            steer.append((servo - SERVO_OFFSET) / SERVO_GAIN)

    # Merge on time by interpolating steering onto ackermann timestamps
    ts_ack = np.array(ts_ack)
    spd    = np.array(spd)
    ts_srv = np.array(ts_srv)
    steer  = np.array(steer)
    steer_at_ack = np.interp(ts_ack, ts_srv, steer)

    return pd.DataFrame({'timestamp': ts_ack, 'speed_cmd': spd, 'steering_cmd': steer_at_ack})


def extract_odom(conn):
    """
    speed:            /sensors/core  VescStateStamped.state.speed        (float64 at offset 76, ERPM)
                      → ERPM × ERPM_TO_MS  [m/s]
    current_motor:    /sensors/core  VescStateStamped.state.current_motor (float64 at offset 36, Amperes)
                      positive = traction, negative = braking/regen
    angular_velocity: /sensors/imu   VescImuStamped yaw_rate              (float64 at offset 84, deg/s)
                      → × pi/180  [rad/s]
    All merged to sensors/core timestamps.

    VescState field offsets (confirmed empirically, speed@76 anchor):
      [20] voltage_input  [28] temperature_pcb  [36] current_motor
      [44] current_input  [52] id  [60] iq  [68] duty_cycle  [76] speed
    """
    # speed + current_motor from /sensors/core
    tid_core = conn.execute("SELECT id FROM topics WHERE name='/sensors/core'").fetchone()[0]
    rows_core = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid_core,)).fetchall()
    ts_core, speed_erpm, current_motor = [], [], []
    for t, data in rows_core:
        b = bytes(data)
        if len(b) >= 84:
            ts_core.append(t * 1e-9)
            speed_erpm.append(read_float64(b, 76))
            current_motor.append(read_float64(b, 36))

    # angular velocity from /sensors/imu (VescImuStamped)
    # offset 36 = ypr.z (yaw ANGLE in degrees) — WRONG, do not use
    # offset 84 = angular_velocity.z — unit is DEG/S (confirmed: range ±135°/s matches MoCap)
    tid_imu = conn.execute("SELECT id FROM topics WHERE name='/sensors/imu'").fetchone()[0]
    rows_imu = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid_imu,)).fetchall()
    ts_imu, omega_degs = [], []
    for t, data in rows_imu:
        b = bytes(data)
        if len(b) >= 92:
            ts_imu.append(t * 1e-9)
            omega_degs.append(read_float64(b, 84))  # angular_velocity.z in deg/s

    ts_core       = np.array(ts_core)
    ts_imu        = np.array(ts_imu)
    speed_ms      = np.array(speed_erpm) * ERPM_TO_MS
    current_motor = np.array(current_motor)
    omega_rad     = np.array(omega_degs) * np.pi / 180.0  # deg/s → rad/s

    omega_at_core = np.interp(ts_core, ts_imu, omega_rad)

    return pd.DataFrame({
        'timestamp':     ts_core,
        'speed':         speed_ms,
        'current_motor': current_motor,   # [A] positive=traction, negative=regen/brake
        'angular_velocity': omega_at_core,
    })


def extract_imu(conn):
    """
    /livox/imu  (sensor_msgs/Imu)
    CDR layout (frame_id="livox_imu", len=10, pad=2):
      Header = 4+4+4+4+10+2 = 28 bytes
      orientation xyzw (4×8=32): 28-59
      orientation_covariance (9×8=72): 60-131
      angular_velocity xyz (3×8=24): 132-155
      angular_velocity_covariance (9×8=72): 156-227
      linear_acceleration xyz (3×8=24): 228-251
      linear_acceleration_covariance (9×8=72): 252-323
    """
    tid = conn.execute("SELECT id FROM topics WHERE name='/livox/imu'").fetchone()[0]
    rows = conn.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)).fetchall()

    ts, ax, ay, az, wx, wy, wz = [], [], [], [], [], [], []
    for t, data in rows:
        b = bytes(data)
        if len(b) >= 252:
            ts.append(t * 1e-9)
            wx.append(read_float64(b, 132))
            wy.append(read_float64(b, 140))
            wz.append(read_float64(b, 148))
            ax.append(read_float64(b, 228))
            ay.append(read_float64(b, 236))
            az.append(read_float64(b, 244))

    return pd.DataFrame({
        'timestamp': ts,
        'IMU_linear_acceleration_x': ax,
        'IMU_linear_acceleration_y': ay,
        'IMU_linear_acceleration_z': az,
        'IMU_angular_velocity_x': wx,
        'IMU_angular_velocity_y': wy,
        'IMU_angular_velocity_z': wz,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',  required=True, help='Path to .db3 bag file')
    parser.add_argument('--name', default='session_20260721', help='Dataset name (used as folder and filename prefix)')
    args = parser.parse_args()

    conn = sqlite3.connect(args.bag)

    print("Extracting pose ...")
    df_pose = extract_pose(conn)
    print(f"  {len(df_pose)} rows, t=[{df_pose.timestamp.min():.1f}, {df_pose.timestamp.max():.1f}]")

    print("Extracting controls ...")
    df_ctrl = extract_controls(conn)
    print(f"  {len(df_ctrl)} rows, speed_cmd=[{df_ctrl.speed_cmd.min():.2f},{df_ctrl.speed_cmd.max():.2f}]m/s"
          f"  steering=[{df_ctrl.steering_cmd.min():.3f},{df_ctrl.steering_cmd.max():.3f}]rad")

    print("Extracting odom ...")
    df_odom = extract_odom(conn)
    print(f"  {len(df_odom)} rows, speed=[{df_odom.speed.min():.2f},{df_odom.speed.max():.2f}]m/s"
          f"  omega=[{df_odom.angular_velocity.min():.3f},{df_odom.angular_velocity.max():.3f}]rad/s")

    print("Extracting IMU ...")
    df_imu = extract_imu(conn)
    print(f"  {len(df_imu)} rows @ ~{len(df_imu)/(df_imu.timestamp.max()-df_imu.timestamp.min()):.0f}Hz")

    conn.close()

    out_dir = os.path.join('data', args.name)
    os.makedirs(out_dir, exist_ok=True)

    df_pose.to_csv(os.path.join(out_dir, f'{args.name}_pose.csv'), index=False)
    df_ctrl.to_csv(os.path.join(out_dir, f'{args.name}_controls.csv'), index=False)
    df_odom.to_csv(os.path.join(out_dir, f'{args.name}_odom.csv'), index=False)
    df_imu.to_csv(os.path.join(out_dir, f'{args.name}_imu.csv'), index=False)

    print(f"\nSaved 4 CSVs to {out_dir}/")
    print("Next: python3 DataPreparation.py  (edit source_dir inside to point to your data folder)")


if __name__ == '__main__':
    main()
