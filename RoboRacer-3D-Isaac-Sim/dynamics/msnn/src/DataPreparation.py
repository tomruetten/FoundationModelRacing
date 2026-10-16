import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, savgol_filter
from scipy.spatial.transform import Rotation as R
import numpy as np

def butter_lowpass_filter(data, cutoff_freq, sampling_freq, order=4):
    """
    Apply Butterworth low-pass filter to data using SOS (Second-Order Sections)
    
    Args:
        data: Input signal
        cutoff_freq: Cutoff frequency in Hz
        sampling_freq: Sampling frequency in Hz
        order: Filter order (higher = steeper rolloff)
    """
    nyquist = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist
    sos = butter(order, normal_cutoff, output='sos')

    return sosfiltfilt(sos, data)


def quaternion_to_euler_scipy(q_x, q_y, q_z, q_w):
    """
    Convert quaternions to Euler angles using SciPy
    Returns roll, pitch, yaw in radians
    """
    # SciPy expects quaternions as [x, y, z, w]
    quaternions = np.column_stack([q_x, q_y, q_z, q_w])
    
    # Create rotation object and convert to Euler angles
    rotations = R.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=False)  # Returns [roll, pitch, yaw]
    
    return euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]  # roll, pitch, yaw


def calculate_frequency(df, time_col='timestamp'):
        if time_col not in df.columns:
            raise ValueError(f"'{time_col}' column not found in dataframe.")
        time_diffs = df[time_col].diff().dropna()
        avg_period = time_diffs.mean()
        if avg_period == 0:
            return float('inf')
        return 1.0 / avg_period

    
def process_data(source_dir, target_dir = "processed_data", target_period=20, plotting=False):
    """
    Process the raw data from the given source directory according and saves a single CSV file with the processed data in the target directory.
    Plotting can be enabled to visualize comparisons between the original and processed data.
    
    Args:
        source_dir (str): Directory containing the raw CSV files.
        target_dir (str): Directory where the processed file will be saved.
        target_period (int): The target period for the data in ms.
        plotting (bool): Whether to generate plots for verification.
    """
    
    MoCap_df = None
    odom_df = None
    controls_df = None
    IMU_df = None
    
    for file in os.listdir(source_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(source_dir, file)
            if 'pose' in file:
                MoCap_df = pd.read_csv(file_path)
            elif 'odom' in file:
                odom_df = pd.read_csv(file_path)
            elif 'controls' in file:
                controls_df = pd.read_csv(file_path)
            elif 'imu' in file:
                IMU_df = pd.read_csv(file_path)
            else:
                print (f"Unknown file type: {file}. Skipping.")
            
    if MoCap_df is None:
        raise ValueError("Pose data not found in the source directory.")
    if odom_df is None:
        raise ValueError("Odom data not found in the source directory.")
    if controls_df is None:
        raise ValueError("Controls data not found in the source directory.") 
    if IMU_df is None:
        raise ValueError("IMU data not found in the source directory.")
    
    if MoCap_df.isna().any().any():
        MoCap_df.dropna(inplace=True)
        print("WARNING!!!Pose data contains NaN values. Please check the data integrity. Incomplete rows are dropped.")
    if IMU_df.isna().any().any():
        IMU_df.dropna(inplace=True)
        print("WARNING!!!IMU data contains NaN values. Please check the data integrity. Incomplete rows are dropped.")
    if controls_df.isna().any().any():
        controls_df.dropna(inplace=True)
        print("WARNING!!!Controls data contains NaN values. Please check the data integrity. Incomplete rows are dropped.")
    if odom_df.isna().any().any():
        odom_df.dropna(inplace=True)
        print("WARNING!!!Odom data contains NaN values. Please check the data integrity. Incomplete rows are dropped.")


    # Calculate the frequency of each dataframe
    mocap_freq = calculate_frequency(MoCap_df)
    imu_freq = calculate_frequency(IMU_df)
    odom_freq = calculate_frequency(odom_df)
    controls_freq = calculate_frequency(controls_df)
    
    print(f"Data from {source_dir}. Target frequency: {1000/target_period}. Target period: {target_period}ms")
    print(f"Pose frequency: {mocap_freq:.2f}Hz. \t Controls frequency: {controls_freq:.2f}Hz \t Odom frequency: {odom_freq:.2f}Hz \
        \t IMU frequency: {imu_freq:.2f}Hz")
    if mocap_freq < (1000 / target_period):
        print(f"WARNING!!!Pose frequency {mocap_freq:.2f}Hz is lower than target frequency {1000 / target_period}Hz. Upsampling might lead to inaccurate results.")
    
    MoCap_df['timestamp'] = pd.to_datetime(MoCap_df['timestamp'], unit='s')
    odom_df['timestamp'] = pd.to_datetime(odom_df['timestamp'], unit='s')
    controls_df['timestamp'] = pd.to_datetime(controls_df['timestamp'], unit='s')
    IMU_df['timestamp'] = pd.to_datetime(IMU_df['timestamp'], unit='s')
    MoCap_df.set_index('timestamp', inplace=True)
    MoCap_df = MoCap_df.apply(pd.to_numeric, errors='raise')
    odom_df.set_index('timestamp', inplace=True)
    odom_df = odom_df.apply(pd.to_numeric, errors='raise')
    odom_df.rename(columns={'speed': 'odom_speed', 'angular_velocity': 'odom_angular_velocity'}, inplace=True)
    controls_df.set_index('timestamp', inplace=True)
    controls_df = controls_df.apply(pd.to_numeric, errors='raise')
    IMU_df.set_index('timestamp', inplace=True)
    IMU_df = IMU_df.apply(pd.to_numeric, errors='raise')

    # Compute mean and std of the timedeltas in milliseconds
    dt_IMU = IMU_df.index.to_series().diff().dt.total_seconds() * 1000  # ms
    dt_MoCap = MoCap_df.index.to_series().diff().dt.total_seconds() * 1000  # ms
    dt_odom = odom_df.index.to_series().diff().dt.total_seconds() * 1000  # ms
    dt_controls = controls_df.index.to_series().diff().dt.total_seconds() * 1000  # ms
    print(f"Mean dt IMU: {dt_IMU.mean():.3f} ms")
    print(f"Std  dt IMU: {dt_IMU.std():.3f} ms")
    print(f"Mean dt MoCap: {dt_MoCap.mean():.3f} ms")
    print(f"Std  dt MoCap: {dt_MoCap.std():.3f} ms")
    print(f"Mean dt Odom: {dt_odom.mean():.3f} ms")
    print(f"Std  dt Odom: {dt_odom.std():.3f} ms")
    print(f"Mean dt Controls: {dt_controls.mean():.3f} ms")
    print(f"Std  dt Controls: {dt_controls.std():.3f} ms")

    ##########################################################
    # Remove offsets from IMU data
    ##########################################################
    first_speed_cmd_index = controls_df[controls_df['speed_cmd'] != 0].index.min()
    if len(IMU_df[IMU_df.index <= first_speed_cmd_index]) < 100:
        print(f"WARNING!!!Few samples before first speed command, offset correction for IMU might not be accurate.")
    IMU_df['IMU_linear_acceleration_x'] -= IMU_df.loc[:first_speed_cmd_index, 'IMU_linear_acceleration_x'].mean()
    IMU_df['IMU_linear_acceleration_y'] -= IMU_df.loc[:first_speed_cmd_index, 'IMU_linear_acceleration_y'].mean()
    IMU_df['IMU_angular_velocity_z'] -= IMU_df.loc[:first_speed_cmd_index, 'IMU_angular_velocity_z'].mean()
    
    ##########################################################
    # Convert quaternion to Euler angles using SciPy
    ##########################################################
    roll, pitch, yaw = quaternion_to_euler_scipy(
        MoCap_df['q_x'].to_numpy(), 
        MoCap_df['q_y'].to_numpy(), 
        MoCap_df['q_z'].to_numpy(), 
        MoCap_df['q_w'].to_numpy()
    )

    MoCap_df['yaw'] = np.unwrap(yaw)
    MoCap_df['roll'] = np.unwrap(roll)
    MoCap_df['pitch'] = np.unwrap(pitch)

    ##########################################################
    # Resampling to intermediate frequency where necessary and filtering
    ##########################################################
    intermediate_period = '10ms'
    
    # Resample  MoCap to 100Hz for evenly spaced data (high deviations in original data)
    MoCap_df_intermediate = MoCap_df.copy()
    intermediate_index = pd.date_range(start=MoCap_df.index.min(), end=MoCap_df.index.max(), freq=intermediate_period)
    combined_index=MoCap_df_intermediate.index.union(intermediate_index).drop_duplicates()
    MoCap_df_intermediate = MoCap_df_intermediate.reindex(combined_index).interpolate('index') 
    MoCap_df_intermediate = MoCap_df_intermediate.loc[intermediate_index]    
    
    # imu and odom data not resampled before filtering, since they are pretty evenly spaced already

    # Anti-aliasing filter - cutoff should be below Nyquist of target frequency
    target_sampling_freq = 1000 / target_period  # 50Hz for 20ms
    anti_alias_cutoff_IMU = target_sampling_freq / 6  # Conservative: ~16Hz for 50Hz target
        
    # Apply anti-aliasing filter at original high frequency
    IMU_df_filtered = IMU_df[['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y', 'IMU_angular_velocity_z']].copy()
    IMU_df_filtered['IMU_linear_acceleration_x'] = butter_lowpass_filter(IMU_df['IMU_linear_acceleration_x'].to_numpy(), anti_alias_cutoff_IMU, imu_freq)
    IMU_df_filtered['IMU_linear_acceleration_y'] = butter_lowpass_filter(IMU_df['IMU_linear_acceleration_y'].to_numpy(), anti_alias_cutoff_IMU, imu_freq)
    IMU_df_filtered['IMU_angular_velocity_z'] = butter_lowpass_filter(IMU_df['IMU_angular_velocity_z'].to_numpy(), anti_alias_cutoff_IMU, imu_freq)
    print(f"Applied anti-aliasing filter for IMU at {anti_alias_cutoff_IMU:.1f}Hz before resampling")

    # Transform IMU linear accelerations from g to m/s²
    # Physical push tests (2026-07-30) confirmed: Livox MID-360 axes already match
    # vehicle frame (x=forward, y=left) — no 90° swap needed.
    IMU_df_filtered[['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y']] *= 9.81

    
    # Appling anti-aliasing filter to odom data
    anti_alias_cutoff_odom = target_sampling_freq / 4.5
    odom_df_filtered = odom_df.copy()
    odom_df_filtered['odom_speed'] = butter_lowpass_filter(odom_df['odom_speed'].to_numpy(), anti_alias_cutoff_odom, odom_freq)
    odom_df_filtered['odom_angular_velocity'] = butter_lowpass_filter(odom_df['odom_angular_velocity'].to_numpy(), anti_alias_cutoff_odom, odom_freq)
    print(f"Applied anti-aliasing filter for Odom at {anti_alias_cutoff_odom:.1f}Hz before resampling")

    ####################################################################
    # Calculate speeds and accelerations for MoCap with Savgol
     ####################################################################
    # Parameters for Savitzky-Golay filtering
    savgol_window = 15  # Must be odd, adjust based on data quality
    savgol_order = 3    # Polynomial order (3 works well for smooth motion)
    dt_intermediate = 0.01  # 10ms sampling period for intermediate data

    MoCap_df_intermediate['p_x_smooth'] = savgol_filter(MoCap_df_intermediate['p_x'], window_length=11, polyorder=1)
    MoCap_df_intermediate['p_y_smooth'] = savgol_filter(MoCap_df_intermediate['p_y'], window_length=11, polyorder=1)
    MoCap_df_intermediate['yaw_smooth'] = savgol_filter(MoCap_df_intermediate['yaw'], window_length=11, polyorder=1)

    print(f"Applying Savitzky-Golay filtering and derivative calculation (window={savgol_window}, order={savgol_order})")

    # Step 1: Calculate GLOBAL velocities directly using Savitzky-Golay derivatives
    MoCap_df_intermediate['v_x_global'] = savgol_filter(MoCap_df_intermediate['p_x_smooth'].to_numpy(), 
        window_length=savgol_window, polyorder=savgol_order, deriv=1, delta=dt_intermediate)

    MoCap_df_intermediate['v_y_global'] = savgol_filter(MoCap_df_intermediate['p_y_smooth'].to_numpy(), 
        window_length=savgol_window, polyorder=savgol_order, deriv=1, delta=dt_intermediate)
    
    MoCap_df_intermediate['v_global'] = np.sqrt(
        MoCap_df_intermediate['v_x_global'].to_numpy()**2 + MoCap_df_intermediate['v_y_global'].to_numpy()**2)


    # Step 2: Transform global velocities to vehicle local frame
    cos_yaw = np.cos(MoCap_df_intermediate['yaw_smooth'].to_numpy())
    sin_yaw = np.sin(MoCap_df_intermediate['yaw_smooth'].to_numpy())

    # Transform global velocities to local vehicle frame
    MoCap_df_intermediate['v_x_local'] = (
        cos_yaw * MoCap_df_intermediate['v_x_global'].to_numpy() + 
        sin_yaw * MoCap_df_intermediate['v_y_global'].to_numpy())

    MoCap_df_intermediate['v_y_local'] = (
        -sin_yaw * MoCap_df_intermediate['v_x_global'].to_numpy() + 
        cos_yaw * MoCap_df_intermediate['v_y_global'].to_numpy())

    # Step 3: Calculate accelerations in local frame using Savgol
    MoCap_df_intermediate['a_x_global'] = savgol_filter(MoCap_df_intermediate['p_x_smooth'].to_numpy(), window_length=savgol_window, polyorder=savgol_order, deriv=2, delta=dt_intermediate)
    MoCap_df_intermediate['a_y_global'] = savgol_filter(MoCap_df_intermediate['p_y_smooth'].to_numpy(), window_length=savgol_window, polyorder=savgol_order, deriv=2, delta=dt_intermediate)

    MoCap_df_intermediate['a_x_local'] = (
    cos_yaw * MoCap_df_intermediate['a_x_global'].to_numpy() + 
    sin_yaw * MoCap_df_intermediate['a_y_global'].to_numpy())
    MoCap_df_intermediate['a_y_local'] = (
    -sin_yaw * MoCap_df_intermediate['a_x_global'].to_numpy() + 
    cos_yaw * MoCap_df_intermediate['a_y_global'].to_numpy())

    # Step 4: Calculate angular velocity
    MoCap_df_intermediate['mocap_angular_velocity'] = savgol_filter(
        MoCap_df_intermediate['yaw_smooth'].to_numpy(), window_length=savgol_window, polyorder=savgol_order, deriv=1, delta=dt_intermediate)
    #print(MoCap_df_intermediate.head(15))

    # Step 5: Correct for 17.8° base_link→mocap_frame mounting offset.
    # Physical push tests (3 runs, 2026-07-30) showed the Chassis2 rigid body
    # is mounted +17.8° (CCW, std=1.9°) relative to the vehicle's true heading.
    # MoCap-derived local-frame velocities and accelerations are therefore in the
    # mocap_frame, not base_link. Rotate by +17.8° to recover true body-frame
    # quantities.  ψ̇ (mocap_angular_velocity) is d(yaw)/dt — a constant offset
    # cancels out, so no correction is needed there.
    MOCAP_OFFSET_RAD = np.radians(17.8)
    _cos_off = np.cos(MOCAP_OFFSET_RAD)
    _sin_off = np.sin(MOCAP_OFFSET_RAD)
    _vx = MoCap_df_intermediate['v_x_local'].to_numpy()
    _vy = MoCap_df_intermediate['v_y_local'].to_numpy()
    MoCap_df_intermediate['v_x_local'] = _cos_off * _vx + _sin_off * _vy
    MoCap_df_intermediate['v_y_local'] = -_sin_off * _vx + _cos_off * _vy
    _ax = MoCap_df_intermediate['a_x_local'].to_numpy()
    _ay = MoCap_df_intermediate['a_y_local'].to_numpy()
    MoCap_df_intermediate['a_x_local'] = _cos_off * _ax + _sin_off * _ay
    MoCap_df_intermediate['a_y_local'] = -_sin_off * _ax + _cos_off * _ay
    print(f"Applied {np.degrees(MOCAP_OFFSET_RAD):.1f}° base_link→mocap_frame rotation correction to v_x/v_y/a_x/a_y")

    # ########################################################
    # # Resample the data to the target frequency and align it
    # ########################################################
    common_start = max(MoCap_df_intermediate.index.min(), odom_df.index.min(), controls_df.index.min(), IMU_df_filtered.index.min())
    common_end = min(MoCap_df_intermediate.index.max(), odom_df.index.max(), controls_df.index.max(), IMU_df_filtered.index.max())
    target_index = pd.date_range(start=common_start,  end=common_end, freq=f'{int(target_period)}ms')
    
    
    MoCap_df_resampled = MoCap_df_intermediate.resample(f'{int(target_period)}ms', origin=common_start).mean()
    #print(MoCap_df_resampled.head(10))

    combined_index = controls_df.index.union(target_index).drop_duplicates()
    controls_df_resampled = controls_df.reindex(combined_index).interpolate(method='index')
    controls_df_resampled = controls_df_resampled.loc[target_index]

    combined_index = odom_df_filtered.index.union(target_index).drop_duplicates()
    odom_df_resampled = odom_df_filtered.reindex(combined_index).interpolate(method='index')
    odom_df_resampled = odom_df_resampled.loc[target_index]

    combined_index = IMU_df_filtered.index.union(target_index).drop_duplicates()
    IMU_df_resampled = IMU_df_filtered.reindex(combined_index).interpolate(method='index')
    IMU_df_resampled = IMU_df_resampled.loc[target_index]
    
    # Reindex downsampled data to match target_index exactly
    MoCap_df_resampled = MoCap_df_resampled.reindex(target_index, method='nearest', tolerance=pd.Timedelta('5ms'))
    
    ##################################################################
    # Smooth the velocities and accelerations using Butterworth filter
    ##################################################################
    anti_alias_cutoff_mocap_v = 1.15
    filter_order_mocap_v = 3
    anti_alias_cutoff_mocap_a = 1.15
    filter_order_mocap_a = 3
    anti_alias_cutoff_mocap_angular_velocity = 12
    filter_order_mocap_angular_velocity = 4
    # Apply anti-aliasing filter to MoCap velocities
    MoCap_df_resampled['v_x_local'] = butter_lowpass_filter(MoCap_df_resampled['v_x_local'].to_numpy(), anti_alias_cutoff_mocap_v, target_sampling_freq, filter_order_mocap_v)
    MoCap_df_resampled['v_y_local'] = butter_lowpass_filter(MoCap_df_resampled['v_y_local'].to_numpy(), anti_alias_cutoff_mocap_v, target_sampling_freq, filter_order_mocap_v)
    MoCap_df_resampled['v_x_global'] = butter_lowpass_filter(MoCap_df_resampled['v_x_global'].to_numpy(), anti_alias_cutoff_mocap_v, target_sampling_freq, filter_order_mocap_v)
    MoCap_df_resampled['v_y_global'] = butter_lowpass_filter(MoCap_df_resampled['v_y_global'].to_numpy(), anti_alias_cutoff_mocap_v, target_sampling_freq, filter_order_mocap_v)
    MoCap_df_resampled['a_x_local'] = butter_lowpass_filter(MoCap_df_resampled['a_x_local'].to_numpy(), anti_alias_cutoff_mocap_a, target_sampling_freq, filter_order_mocap_a)
    MoCap_df_resampled['a_y_local'] = butter_lowpass_filter(MoCap_df_resampled['a_y_local'].to_numpy(), anti_alias_cutoff_mocap_a, target_sampling_freq, filter_order_mocap_a)
    MoCap_df_resampled['a_x_global'] = butter_lowpass_filter(MoCap_df_resampled['a_x_global'].to_numpy(), anti_alias_cutoff_mocap_a, target_sampling_freq, filter_order_mocap_a)
    MoCap_df_resampled['a_y_global'] = butter_lowpass_filter(MoCap_df_resampled['a_y_global'].to_numpy(), anti_alias_cutoff_mocap_a, target_sampling_freq, filter_order_mocap_a)
    MoCap_df_resampled['mocap_angular_velocity'] = butter_lowpass_filter(
        MoCap_df_resampled['mocap_angular_velocity'].to_numpy(), 
        anti_alias_cutoff_mocap_angular_velocity, target_sampling_freq, filter_order_mocap_angular_velocity)
    MoCap_df_resampled['v_global'] = np.sqrt(
        MoCap_df_resampled['v_x_local'].to_numpy()**2 + 
        MoCap_df_resampled['v_y_local'].to_numpy()**2)
    print(f"Applied anti-aliasing filter for MoCap velocities at {anti_alias_cutoff_mocap_v:.3f}Hz")
    print(f"Applied anti-aliasing filter for MoCap accelerations at {anti_alias_cutoff_mocap_a:.3f}Hz")
    print(f"Applied anti-aliasing filter for MoCap angular velocities at {anti_alias_cutoff_mocap_angular_velocity:.3f}Hz")

    # Join all dataframes on the common index
    df_resampled = MoCap_df_resampled.join([odom_df_resampled, controls_df_resampled, IMU_df_resampled], how='left')
    # check for NaN values after join
    if df_resampled.isna().any().any():
        print("Warning: NaN values found in the resampled data. This may indicate misalignment or missing data.")
        print(df_resampled.isna().sum())
    
    print(df_resampled)
    
    # Save IMU filtering comparison data
    if plotting:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # Plot IMU linear acceleration x
        axs[0].plot(IMU_df.index, IMU_df['IMU_linear_acceleration_x'], label='Raw a_x', alpha=0.5)
        axs[0].plot(df_resampled.index, df_resampled['IMU_linear_acceleration_x'], label='Resampled a_x', linewidth=2)
        axs[0].set_ylabel('a_x [g]')
        axs[0].legend()
        axs[0].set_title('IMU Linear Acceleration X')

        # Plot IMU linear acceleration y
        axs[1].plot(IMU_df.index, IMU_df['IMU_linear_acceleration_y'], label='Raw a_y', alpha=0.5)
        axs[1].plot(df_resampled.index, df_resampled['IMU_linear_acceleration_y'], label='Resampled a_y', linewidth=2)
        axs[1].set_ylabel('a_y [g]')
        axs[1].legend()
        axs[1].set_title('IMU Linear Acceleration Y')

        # Plot IMU angular velocity z
        axs[2].plot(IMU_df.index, IMU_df['IMU_angular_velocity_z'], label='Raw w_z', alpha=0.5)
        axs[2].plot(df_resampled.index, df_resampled['IMU_angular_velocity_z'], label='Resampled w_z', linewidth=2)
        axs[2].set_ylabel('Angular Velocity [degree/s]')
        axs[2].legend()
        axs[2].set_title('IMU Angular Velocity Z')
        axs[2].set_xlabel('Timestamp')

        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        # Plot odom speed
        axs[0].plot(odom_df.index, odom_df['odom_speed'], label='Raw odom speed', alpha=0.5)
        axs[0].plot(df_resampled.index, df_resampled['odom_speed'], label='Resampled odom speed', linewidth=2)
        axs[0].set_ylabel('Speed [m/s]')
        axs[0].legend()
        axs[0].set_title('Odom Speed')

        # Plot odom angular velocity
        axs[1].plot(odom_df.index, odom_df['odom_angular_velocity'], label='Raw odom angular velocity', alpha=0.5)
        axs[1].plot(df_resampled.index, df_resampled['odom_angular_velocity'], label='Resampled odom angular velocity', linewidth=2)
        axs[1].set_ylabel('Angular Velocity [rad/s]')
        axs[1].legend()
        axs[1].set_title('Odom Angular Velocity')
        axs[1].set_xlabel('Timestamp')

        plt.tight_layout()
        plt.show()
        
        # First figure: 3x2 grid for MoCap data
        fig1, axs1 = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

        # Row 1: v_x and v_y (local)
        axs1[0, 0].plot(df_resampled.index, df_resampled['v_x_local'], label='MoCap v_x_local')
        axs1[0, 0].set_ylabel('Velocity [m/s]')
        axs1[0, 0].legend()
        axs1[0, 0].set_title('MoCap v_x_local')

        axs1[0, 1].plot(df_resampled.index, df_resampled['v_y_local'], label='MoCap v_y_local')
        axs1[0, 1].set_ylabel('Velocity [m/s]')
        axs1[0, 1].legend()
        axs1[0, 1].set_title('MoCap v_y_local')

        # Row 2: a_x and a_y (local)
        axs1[1, 0].plot(df_resampled.index, df_resampled['a_x_local'], label='MoCap a_x_local')
        axs1[1, 0].set_ylabel('Acceleration [m/s²]')
        axs1[1, 0].legend()
        axs1[1, 0].set_title('MoCap a_x_local')

        axs1[1, 1].plot(df_resampled.index, df_resampled['a_y_local'], label='MoCap a_y_local')
        axs1[1, 1].set_ylabel('Acceleration [m/s²]')
        axs1[1, 1].legend()
        axs1[1, 1].set_title('MoCap a_y_local')
        # Row 3: p_x and p_y (global)
        axs1[2, 0].plot(df_resampled.index, df_resampled['p_x'], label='MoCap p_x')
        axs1[2, 0].plot(df_resampled.index, df_resampled['p_x_smooth'], label='MoCap p_x_smooth', alpha=0.7)
        axs1[2, 0].set_ylabel('Position [m]')
        axs1[2, 0].legend()
        axs1[2, 0].set_title('MoCap p_x')
        axs1[2, 1].plot(df_resampled.index, df_resampled['p_y'], label='MoCap p_y')
        axs1[2, 1].plot(df_resampled.index, df_resampled['p_y_smooth'], label='MoCap p_y_smooth', alpha=0.7)
        axs1[2, 1].set_ylabel('Position [m]')
        axs1[2, 1].legend()
        axs1[2, 1].set_title('MoCap p_y')
        # Row 3: angular velocity and global speed
        axs1[3, 0].plot(df_resampled.index, df_resampled['mocap_angular_velocity'], label='MoCap Angular Velocity')
        axs1[3, 0].set_ylabel('Angular Velocity [rad/s]')
        axs1[3, 0].legend()
        axs1[3, 0].set_title('MoCap Angular Velocity')
        axs1[3, 0].set_xlabel('Timestamp')
        
        axs1[3, 1].plot(df_resampled.index, df_resampled['yaw'], label='MoCap yaw')
        axs1[3, 1].plot(df_resampled.index, df_resampled['yaw_smooth'], label='MoCap yaw_smooth', alpha=0.7)
        axs1[3, 1].set_ylabel('Yaw [rad]')
        axs1[3, 1].legend()
        axs1[3, 1].set_title('MoCap Yaw')
        axs1[3, 1].set_xlabel('Timestamp')

        # axs1[3, 1].plot(df_resampled.index, df_resampled['v_global'], label='MoCap v_global')
        # axs1[3, 1].set_ylabel('Speed [m/s]')
        # axs1[3, 1].legend()
        # axs1[3, 1].set_title('MoCap Global Speed')
        # axs1[3, 1].set_xlabel('Timestamp')

        plt.tight_layout()
        plt.show()

        # Second figure: Comparison plots
        fig2, axs2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # Plot 1: odom_speed vs v_x_local
        axs2[0].plot(df_resampled.index, df_resampled['odom_speed'], label='Odom', color="#7F7F7F")
        axs2[0].plot(df_resampled.index, df_resampled['v_x_local'], label='MoCap', color="#005293")
        #axs2[0].plot(df_resampled.index, df_resampled['v_global'], label='MoCap v_global', alpha=0.6)
        axs2[0].set_ylabel(r"$v_x$  in m/s")
        axs2[0].legend()
        #axs2[0].set_title('Longitudinal Speed Comparison: Odom vs MoCap')

        # Plot 2: odom angular velocity vs mocap angular velocity
        axs2[1].plot(df_resampled.index, df_resampled['odom_angular_velocity'], label='Odom', color="#7F7F7F")
        axs2[1].plot(df_resampled.index, df_resampled['mocap_angular_velocity'], label='MoCap', color="#005293")
        axs2[1].set_ylabel(r"$\dot{\psi}$  in rad/s")
        axs2[1].legend()
        #axs2[1].set_title('Angular Velocity Comparison: Odom vs MoCap')
        axs2[1].set_xlabel('Timestamp')

        plt.tight_layout()
        plt.show()

        
    # Save the processed data to a CSV file
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, source_dir.split("/")[-1] + '_' + str(target_period) + 'ms.csv')
    df_resampled.to_csv(target_path, index=True)
    print(f"Processed data saved to {target_path}")
    

def main():
    ## Uncomment to process all subdirectories in data folder
    # for subdir in os.listdir("data"):
    #     full_path = os.path.join("data", subdir)
    #     if os.path.isdir(full_path):
    #         process_data(full_path, plotting=False)
      
    process_data("data/stanley_0.6", plotting=True)

if __name__ == "__main__":
    main()