# Camera-LiDAR TF usage notes

## Current status

The current camera-LiDAR transform is a visually validated project candidate, not a final high-precision calibration.
It was calibrated with an AprilGrid board and refined by visual LiDAR-to-RGB projection.

## Files to keep

Recommended project folder:

```bash
<your OKF1tenth project root>/config/calibration/
```

Keep these files there:

```text
extrinsics_candidate_checked_v2.yaml
camera_intrinsics_1280x800_checked.yaml
static_tf_command_candidate.sh
T_camera_lidar_candidate.txt
T_lidar_camera_candidate.txt
README_sensor_tf_usage.md
```

## ROS usage with layout_00_slow_run_01

The bag already contains:

```text
base_link -> livox_frame
translation: [0.270, 0.000, 0.110]
rotation: identity
```

So when playing the bag, only publish the additional camera TF:

```bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.02266665 --y 0.06987346 --z -0.04441125 \
  --qx -0.46552296 --qy 0.48718884 --qz -0.52270603 --qw 0.52220090 \
  --frame-id livox_frame \
  --child-frame-id oak_rgb_camera_optical_frame
```

Do not publish another base_link -> livox_frame transform while playing this bag.
Do not publish both livox_frame -> camera and base_link -> camera as TF at the same time.

## Check TF

```bash
ros2 run tf2_ros tf2_echo livox_frame oak_rgb_camera_optical_frame
ros2 run tf2_ros tf2_echo base_link oak_rgb_camera_optical_frame
```

## Generate TF graph

```bash
ros2 run tf2_tools view_frames
ls -lh frames*.pdf *.gv
xdg-open $(ls -t frames*.pdf | head -1)
```

## Offline projection convention

Use `T_camera_lidar_candidate.txt` for projecting LiDAR points into the RGB camera image:

```text
X_camera = T_camera_lidar * X_lidar
```

Use `T_lidar_camera_candidate.txt` or the YAML `livox_frame_to_oak_rgb_camera_optical_frame` for ROS static TF.

## Camera data

For this dataset, OAK-D RGB was not recorded as a ROS image topic in the bag. The camera frame is therefore manually defined for the offline RGB video as:

```text
oak_rgb_camera_optical_frame
```

Use `camera_intrinsics_1280x800_checked.yaml` for 1280x800 RGB videos.
