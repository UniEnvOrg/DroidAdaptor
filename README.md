# Droid Environment / Data Adaptor for UniEnv

This repository contains the implementation of the adaptor that connects the Droid environment and dataset to the UniEnv interface.

To install the adaptor, you can simply install this package from PyPI:

```bash
pip install unienv-droid[dataset,env]
```

You can specify the features that you want to install through the optional dependencies. For example, if you only want to install the dataset related code, you can do:

```bash
pip install unienv-droid[dataset]
```

## Usage of Droid Environment Wrapper

```python
from unienv_droid import DroidEnv
from droid.robot_env import RobotEnv as DroidRobotEnv

# Create the underlying Droid RobotEnv
robotenv = DroidRobotEnv()

# Create the DroidEnv wrapper
env = DroidEnv(robotenv)

# Now you can use the env as a standard UniEnv environment
ctx, obs, info = env.reset()
while True:
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

The observation is nested like the robot (`unienv_droid.conventions`):

```text
obs["robot"]["arm"]       joint_position, joint_velocity, joint_torque, joint_motor_torque,
                          eef_position, eef_quaternion,
                          prev_joint_torques_computed(_safened), prev_controller_latency_ms,
                          prev_command_successful
obs["robot"]["gripper"]   position (1,)
obs["hand_cam"], obs["external_cam_0"], obs["external_cam_1"], ...
                          rgb, depth, intrinsics, extrinsics
```

- `eef_position` / `eef_quaternion` are the flange pose (the fingertip with
  `use_tip_as_eef=True`) in the robot base frame; the quaternion is
  `(w, x, y, z)` with `w >= 0`.
- `joint_torque` is the measured joint torque (upstream
  `motor_torques_measured`) and `joint_motor_torque` the torque the
  controller commands (upstream `joint_torques_computed`).
- Cameras are named by role: the wrist camera is `hand_cam`, the exterior
  cameras `external_cam_0`, `external_cam_1`, ... in sorted-serial order;
  `camera_names={serial: name}` overrides the names. Each has `rgb` (`uint8`,
  RGB), `depth` (`float32`, metres), `intrinsics` (OpenCV `3x3`) and
  `extrinsics` (base -> camera, `3x4`). For stereo ZED cameras this is the left
  stream; `include_right_images=True` adds `<name>_right`.
- `env.camera_names` lists the camera streams and `env.camera_serials` maps
  each to its serial.

The action is the flat upstream vector `[x, y, z, rx, ry, rz, gripper]`
(`[x, y, z, r00 ... r12, gripper]` with `use_6d_rotation_in_action=True`).

### Space metadata

Every observation and action leaf carries unienv space metadata
(`space.metadata`, unienv >= 0.0.1b16) describing what it holds; the
vocabulary is unienv's (`kind`, `units`, `frame`, `rotation`, `names`, ...):

| Leaf | Metadata |
|---|---|
| `robot.arm.joint_*` | `proprio`, `rad` / `rad/s` / `N*m`, names `fr3_joint1..7` |
| `robot.arm.eef_position` / `eef_quaternion` | `proprio`, frame `base`; `m` / `quat_wxyz` |
| `robot.arm.prev_joint_torques_computed(_safened)` | `proprio`, `N*m`, names `fr3_joint1..7` |
| `robot.arm.prev_controller_latency_ms` / `prev_command_successful` | `state` (`ms` / unitless) |
| `robot.gripper.position` | `proprio`, `normalized` (`0` open, `1` closed) |
| `<camera>.rgb` | `image`, `rgb`, `HWC`, source `<camera>`, `serial_number`, fps 15 |
| `<camera>.depth` | `depth`, `m`, `HW`, source `<camera>`, `serial_number`, fps 15 |
| `<camera>.intrinsics` / `.extrinsics` | `intrinsics` (`px`) / `extrinsics` (base -> camera), frame `camera_cv`, source `<camera>`, `serial_number` |
| action, `cartesian_position` RobotEnv | `action`, control `eef_pose`, mode `absolute`, frame `base`, units `[m]*3 + [rad]*3 (or 6D rows) + [normalized]` |
| action, `cartesian_velocity` RobotEnv | `action`, control `eef_velocity`, mode `velocity`, `normalized`, frame `base` |

Action bounds follow the mode: `cartesian_velocity` actions are `[-1, 1]`;
`cartesian_position` actions are `[-1, 1]` m, `[-pi, pi]` rad (or `[-1, 1]`
6D rows) and a `[0, 1]` gripper. The schema, its metadata and the conversion
math live in `unienv_droid.conventions`, which the dataset transformation uses
too.

### Depth units

`DroidEnv` always reports depth in **metres**. Upstream DROID's ZED reader
opens the camera without setting `coordinate_units`, so the ZED SDK default,
**millimetres**, applies (its depth retrieval is commented out upstream; forks
that re-enable it return millimetres, as does DROID's own
`postprocessing/util/svo2depth.py`). `depth_units` describes what the upstream
`RobotEnv` returns and defaults to `"mm"`, which `DroidEnv` divides by 1000.
Pass `depth_units="m"` for a camera stack that already returns metres (e.g. a
reader opened with `coordinate_units = sl.UNIT.METER`):

```python
env = DroidEnv(robotenv, depth_units="m")
```


## Usage of Droid Raw Data to retrieve depth

![SVO Bug](assets/svo_bug.png)
> Please be aware of a bug in the SVO reading code that may cause the last few frames of the SVO recording to be noise when you only read a subset of frames in an episode. This is caused by a bug in the PyZed SDK and there's really nothing we can do about it. 

First download the raw dataset

```bash
gsutil -m cp -r gs://gresearch/robotics/droid_raw/v1.0.1 <path_to_target_dir>
```

Then download the additional updated annotations for droid from [huggingface](https://huggingface.co/KarlP/droid), and unzip it and place it in `<path_to_annotation_dir>`.

```python
from unienv_droid_data.raw import RawDroidDataset
import numpy as np

path_to_target_dir = "<path_to_target_dir>" # Replace with the path where you downloaded the droid raw data
path_to_annotation_dir = "<path_to_annotation_dir>" # Replace with the path where you placed the droid annotations

dataset = RawDroidDataset(
    root_dir=path_to_target_dir, 
    droid_annotations_path=path_to_annotation_dir,
    target_resolution=sl.Resolution(672, 376), 
    success_only=True, # Only load successful episodes
    read_right=False # Whether to read additional right camera images from the stereo camera
)

# Access dataset properties
print("Raw dataset length:", len(dataset))
print("Raw dataset space:", dataset.single_space)
print("Metadata space:", dataset.single_metadata_space)

# Access data
first_step_data = dataset[0]

# Access slices of data
first_ten_steps = dataset[:10]

# Access data through indexes
indexed_data = dataset[np.array([0, 5, 10, 15])]

# Access data and metadata
first_step_data, first_step_metadata = dataset.get_at_with_metadata(0)

# Access slice of data and metadata
first_ten_steps_data, first_ten_steps_metadata = dataset.get_at_with_metadata(slice(10))

# Access trajectory (trajectory doesn't support slices)
print("Trajectory length:", dataset.trajectory_length)
first_trajectory_data = dataset.get_trajectory_at(0)
first_trajectory_data, first_trajectory_metadata = dataset.get_trajectory_at_with_metadata(0)
```

`dataset.single_space` mirrors the recordings (`trajectory.h5` + SVO) and is
described with unienv space metadata: images are RGB `HWC`, depth is `float32`
metres (the SVO recordings are opened with `coordinate_units = METER`), and the
camera `source` is the positional name (`exterior_1`, `exterior_2`, `wrist`;
`<name>_right` for right streams). `camera_extrinsics` are camera poses in the
robot base frame (position + extrinsic-XYZ euler). This space metadata
(`space.metadata`) is unrelated to `dataset.single_metadata_space`, the space of
the per-sample record returned by `get_at_with_metadata`: the episode
`file_path` and its `camera_serials` (`wrist`, `exterior_1`, `exterior_2`).

### Converting to the `DroidEnv` observation schema

`unienv_droid_data.transformations` turns the raw frames into exactly what
`DroidEnv` reports (same conversion code, same space metadata):

```python
from unienv_data.batches import TransformedBatch
from unienv_droid_data.transformations import (
    DroidRawToEnvTransformation,
    DroidRawToEnvMetadataTransformation,
)

env_dataset = TransformedBatch(
    dataset,
    DroidRawToEnvTransformation(use_tip_as_eef=False),
    metadata_transformation=DroidRawToEnvMetadataTransformation(),
)
frame, metadata = env_dataset.get_at_with_metadata(0)
frame["steps"]["observation"]["robot"]["arm"]["eef_position"]
frame["steps"]["observation"]["external_cam_0"]["rgb"]
metadata["camera_serials"]["external_cam_0"]
```

Only `steps.observation` changes; `action_dict`, the language instructions
and `is_first` / `is_last` pass through. Cameras are named by DROID position:
`wrist` -> `hand_cam`, `exterior_1` -> `external_cam_0`, `exterior_2` ->
`external_cam_1` (`camera_names={position: name}` overrides them; pass the
same mapping to both transformations). This is DROID's camera order, not the
sorted-serial order `DroidEnv` uses by default, so on a rig whose
`exterior_1` has the larger serial pass `camera_names` to one side to make the
names agree. Exterior cameras also carry the dataset-only
`extrinsics_is_postcalibrated` flag.

Note that for the resolution we picked the VGA resolution, instead you can pick any resolution supported by ZED. 
See [Zed SDK Documentation](https://www.stereolabs.com/docs/api/group__Video__group.html#gabd0374c748530a64a72872c43b2cc828) for complete list of supported resolutions.