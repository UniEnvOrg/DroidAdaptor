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

Each camera is exposed as `obs["cam_<serial>"]` with `rgb` (`uint8`, RGB),
`depth` (`float32`, metres), `intrinsics` (OpenCV `3x3`) and `extrinsics`
(world -> camera, `3x4`).
For stereo ZED cameras this is the left stream; pass
`include_right_images=True` to also get `cam_<serial>_right`.
`env.camera_types` maps each camera name to its type (`0` = wrist,
`1` = exterior).

### Space metadata

Every observation and action leaf carries unienv space metadata
(`space.metadata`, unienv >= 0.0.1b16) describing what it holds; the
vocabulary is unienv's (`kind`, `units`, `frame`, `rotation`, `names`, ...):

| Leaf | Metadata |
|---|---|
| `cartesian_position` | `proprio`, `[m, m, m, rad, rad, rad]`, names `[x, y, z, rx, ry, rz]`, frame `base`, rotation `euler_sxyz` (extrinsic XYZ, transforms3d default) |
| `end_effector_pos` / `_euler` / `_quaternion` / `_6d_rotation` | `proprio`, frame `base`; `m` / `rad` + `euler_sxyz` / `quat_wxyz` / `rot6d` (first two rows of the rotation matrix) |
| `gripper_position` | `proprio`, `normalized` (`0` open, `1` closed) |
| `joint_positions` / `joint_velocities` / `*torques*` | `proprio`, `rad` / `rad/s` / `N*m`, names `fr3_joint1..7` |
| `prev_controller_latency_ms` / `prev_command_successful` | `state` (`ms` / unitless) |
| `cam_<name>.rgb` | `image`, `rgb`, `HWC`, source `<name>`, fps 15 |
| `cam_<name>.depth` | `depth`, `m`, `HW`, source `<name>`, fps 15 |
| `cam_<name>.intrinsics` / `.extrinsics` | `intrinsics` (`px`) / `extrinsics` (base -> camera), frame `camera_cv`, source `<name>` |
| action, `cartesian_position` RobotEnv | `action`, control `eef_pose`, mode `absolute`, frame `base`, units `[m]*3 + [rad]*3 (or 6D rows) + [normalized]` |
| action, `cartesian_velocity` RobotEnv | `action`, control `eef_velocity`, mode `velocity`, `normalized`, frame `base` |

Action bounds follow the mode: `cartesian_velocity` actions are `[-1, 1]`;
`cartesian_position` actions are `[-1, 1]` m, `[-pi, pi]` rad (or `[-1, 1]`
6D rows) and a `[0, 1]` gripper. The shared names and builders live in
`unienv_droid.conventions`, which the raw dataset uses too.

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

Note that for the resolution we picked the VGA resolution, instead you can pick any resolution supported by ZED. 
See [Zed SDK Documentation](https://www.stereolabs.com/docs/api/group__Video__group.html#gabd0374c748530a64a72872c43b2cc828) for complete list of supported resolutions.