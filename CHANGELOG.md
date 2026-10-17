# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1a5 (unreleased)

- **Breaking — nested `DroidEnv` observation** (no compatibility layer):
  `obs["robot"]["arm"]` (`joint_position`, `joint_velocity`, `joint_torque`,
  `joint_motor_torque`, `eef_position`, `eef_quaternion`, and the controller
  feedback `prev_joint_torques_computed(_safened)`,
  `prev_controller_latency_ms`, `prev_command_successful`) and
  `obs["robot"]["gripper"]["position"]` (`(1,)`), plus one entry per camera.
  - Removed as redundant: `cartesian_position`, `end_effector_pos`,
    `end_effector_euler`, `end_effector_quaternion` (now
    `robot.arm.eef_quaternion`, `(w, x, y, z)` with `w >= 0`) and
    `end_effector_6d_rotation`.
  - Torques: `joint_torque` is the measured torque (upstream
    `motor_torques_measured`), `joint_motor_torque` the commanded one
    (upstream `joint_torques_computed`).
  - Cameras are keyed by role, not serial: `hand_cam` (wrist),
    `external_cam_0`, `external_cam_1`, ... (exterior, sorted-serial order),
    `<name>_right` for right streams; the new `camera_names={serial: name}`
    option overrides the names. `source` metadata is the camera name and the
    serial is the `serial_number` metadata key. `flip_ext_cams_y` mappings take
    camera names or serials.
  - Removed: the `cam_<serial>` keys, `camera_types`, `all_cam_names`,
    `all_cam_is_ext`, `ROBOT_STATE_OBSERVATION_SPACE`,
    `ADDITIONAL_STATE_OBSERVATION_SPACE`, `extrinsic_from_pose` and
    `transform_droid_eef_location_to_tip`. `camera_serials` is now
    `{camera name: serial}`; `camera_names` lists the camera streams.
  - `reset()` returns an empty `info` (it repeated the last camera's
    intrinsics / extrinsics).
- `unienv_droid.conventions` holds the schema: leaf metadata
  (`ARM_OBSERVATION_METADATA`, `GRIPPER_OBSERVATION_METADATA`,
  `camera_leaf_metadata`), space builders (`robot_observation_space`,
  `camera_observation_space`), camera naming (`default_camera_names`) and the
  vectorized conversions (`robot_observation`, `euler_to_quaternion`,
  `canonical_quaternion`, `extrinsics_from_camera_pose`, `TIP_EEF_OFFSET`).
- **`unienv_droid_data.transformations`**: `DroidRawToEnvTransformation`
  turns `RawDroidDataset` frames into the `DroidEnv` observation schema
  (`steps.observation` only; `use_tip_as_eef`, `camera_names` by DROID
  position) and `DroidRawToEnvMetadataTransformation` keys the per-frame camera
  serials by camera name; use them with `unienv_data`'s `TransformedBatch`.
- `RawDroidDataset` reads the full robot state
  (`prev_joint_torques_computed(_safened)`, `prev_controller_latency_ms`,
  `prev_command_successful`), the right streams' camera poses with
  `read_right=True` (`camera_extrinsics.<camera>_right`), and reports the
  episode's camera serials in its per-frame metadata (`camera_serials.wrist`
  / `exterior_1` / `exterior_2`).
- `RawDroidDataset` spaces carry unienv space metadata too: actions
  (`eef_pose` / `eef_velocity` / `gripper` / `joint_position` /
  `joint_velocity`), robot state (described like `DroidEnv`'s), images (RGB
  `HWC`) and depth (metres), source = the positional camera name
  (`exterior_1`, `exterior_2`, `wrist`, `<name>_right` for right streams), fps
  15, intrinsics and the camera poses (`extrinsics`, base frame,
  `euler_sxyz`). The action and observation `joint_position` leaves are now
  separate spaces (previously one shared `_FR3_JNT_POSITION_BOX`).
- Requires `unienv>=0.0.1b16`. `DroidEnv` observation and action spaces
  carry unienv space metadata: robot state (`proprio`, units, joint / element
  names, base frame, `euler_sxyz` / `quat_wxyz` / `rot6d`), cameras (`image`
  RGB `HWC`, `depth` metres, `intrinsics`, `extrinsics`, source = camera name,
  fps 15) and the action (`eef_pose` absolute vs `eef_velocity` normalized,
  per-element names / units). Shared names and builders are in the new
  `unienv_droid.conventions`.
- **Behaviour change:** `DroidEnv` now reports depth in metres. The new
  `depth_units` option (`"mm"` default, or `"m"`) describes what the upstream
  `RobotEnv` returns; upstream DROID's ZED reader leaves the SDK default
  (millimetres), so depth is divided by 1000 by default. Pass
  `depth_units="m"` if your camera stack already returns metres.
- Fixed `RawDroidDataset(read_right=True)` depth: the right depth maps were
  empty (`svo_convert` raised `ValueError: could not broadcast input array
  from shape (0,0)`), since ZED computes `MEASURE.DEPTH_RIGHT` only with
  `enable_right_side_measure`, which `svo_convert` now sets with `read_right`.
- Fixed invalid ZED depth values: `svo_convert` mapped NaN to 0 but left
  `+inf` (too far) / `-inf` (too close) as `+-3.4e38` (`np.nan_to_num`'s
  default), outside the `[0, inf)` depth space; all invalid depth is now 0.
- Fixed `RawDroidDataset(read_right=True)`, which raised `AttributeError`
  (`DictSpace` has no `update`) while building its space.
- Fixed the `svo_convert` docstring: depth is `float32` metres (the SVO is
  opened with `coordinate_units = METER`), not `uint16`.
- Fixed the `DroidEnv` action bounds for a `cartesian_position` `RobotEnv`:
  the absolute euler angles are radians and are now bounded by `[-pi, pi]`
  (they were `[-1, 1]`), and the absolute gripper position by `[0, 1]`
  (upstream clips it there). `cartesian_velocity` actions stay `[-1, 1]`.
  `DroidEnv.action_bounds` / `build_action_space` expose the bounds, and a
  `RobotEnv` with a joint action space now raises (`DroidEnv` only sends
  cartesian actions).
- Fixed `DroidEnv` on real stereo ZED rigs: the left stream (`<serial>_left`)
  is now exposed as `cam_<serial>`, and the observation space uses the same
  key. Previously the space declared `cam_<serial>_left` while observations
  were emitted under `cam_<serial>`, so `reset()` raised `KeyError`, and the
  right stream silently overwrote the left one.
- Added `include_right_images` to also expose right streams as
  `cam_<serial>_right`.
- Added `DroidEnv.cameras` (`DroidCamera` records), `camera_serials` and
  `camera_types` (`0` = wrist, `1` = exterior).
- `flip_ext_cams_y=True` now flips exterior cameras only (it used to flip the
  wrist camera too); a mapping may key cameras by serial or upstream stream key.
- Robot-state observations are cast to their declared space dtypes
  (`prev_command_successful` is `bool`, not `int32`), and
  `end_effector_quaternion`, `end_effector_6d_rotation` and `cam_*.extrinsics`
  are `float32` as declared.
- Renamed the observation-space key `prev_joint_torques_applied_safened` to
  `prev_joint_torques_computed_safened`, the key DROID actually emits.
- Fixed `use_6d_rotation_in_action=True` under NumPy 2.
- `droid` is only imported for type annotations, so `unienv_droid` imports
  without the `env` extra installed.

## 0.0.1a4

- Changed the `use_tip_as_eef` flange-to-tip offset from 0.152 m to 0.159 m
  (`DroidEnv.TIP_EEF_OFFSET`) to match the mujoco_menagerie
  franka_fr3_robotiq_2f85 CAD model, unifying the sim/real EEF convention

## 0.0.1a3 (2026-05-07)

- Added support for SVO2 recordings

## 0.0.1a2 (2026-02-10)

- Added droid environment adaptor

## 0.0.1a1 (2025-10-22)

- Initial release with support for raw data