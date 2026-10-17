# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1a5 (unreleased)

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