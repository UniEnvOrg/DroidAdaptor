"""The vectorized DROID conventions math against transforms3d (per sample)."""

import numpy as np
import pytest
import transforms3d as tr3d

from unienv_droid import conventions

RNG = np.random.default_rng(0)
EULERS = RNG.uniform(-np.pi, np.pi, size=(16, 3))


def test_euler_to_quaternion_matches_transforms3d():
    quaternions = conventions.euler_to_quaternion(EULERS)
    for euler, quaternion in zip(EULERS, quaternions):
        np.testing.assert_allclose(quaternion, tr3d.euler.euler2quat(*euler), atol=1e-12)


def test_canonical_quaternion_has_nonnegative_w_and_same_rotation():
    quaternions = conventions.canonical_quaternion(conventions.euler_to_quaternion(EULERS))
    assert (quaternions[:, 0] >= 0).all()
    for euler, quaternion in zip(EULERS, quaternions):
        np.testing.assert_allclose(tr3d.quaternions.quat2mat(quaternion), tr3d.euler.euler2mat(*euler), atol=1e-12)
        np.testing.assert_allclose(
            conventions.quaternion_to_rotation_matrix(quaternion), tr3d.euler.euler2mat(*euler), atol=1e-12,
        )


def test_extrinsics_from_camera_pose_is_inverse_pose():
    poses = np.concatenate([RNG.uniform(-1, 1, size=(16, 3)), EULERS], axis=-1)
    extrinsics = conventions.extrinsics_from_camera_pose(poses)
    assert extrinsics.shape == (16, 3, 4)
    for pose, ext in zip(poses, extrinsics):
        camera_to_base = tr3d.affines.compose(pose[:3], tr3d.euler.euler2mat(*pose[3:]), np.ones(3))
        np.testing.assert_allclose(ext, np.linalg.inv(camera_to_base)[:3], atol=1e-12)


def _robot_state(batch=()):
    def full(shape, value):
        return np.full(batch + shape, value, dtype=np.float64)

    cartesian = np.concatenate([np.full(batch + (3,), 0.3), np.broadcast_to(EULERS[0], batch + (3,))], axis=-1)
    return {
        "cartesian_position": cartesian,
        "gripper_position": full((), 0.5),
        "joint_positions": full((7,), 0.1),
        "joint_velocities": full((7,), 0.2),
        "joint_torques_computed": full((7,), 1.0),
        "prev_joint_torques_computed": full((7,), 2.0),
        "prev_joint_torques_computed_safened": full((7,), 3.0),
        "motor_torques_measured": full((7,), 4.0),
        "prev_controller_latency_ms": full((), 5.0),
        "prev_command_successful": np.ones(batch, dtype=bool),
    }


@pytest.mark.parametrize("batch", [(), (4,), (2, 3)])
@pytest.mark.parametrize("use_tip_as_eef", [False, True])
def test_robot_observation_fits_the_space_for_any_leading_dims(batch, use_tip_as_eef):
    obs = conventions.robot_observation(_robot_state(batch), use_tip_as_eef=use_tip_as_eef)
    space = conventions.robot_observation_space()
    for part in ("arm", "gripper"):
        assert set(obs[part]) == set(space[part].spaces)
        for key, value in obs[part].items():
            leaf = space[part][key]
            assert value.shape == batch + leaf.shape, (part, key)
            assert value.dtype == leaf.dtype, (part, key)
    if batch == ():
        assert space.contains(obs)
    shift = np.linalg.norm(obs["arm"]["eef_position"] - 0.3, axis=-1)
    np.testing.assert_allclose(shift, conventions.TIP_EEF_OFFSET if use_tip_as_eef else 0.0, atol=1e-6)


def test_default_camera_names():
    names = conventions.default_camera_names({"30": 1, "10": 0, "20": 1})
    assert names == {"10": "hand_cam", "20": "external_cam_0", "30": "external_cam_1"}
    # Overrides; serials not on the rig are ignored.
    names = conventions.default_camera_names({"30": 1, "20": 1}, {"20": "left_cam", "99": "unused"})
    assert names == {"20": "left_cam", "30": "external_cam_1"}
