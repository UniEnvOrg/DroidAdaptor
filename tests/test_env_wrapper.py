"""Tests for :class:`unienv_droid.DroidEnv` against an offline mock ``RobotEnv``.

The mock reproduces the upstream ``droid.robot_env.RobotEnv.get_observation``
layout of a real rig: stereo ZED streams keyed ``<serial>_left`` /
``<serial>_right``, numeric serials, ``camera_type`` keyed by bare serial, and
camera poses (position + extrinsic-XYZ euler) keyed by stream.
"""

import numpy as np
import pytest
import transforms3d as tr3d

from unienv_droid import DroidCamera, DroidEnv

WRIST = "13062452"
EXT1 = "24259877"
EXT2 = "26368109"
H, W = 6, 8


class MockRobotEnv:
    def __init__(self, action_space="cartesian_position", mono=False):
        self.action_space = action_space
        self.mono = mono
        self.actions = []
        self.cartesian_position = [0.4, 0.05, 0.3, np.pi, 0.1, -0.2]

    def reset(self):
        pass

    def step(self, action):
        self.actions.append(np.asarray(action))

    def _streams(self, serial):
        return [serial] if self.mono else [f"{serial}_left", f"{serial}_right"]

    def get_observation(self):
        image, depth, intrinsics, extrinsics = {}, {}, {}, {}
        for i, serial in enumerate((WRIST, EXT1, EXT2)):
            for j, key in enumerate(self._streams(serial)):
                bgra = np.zeros((H, W, 4), dtype=np.uint8)
                bgra[..., 0] = 10 * i + j  # blue channel carries the stream id
                image[key] = bgra
                depth[key] = np.full((H, W), 0.5 + i + 0.1 * j)
                intrinsics[key] = np.array([[100.0, 0, 3.0], [0, 100.0, 2.0], [0, 0, 1]])
                extrinsics[key] = [0.1 * i, 0.2, 0.5, 0.3, -0.2, 0.1 * j]
        return {
            "robot_state": {
                "cartesian_position": list(self.cartesian_position),
                "gripper_position": 0.25,
                "joint_positions": [0.0] * 7,
                "joint_velocities": [0.0] * 7,
                "joint_torques_computed": [0.0] * 7,
                "prev_joint_torques_computed": [0.0] * 7,
                "prev_joint_torques_computed_safened": [0.0] * 7,
                "motor_torques_measured": [0.0] * 7,
                "prev_controller_latency_ms": 3,
                "prev_command_successful": True,
            },
            "image": image,
            "depth": depth,
            "camera_intrinsics": intrinsics,
            "camera_extrinsics": extrinsics,
            "camera_type": {WRIST: 0, EXT1: 1, EXT2: 1},
        }


def _leaves(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _space_leaves(space, prefix=""):
    for key, value in space.spaces.items():
        if hasattr(value, "spaces"):
            yield from _space_leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


@pytest.mark.parametrize("mono", [False, True])
def test_reset_obs_matches_space(mono):
    env = DroidEnv(MockRobotEnv(mono=mono))
    _, obs, _ = env.reset()

    space = dict(_space_leaves(env.observation_space))
    leaves = dict(_leaves(obs))
    assert set(leaves) == set(space)
    for key, value in leaves.items():
        assert value.shape == space[key].shape, key
        assert value.dtype == space[key].dtype, key
    assert {f"cam_{s}" for s in (WRIST, EXT1, EXT2)} <= set(obs)


def test_stereo_left_stream_is_exposed_under_bare_serial():
    env = DroidEnv(MockRobotEnv())
    assert env.camera_serials == (WRIST, EXT1, EXT2)
    assert env.camera_types == {WRIST: 0, EXT1: 1, EXT2: 1}
    _, obs, _ = env.reset()
    # Blue (BGRA channel 0) becomes RGB channel 2; ``j == 0`` is the left stream.
    assert obs[f"cam_{EXT1}"]["rgb"][0, 0, 2] == 10
    assert len(env.render()) == 3


def test_include_right_images():
    env = DroidEnv(MockRobotEnv(), include_right_images=True)
    assert env.camera_serials == (
        WRIST, f"{WRIST}_right", EXT1, f"{EXT1}_right", EXT2, f"{EXT2}_right"
    )
    _, obs, _ = env.reset()
    assert obs[f"cam_{EXT1}_right"]["rgb"][0, 0, 2] == 11
    assert len(env.render()) == 3  # right streams are not rendered


def test_flip_bool_flips_exterior_cameras_only():
    env = DroidEnv(MockRobotEnv(), flip_ext_cams_y=True)
    plain = DroidEnv(MockRobotEnv())
    _, obs, _ = env.reset()
    _, ref, _ = plain.reset()
    np.testing.assert_array_equal(obs[f"cam_{WRIST}"]["extrinsics"], ref[f"cam_{WRIST}"]["extrinsics"])
    np.testing.assert_array_equal(obs[f"cam_{EXT1}"]["extrinsics"][:2], -ref[f"cam_{EXT1}"]["extrinsics"][:2])


def test_flip_mapping_accepts_serial_or_stream_key():
    env = DroidEnv(MockRobotEnv(), flip_ext_cams_y={EXT1: True, f"{WRIST}_left": True})
    plain = DroidEnv(MockRobotEnv())
    _, obs, _ = env.reset()
    _, ref, _ = plain.reset()
    for serial, flipped in ((WRIST, True), (EXT1, True), (EXT2, False)):
        sign = -1 if flipped else 1
        np.testing.assert_allclose(
            obs[f"cam_{serial}"]["extrinsics"][0], sign * ref[f"cam_{serial}"]["extrinsics"][0]
        )


def test_robot_state_dtypes_follow_space():
    env = DroidEnv(MockRobotEnv())
    _, obs, _ = env.reset()
    assert obs["prev_command_successful"].dtype == np.bool_
    assert obs["prev_controller_latency_ms"].dtype == np.float32
    assert "prev_joint_torques_computed_safened" in env.observation_space.spaces


def test_derived_keys():
    robotenv = MockRobotEnv()
    env = DroidEnv(robotenv)
    _, obs, _ = env.reset()
    rotation = tr3d.euler.euler2mat(*robotenv.cartesian_position[3:])
    np.testing.assert_allclose(obs["end_effector_6d_rotation"], rotation[:2].reshape(-1), atol=1e-6)
    np.testing.assert_allclose(
        tr3d.quaternions.quat2mat(obs["end_effector_quaternion"]), rotation, atol=1e-6
    )
    np.testing.assert_allclose(obs["end_effector_pos"], robotenv.cartesian_position[:3], atol=1e-6)


def test_6d_rotation_action_roundtrip():
    robotenv = MockRobotEnv()
    env = DroidEnv(robotenv, use_6d_rotation_in_action=True)
    env.reset()
    euler = np.array([np.pi - 0.1, 0.2, -0.3])
    rotation = tr3d.euler.euler2mat(*euler)
    action = np.concatenate([[0.4, 0.0, 0.2], rotation[:2].reshape(-1), [1.0]]).astype(np.float32)
    env.step(action)
    sent = robotenv.actions[-1]
    assert sent.shape == (7,)
    np.testing.assert_allclose(tr3d.euler.euler2mat(*sent[3:6]), rotation, atol=1e-5)
    assert sent[6] == 1.0


def test_tip_eef_roundtrip():
    robotenv = MockRobotEnv()
    env = DroidEnv(robotenv, use_tip_as_eef=True)
    _, obs, _ = env.reset()
    env.step(np.concatenate([obs["cartesian_position"], [0.0]]).astype(np.float32))
    np.testing.assert_allclose(robotenv.actions[-1][:6], robotenv.cartesian_position, atol=1e-5)


def test_discover_cameras_static():
    cams = DroidEnv.discover_cameras(MockRobotEnv().get_observation())
    assert cams[0] == DroidCamera(WRIST, f"{WRIST}_left", WRIST, 0)
    assert not cams[0].is_ext and cams[1].is_ext
