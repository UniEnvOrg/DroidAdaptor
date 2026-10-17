"""Tests for :class:`unienv_droid.DroidEnv` against an offline mock ``RobotEnv``.

The mock reproduces the upstream ``droid.robot_env.RobotEnv.get_observation``
layout of a real rig: stereo ZED streams keyed ``<serial>_left`` /
``<serial>_right``, numeric serials, ``camera_type`` keyed by bare serial, and
camera poses (position + extrinsic-XYZ euler) keyed by stream.
"""

import numpy as np
import pytest
import transforms3d as tr3d

from unienv_interface.space.space_utils.metadata_utils import find_spaces_by_metadata, validate_metadata

from unienv_droid import DroidCamera, DroidEnv
from unienv_droid import conventions

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
                "joint_positions": [0.1 * i for i in range(7)],
                "joint_velocities": [0.2] * 7,
                "joint_torques_computed": [1.0] * 7,
                "prev_joint_torques_computed": [2.0] * 7,
                "prev_joint_torques_computed_safened": [3.0] * 7,
                "motor_torques_measured": [4.0] * 7,
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


ROBOT_LEAVES = {
    "robot.arm.joint_position", "robot.arm.joint_velocity", "robot.arm.joint_torque",
    "robot.arm.joint_motor_torque", "robot.arm.eef_position", "robot.arm.eef_quaternion",
    "robot.arm.prev_joint_torques_computed", "robot.arm.prev_joint_torques_computed_safened",
    "robot.arm.prev_controller_latency_ms", "robot.arm.prev_command_successful",
    "robot.gripper.position",
}


@pytest.mark.parametrize("mono", [False, True])
def test_reset_obs_matches_space(mono):
    env = DroidEnv(MockRobotEnv(mono=mono))
    _, obs, info = env.reset()
    assert info == {}

    space = dict(_space_leaves(env.observation_space))
    leaves = dict(_leaves(obs))
    assert set(leaves) == set(space)
    for key, value in leaves.items():
        assert value.shape == space[key].shape, key
        assert value.dtype == space[key].dtype, key
        assert space[key].contains(value), key
    assert ROBOT_LEAVES <= set(leaves)
    assert set(obs) == {"robot", "hand_cam", "external_cam_0", "external_cam_1"}


def test_camera_names_default_to_role_and_sorted_serial():
    env = DroidEnv(MockRobotEnv())
    assert env.camera_names == ("hand_cam", "external_cam_0", "external_cam_1")
    assert env.camera_serials == {"hand_cam": WRIST, "external_cam_0": EXT1, "external_cam_1": EXT2}
    _, obs, _ = env.reset()
    # Blue (BGRA channel 0) becomes RGB channel 2; ``j == 0`` is the left stream.
    assert obs["external_cam_0"]["rgb"][0, 0, 2] == 10
    assert len(env.render()) == 3
    assert env.observation_space["external_cam_1"]["rgb"].metadata["serial_number"] == EXT2


def test_camera_names_override():
    env = DroidEnv(MockRobotEnv(), camera_names={EXT1: "external_cam_1", EXT2: "external_cam_0"})
    assert env.camera_serials == {"hand_cam": WRIST, "external_cam_1": EXT1, "external_cam_0": EXT2}
    _, obs, _ = env.reset()
    assert obs["external_cam_1"]["rgb"][0, 0, 2] == 10
    assert env.observation_space["external_cam_1"]["rgb"].metadata["source"] == "external_cam_1"


@pytest.mark.parametrize("bad", [{EXT1: "external_cam_1"}, {EXT1: "robot"}, {EXT1: "cam_right"}])
def test_camera_names_rejects_duplicate_and_reserved_names(bad):
    with pytest.raises(ValueError):
        DroidEnv(MockRobotEnv(), camera_names=bad)


def test_include_right_images():
    env = DroidEnv(MockRobotEnv(), include_right_images=True)
    assert env.camera_names == (
        "hand_cam", "hand_cam_right", "external_cam_0", "external_cam_0_right",
        "external_cam_1", "external_cam_1_right",
    )
    assert env.camera_serials["external_cam_0_right"] == EXT1
    _, obs, _ = env.reset()
    assert obs["external_cam_0_right"]["rgb"][0, 0, 2] == 11
    assert len(env.render()) == 3  # right streams are not rendered


def test_flip_bool_flips_exterior_cameras_only():
    env = DroidEnv(MockRobotEnv(), flip_ext_cams_y=True)
    plain = DroidEnv(MockRobotEnv())
    _, obs, _ = env.reset()
    _, ref, _ = plain.reset()
    np.testing.assert_array_equal(obs["hand_cam"]["extrinsics"], ref["hand_cam"]["extrinsics"])
    np.testing.assert_array_equal(obs["external_cam_0"]["extrinsics"][:2], -ref["external_cam_0"]["extrinsics"][:2])


def test_flip_mapping_accepts_name_or_serial():
    env = DroidEnv(MockRobotEnv(), flip_ext_cams_y={"external_cam_0": True, WRIST: True})
    plain = DroidEnv(MockRobotEnv())
    _, obs, _ = env.reset()
    _, ref, _ = plain.reset()
    for name, flipped in (("hand_cam", True), ("external_cam_0", True), ("external_cam_1", False)):
        sign = -1 if flipped else 1
        np.testing.assert_allclose(obs[name]["extrinsics"][0], sign * ref[name]["extrinsics"][0])


def test_robot_observation_values():
    robotenv = MockRobotEnv()
    env = DroidEnv(robotenv)
    _, obs, _ = env.reset()
    arm, gripper = obs["robot"]["arm"], obs["robot"]["gripper"]
    np.testing.assert_allclose(arm["joint_position"], [0.1 * i for i in range(7)], atol=1e-6)
    # joint_torque is the measured torque, joint_motor_torque the commanded one.
    np.testing.assert_array_equal(arm["joint_torque"], 4.0)
    np.testing.assert_array_equal(arm["joint_motor_torque"], 1.0)
    np.testing.assert_array_equal(arm["prev_joint_torques_computed"], 2.0)
    np.testing.assert_array_equal(arm["prev_joint_torques_computed_safened"], 3.0)
    assert arm["prev_command_successful"].dtype == np.bool_
    assert arm["prev_controller_latency_ms"].dtype == np.float32
    np.testing.assert_allclose(gripper["position"], [0.25])

    np.testing.assert_allclose(arm["eef_position"], robotenv.cartesian_position[:3], atol=1e-6)
    rotation = tr3d.euler.euler2mat(*robotenv.cartesian_position[3:])
    np.testing.assert_allclose(tr3d.quaternions.quat2mat(arm["eef_quaternion"]), rotation, atol=1e-6)
    assert arm["eef_quaternion"][0] >= 0


def test_extrinsics_are_base_to_camera():
    robotenv = MockRobotEnv()
    _, obs, _ = DroidEnv(robotenv).reset()
    pose = np.asarray(robotenv.get_observation()["camera_extrinsics"][f"{EXT1}_left"], dtype=np.float64)
    camera_to_base = tr3d.affines.compose(pose[:3], tr3d.euler.euler2mat(*pose[3:]), np.ones(3))
    np.testing.assert_allclose(obs["external_cam_0"]["extrinsics"], np.linalg.inv(camera_to_base)[:3], atol=1e-6)


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
    arm = obs["robot"]["arm"]
    rotation = tr3d.quaternions.quat2mat(arm["eef_quaternion"].astype(np.float64))
    np.testing.assert_allclose(
        arm["eef_position"], np.asarray(robotenv.cartesian_position[:3]) + DroidEnv.TIP_EEF_OFFSET * rotation[:, 2],
        atol=1e-6,
    )
    # Commanding the reported tip pose sends the current flange pose upstream.
    euler = np.asarray(tr3d.euler.mat2euler(rotation))
    env.step(np.concatenate([arm["eef_position"], euler, [0.0]]).astype(np.float32))
    np.testing.assert_allclose(robotenv.actions[-1][:3], robotenv.cartesian_position[:3], atol=1e-5)


def test_discover_cameras_static():
    cams = DroidEnv.discover_cameras(MockRobotEnv().get_observation())
    assert cams[0] == DroidCamera("hand_cam", f"{WRIST}_left", WRIST, 0)
    assert not cams[0].is_ext and cams[1].is_ext


@pytest.mark.parametrize("use_6d", [False, True])
def test_absolute_action_bounds_cover_radians(use_6d):
    env = DroidEnv(MockRobotEnv("cartesian_position"), use_6d_rotation_in_action=use_6d)
    low, high = env.action_space.low, env.action_space.high
    np.testing.assert_array_equal(low[:3], -1.0)
    np.testing.assert_array_equal(high[:3], 1.0)
    if use_6d:
        np.testing.assert_array_equal(low[3:9], -1.0)
        np.testing.assert_array_equal(high[3:9], 1.0)
    else:
        # Absolute euler angles are radians, not normalized.
        np.testing.assert_allclose(low[3:6], -np.pi, rtol=1e-6)
        np.testing.assert_allclose(high[3:6], np.pi, rtol=1e-6)
    # Upstream clips an absolute gripper command to [0, 1].
    assert (low[-1], high[-1]) == (0.0, 1.0)

    # A reachable absolute pose with |euler| > 1 rad lies in the space.
    euler = np.array([np.pi - 0.1, 0.2, -2.5])
    rotation = euler if not use_6d else tr3d.euler.euler2mat(*euler)[:2].reshape(-1)
    action = np.concatenate([[0.4, 0.0, 0.2], rotation, [1.0]]).astype(np.float32)
    assert env.action_space.contains(action)


@pytest.mark.parametrize("use_6d", [False, True])
def test_velocity_action_bounds_stay_normalized(use_6d):
    env = DroidEnv(MockRobotEnv("cartesian_velocity"), use_6d_rotation_in_action=use_6d)
    assert env.action_space.shape == ((10,) if use_6d else (7,))
    np.testing.assert_array_equal(env.action_space.low, -1.0)
    np.testing.assert_array_equal(env.action_space.high, 1.0)


def test_unsupported_action_space_raises():
    with pytest.raises(ValueError, match="action_space"):
        DroidEnv(MockRobotEnv("joint_position"))


def test_depth_defaults_to_millimetres_and_is_reported_in_metres():
    robotenv = MockRobotEnv()
    raw = robotenv.get_observation()["depth"][f"{EXT1}_left"]
    _, obs, _ = DroidEnv(robotenv).reset()
    depth = obs["external_cam_0"]["depth"]
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, raw / 1000.0, rtol=1e-6)


def test_depth_units_metres_passes_through():
    robotenv = MockRobotEnv()
    raw = robotenv.get_observation()["depth"][f"{EXT1}_left"]
    _, obs, _ = DroidEnv(robotenv, depth_units="m").reset()
    np.testing.assert_allclose(obs["external_cam_0"]["depth"], raw, rtol=1e-6)


def test_depth_conversion_follows_flip():
    robotenv = MockRobotEnv()
    raw_ramp = np.arange(H * W, dtype=np.float64).reshape(H, W) * 10.0
    orig = robotenv.get_observation

    def get_observation():
        out = orig()
        out["depth"][f"{EXT1}_left"] = raw_ramp.astype(np.uint16)  # e.g. a fork saving uint16 mm
        return out

    robotenv.get_observation = get_observation
    _, obs, _ = DroidEnv(robotenv, flip_ext_cams_y=True).reset()
    np.testing.assert_allclose(obs["external_cam_0"]["depth"], raw_ramp[::-1, ::-1] / 1000.0, rtol=1e-6)
    assert obs["external_cam_0"]["depth"].dtype == np.float32


def test_depth_units_rejects_unknown():
    with pytest.raises(ValueError, match="depth_units"):
        DroidEnv(MockRobotEnv(), depth_units="cm")


@pytest.mark.parametrize("action_space", ["cartesian_position", "cartesian_velocity"])
@pytest.mark.parametrize("use_6d", [False, True])
@pytest.mark.parametrize("include_right", [False, True])
def test_spaces_are_described(action_space, use_6d, include_right):
    env = DroidEnv(
        MockRobotEnv(action_space),
        use_6d_rotation_in_action=use_6d,
        include_right_images=include_right,
    )
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)
    # Every observation leaf is described.
    for key, leaf in _space_leaves(env.observation_space):
        assert "kind" in leaf.metadata, key


def test_observation_metadata():
    env = DroidEnv(MockRobotEnv(), include_right_images=True)
    space = env.observation_space
    arm = space["robot"]["arm"]
    assert arm["eef_position"].metadata["frame"] == "base"
    assert arm["eef_quaternion"].metadata["rotation"] == "quat_wxyz"
    assert arm["joint_position"].metadata["units"] == "rad"
    assert arm["joint_velocity"].metadata["units"] == "rad/s"
    assert arm["joint_torque"].metadata["units"] == "N*m"
    assert list(arm["joint_position"].metadata["names"]) == list(conventions.FR3_JOINT_NAMES)
    assert arm["prev_controller_latency_ms"].metadata["units"] == "ms"
    assert space["robot"]["gripper"]["position"].metadata["units"] == "normalized"
    for key, md in conventions.ARM_OBSERVATION_METADATA.items():
        assert arm[key].metadata == md, key

    for name, serial in (("hand_cam", WRIST), ("external_cam_0", EXT1), ("external_cam_0_right", EXT1)):
        cam = space[name]
        assert cam["rgb"].metadata == {
            "kind": "image", "format": "rgb", "layout": "HWC", "source": name, "fps": 15,
            "serial_number": serial,
        }
        assert cam["depth"].metadata == {
            "kind": "depth", "units": "m", "layout": "HW", "source": name, "fps": 15,
            "serial_number": serial,
        }
        assert cam["intrinsics"].metadata["kind"] == "intrinsics"
        assert cam["extrinsics"].metadata["kind"] == "extrinsics"
        assert cam["extrinsics"].metadata["frame"] == "camera_cv"
    images = find_spaces_by_metadata(space, kind="image")
    assert len(images) == 6


@pytest.mark.parametrize("use_6d", [False, True])
def test_action_metadata(use_6d):
    absolute = DroidEnv(MockRobotEnv("cartesian_position"), use_6d_rotation_in_action=use_6d).action_space.metadata
    velocity = DroidEnv(MockRobotEnv("cartesian_velocity"), use_6d_rotation_in_action=use_6d).action_space.metadata
    n = 10 if use_6d else 7
    assert absolute["kind"] == velocity["kind"] == "action"
    assert (absolute["control"], absolute["mode"]) == ("eef_pose", "absolute")
    assert (velocity["control"], velocity["mode"]) == ("eef_velocity", "velocity")
    assert velocity["normalized"] is True and velocity["units"] == "normalized"
    assert "normalized" not in absolute
    assert absolute["frame"] == velocity["frame"] == "base"
    assert absolute["rotation"] == ("rot6d" if use_6d else "euler_sxyz")
    assert len(absolute["names"]) == len(absolute["units"]) == n
    assert absolute["names"][-1] == "gripper"
    assert list(absolute["units"][:4]) == ["m", "m", "m", "normalized" if use_6d else "rad"]
