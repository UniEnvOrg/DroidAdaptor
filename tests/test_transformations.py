"""``DroidRawToEnvTransformation`` / ``DroidRawToEnvMetadataTransformation`` on synthetic raw frames.

The transformed raw frame must equal what ``DroidEnv`` reports for the same
robot state and camera data, and fit the transformed space.
"""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pyzed")
pytest.importorskip("serde")

from unienv_interface.space.space_utils import batch_utils as sbu  # noqa: E402
from unienv_interface.space.space_utils.metadata_utils import validate_metadata  # noqa: E402
from unienv_interface.transformations import serialization_utils as tsu  # noqa: E402

from unienv_droid import DroidEnv  # noqa: E402
from unienv_droid_data.raw.raw_data import _get_frame_space, _get_metadata_space  # noqa: E402
from unienv_droid_data.transformations import (  # noqa: E402
    DroidRawToEnvMetadataTransformation,
    DroidRawToEnvTransformation,
)

H, W = 4, 5
SERIALS = {"wrist": "13062452", "exterior_1": "26368109", "exterior_2": "24259877"}
RNG = np.random.default_rng(0)
CARTESIAN = np.array([0.4, 0.05, 0.3, np.pi - 0.2, 0.1, -0.2])
POSES = {position: np.concatenate([RNG.uniform(-1, 1, 3), RNG.uniform(-3, 3, 3)]) for position in SERIALS}
ROBOT_STATE = {
    "cartesian_position": CARTESIAN,
    "gripper_position": 0.25,
    "joint_positions": RNG.uniform(-1, 1, 7),
    "joint_velocities": RNG.uniform(-1, 1, 7),
    "joint_torques_computed": RNG.uniform(-1, 1, 7),
    "prev_joint_torques_computed": RNG.uniform(-1, 1, 7),
    "prev_joint_torques_computed_safened": RNG.uniform(-1, 1, 7),
    "motor_torques_measured": RNG.uniform(-1, 1, 7),
    "prev_controller_latency_ms": 3.0,
    "prev_command_successful": True,
}
CAMERA_KEYS = {
    "wrist": ("wrist_image", "wrist_depth", "wrist_camera"),
    "exterior_1": ("exterior_image_1", "exterior_depth_1", "exterior_camera_1"),
    "exterior_2": ("exterior_image_2", "exterior_depth_2", "exterior_camera_2"),
}


def _raw_frame(read_right=False):
    """A single raw frame (``RawDroidDataset`` layout) of ROBOT_STATE and POSES."""
    sides = ("left", "right") if read_right else ("left",)
    obs = {
        "cartesian_position": CARTESIAN.astype(np.float32),
        "gripper_position": np.float32(ROBOT_STATE["gripper_position"]),
        "joint_position": ROBOT_STATE["joint_positions"].astype(np.float32),
        **{
            key: np.asarray(ROBOT_STATE[key], dtype=np.float32)
            for key in (
                "joint_velocities", "joint_torques_computed", "motor_torques_measured",
                "prev_joint_torques_computed", "prev_joint_torques_computed_safened",
                "prev_controller_latency_ms",
            )
        },
        "prev_command_successful": np.bool_(True),
        "camera_intrinsics": {},
        "camera_extrinsics": {
            "exterior_camera_1_is_postcalibrated": np.bool_(True),
            "exterior_camera_2_is_postcalibrated": np.bool_(False),
        },
    }
    for i, (position, (image, depth, camera)) in enumerate(CAMERA_KEYS.items()):
        for j, side in enumerate(sides):
            obs[f"{image}_{side}"] = np.full((H, W, 3), 10 * i + j, dtype=np.uint8)
            obs[f"{depth}_{side}"] = np.full((H, W), 0.5 + i, dtype=np.float32)
            obs["camera_intrinsics"][f"{camera}_{side}"] = np.eye(3, dtype=np.float32) * (i + 1)
            obs["camera_extrinsics"][f"{camera}_{side}"] = POSES[position].astype(np.float32)
    return {
        "steps": {
            "action_dict": {"cartesian_position": CARTESIAN.astype(np.float32)},
            "observation": obs,
            "is_first": np.uint8(1),
        }
    }


class _MockRobotEnv:
    """The same robot state / camera poses as ``_raw_frame``, as upstream ``RobotEnv`` reports them."""
    action_space = "cartesian_position"

    def get_observation(self):
        streams = {SERIALS[position]: position for position in SERIALS}
        return {
            "robot_state": dict(ROBOT_STATE),
            "image": {f"{s}_left": np.zeros((H, W, 4), dtype=np.uint8) for s in streams},
            "depth": {f"{s}_left": np.zeros((H, W)) for s in streams},
            "camera_intrinsics": {f"{s}_left": np.eye(3) for s in streams},
            "camera_extrinsics": {f"{s}_left": POSES[p].astype(np.float32) for s, p in streams.items()},
            "camera_type": {s: 0 if p == "wrist" else 1 for s, p in streams.items()},
        }


def _leaves(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


@pytest.mark.parametrize("read_right", [False, True])
@pytest.mark.parametrize("use_tip_as_eef", [False, True])
def test_transformed_frame_matches_droid_env(read_right, use_tip_as_eef):
    transform = DroidRawToEnvTransformation(use_tip_as_eef=use_tip_as_eef)
    source_space = _get_frame_space(H, W, read_right=read_right)
    target_space = transform.get_target_space_from_source(source_space)
    validate_metadata(target_space)

    frame = transform.transform(source_space, _raw_frame(read_right))
    assert target_space["steps"]["observation"].contains(frame["steps"]["observation"])
    assert frame["steps"]["is_first"] == 1  # untouched

    # DroidEnv names exterior cameras by sorted serial: exterior_2 (24259877) is
    # external_cam_0 there, so map the dataset's positions to the env's names.
    env = DroidEnv(_MockRobotEnv(), use_tip_as_eef=use_tip_as_eef, camera_names={
        SERIALS["exterior_1"]: "external_cam_0", SERIALS["exterior_2"]: "external_cam_1",
    })
    env_obs = env.extract_obs(env.robotenv.get_observation())
    for key, value in _leaves(env_obs["robot"]):
        np.testing.assert_allclose(dict(_leaves(frame["steps"]["observation"]["robot"]))[key], value, atol=1e-6)
    for name in ("hand_cam", "external_cam_0", "external_cam_1"):
        np.testing.assert_allclose(
            frame["steps"]["observation"][name]["extrinsics"], env_obs[name]["extrinsics"], atol=1e-5,
        )
    # Shared leaves are described identically.
    env_space = dict(_leaves_space(env.observation_space))
    for key, leaf in _leaves_space(target_space["steps"]["observation"]):
        if key in env_space:
            md = {k: v for k, v in env_space[key].metadata.items() if k != "serial_number"}
            assert leaf.metadata == md, key

    obs = frame["steps"]["observation"]
    assert set(obs) == {"robot", "hand_cam", "external_cam_0", "external_cam_1"} | (
        {"hand_cam_right", "external_cam_0_right", "external_cam_1_right"} if read_right else set()
    )
    assert obs["external_cam_0"]["extrinsics_is_postcalibrated"]
    assert not obs["external_cam_1"]["extrinsics_is_postcalibrated"]
    assert obs["external_cam_0"]["rgb"][0, 0, 0] == 10
    if read_right:
        assert obs["external_cam_0_right"]["rgb"][0, 0, 0] == 11


def _leaves_space(space, prefix=""):
    for key, value in space.spaces.items():
        if hasattr(value, "spaces"):
            yield from _leaves_space(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def test_batched_frames():
    transform = DroidRawToEnvTransformation()
    source_space = _get_frame_space(H, W)
    frame = _raw_frame()
    batch = {"steps": {
        key: (
            {k: np.stack([v, v]) for k, v in value.items()} if key == "action_dict"
            else np.stack([value, value]) if key == "is_first"
            else None
        )
        for key, value in frame["steps"].items()
    }}

    def stack(tree):
        return {k: stack(v) if isinstance(v, dict) else np.stack([v, v]) for k, v in tree.items()}

    batch["steps"]["observation"] = stack(frame["steps"]["observation"])
    out = transform.transform(sbu.batch_space(source_space, 2), batch)
    single = transform.transform(source_space, frame)
    target = sbu.batch_space(transform.get_target_space_from_source(source_space), 2)
    assert target["steps"]["observation"].contains(out["steps"]["observation"])
    for key, value in _leaves(single["steps"]["observation"]):
        np.testing.assert_allclose(dict(_leaves(out["steps"]["observation"]))[key][1], value, atol=1e-6)


def test_camera_names_override_and_validation():
    transform = DroidRawToEnvTransformation(camera_names={"exterior_1": "external_cam_1", "exterior_2": "external_cam_0"})
    frame = transform.transform(_get_frame_space(H, W), _raw_frame())
    assert frame["steps"]["observation"]["external_cam_1"]["rgb"][0, 0, 0] == 10
    with pytest.raises(ValueError):
        DroidRawToEnvTransformation(camera_names={"exterior_3": "x"})
    with pytest.raises(ValueError):
        DroidRawToEnvTransformation(camera_names={"exterior_1": "external_cam_1"})


def test_metadata_transformation():
    transform = DroidRawToEnvMetadataTransformation()
    source = _get_metadata_space()
    target = transform.get_target_space_from_source(source)
    assert set(target["camera_serials"].spaces) == {"hand_cam", "external_cam_0", "external_cam_1"}
    out = transform.transform(source, {"file_path": "ep", "camera_serials": dict(SERIALS)})
    assert out == {"file_path": "ep", "camera_serials": {
        "hand_cam": SERIALS["wrist"], "external_cam_0": SERIALS["exterior_1"], "external_cam_1": SERIALS["exterior_2"],
    }}
    assert target.contains(out)


def test_serialization_roundtrip():
    for transform in (
        DroidRawToEnvTransformation(use_tip_as_eef=True, camera_names={"wrist": "wrist_cam"}),
        DroidRawToEnvMetadataTransformation(camera_names={"wrist": "wrist_cam"}),
    ):
        restored = tsu.json_to_transformation(tsu.transformation_to_json(transform))
        assert type(restored) is type(transform)
        assert restored.serialize() == transform.serialize()
