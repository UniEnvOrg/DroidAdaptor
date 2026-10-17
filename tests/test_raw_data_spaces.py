"""Offline tests of the raw DROID dataset's spaces (no recordings needed)."""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pyzed")
pytest.importorskip("serde")

from unienv_droid_data.raw.raw_data import _get_frame_space  # noqa: E402

H, W = 4, 5


@pytest.mark.parametrize("read_right", [False, True])
def test_frame_space_builds(read_right):
    space = _get_frame_space(H, W, read_right=read_right)
    obs = space["steps"]["observation"]
    for cam in ("exterior_image_1", "exterior_image_2", "wrist_image"):
        assert obs[f"{cam}_left"].shape == (H, W, 3)
        assert (f"{cam}_right" in obs.spaces) == read_right
    for cam in ("exterior_depth_1", "exterior_depth_2", "wrist_depth"):
        assert obs[f"{cam}_left"].shape == (H, W)
        assert obs[f"{cam}_left"].dtype == np.float32
    assert ("wrist_camera_right" in obs["camera_intrinsics"].spaces) == read_right


@pytest.mark.parametrize("read_right", [False, True])
def test_frame_space_is_described(read_right):
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    space = _get_frame_space(H, W, read_right=read_right)
    validate_metadata(space)
    obs = space["steps"]["observation"]
    sides = ("left", "right") if read_right else ("left",)
    for side in sides:
        suffix = "" if side == "left" else "_right"
        for position, image, depth in (
            ("exterior_1", "exterior_image_1", "exterior_depth_1"),
            ("exterior_2", "exterior_image_2", "exterior_depth_2"),
            ("wrist", "wrist_image", "wrist_depth"),
        ):
            assert obs[f"{image}_{side}"].metadata == {
                "kind": "image", "format": "rgb", "layout": "HWC", "source": position + suffix, "fps": 15,
            }
            assert obs[f"{depth}_{side}"].metadata == {
                "kind": "depth", "units": "m", "layout": "HW", "source": position + suffix, "fps": 15,
            }
        assert obs["camera_intrinsics"][f"wrist_camera_{side}"].metadata["source"] == "wrist" + suffix
    pose = obs["camera_extrinsics"]["exterior_camera_1_left"].metadata
    assert (pose["kind"], pose["frame"], pose["rotation"], pose["source"]) == (
        "extrinsics", "base", "euler_sxyz", "exterior_1",
    )


def test_joint_position_action_and_proprio_are_separate_spaces():
    space = _get_frame_space(H, W)
    action = space["steps"]["action_dict"]["joint_position"]
    proprio = space["steps"]["observation"]["joint_position"]
    assert action is not proprio
    assert action == proprio  # same domain
    assert action.metadata["kind"] == "action"
    assert (action.metadata["control"], action.metadata["mode"]) == ("joint_position", "absolute")
    assert proprio.metadata["kind"] == "proprio"
    assert proprio.metadata["units"] == action.metadata["units"] == "rad"


def test_action_and_observation_metadata():
    space = _get_frame_space(H, W)
    actions = space["steps"]["action_dict"]
    assert actions["cartesian_position"].metadata["mode"] == "absolute"
    assert actions["cartesian_position"].metadata["rotation"] == "euler_sxyz"
    assert actions["cartesian_velocity"].metadata["normalized"] is True
    assert actions["gripper_position"].metadata["control"] == "gripper"
    obs = space["steps"]["observation"]
    from unienv_droid import conventions

    assert obs["cartesian_position"].metadata == conventions.cartesian_position_metadata()
    assert obs["joint_velocities"].metadata["units"] == "rad/s"
    assert obs["gripper_position"].metadata["units"] == "normalized"


def test_full_robot_state_is_read():
    obs = _get_frame_space(H, W)["steps"]["observation"]
    for key in (
        "prev_joint_torques_computed", "prev_joint_torques_computed_safened",
        "prev_controller_latency_ms", "prev_command_successful",
    ):
        assert key in obs.spaces, key
    assert obs["prev_controller_latency_ms"].metadata["units"] == "ms"


@pytest.mark.parametrize("read_right", [False, True])
def test_right_stream_extrinsics(read_right):
    extrinsics = _get_frame_space(H, W, read_right=read_right)["steps"]["observation"]["camera_extrinsics"]
    for camera in ("wrist_camera", "exterior_camera_1", "exterior_camera_2"):
        assert (f"{camera}_right" in extrinsics.spaces) == read_right
    if read_right:
        assert extrinsics["exterior_camera_1_right"].metadata["source"] == "exterior_1_right"


def test_metadata_space_has_camera_serials():
    from unienv_droid_data.raw.raw_data import _get_metadata_space

    space = _get_metadata_space()
    assert set(space["camera_serials"].spaces) == {"wrist", "exterior_1", "exterior_2"}
