"""End to end on a real DROID recording: ``RawDroidDataset`` -> ``DroidRawToEnvTransformation``.

Needs the DROID raw dataset (``DROID_RAW_ROOT``, default
``/share/fang/droid_raw_v1.0.1``, with its ``unienv_success_dataset_idx.json``),
``pyzed`` and a GPU for SVO depth; skipped otherwise.
"""

import os

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pyzed")
pytest.importorskip("serde")

ROOT = os.environ.get("DROID_RAW_ROOT", "/share/fang/droid_raw_v1.0.1")
if not os.path.exists(os.path.join(ROOT, "unienv_success_dataset_idx.json")):
    pytest.skip(f"DROID raw dataset not found at {ROOT}", allow_module_level=True)

import transforms3d as tr3d  # noqa: E402
from unienv_data.batches import TransformedBatch  # noqa: E402
from unienv_interface.space.space_utils import batch_utils as sbu  # noqa: E402
from unienv_interface.space.space_utils.metadata_utils import validate_metadata  # noqa: E402

from unienv_droid import conventions  # noqa: E402
from unienv_droid_data.raw.raw_data import RawDroidDataset  # noqa: E402
from unienv_droid_data.transformations import (  # noqa: E402
    DroidRawToEnvMetadataTransformation,
    DroidRawToEnvTransformation,
)


@pytest.fixture(scope="module")
def raw():
    # success_only=True reads the existing index; False would scan the dataset.
    return RawDroidDataset(ROOT, success_only=True)


def test_first_frames_of_an_episode(raw):
    dataset = TransformedBatch(
        raw, DroidRawToEnvTransformation(), metadata_transformation=DroidRawToEnvMetadataTransformation(),
    )
    validate_metadata(dataset.single_space)
    start, _ = raw.get_trajectory_start_idx_and_length(0)
    data, metadata = dataset.get_at_with_metadata(np.arange(start, start + 3))
    raw_data, raw_metadata = raw.get_at_with_metadata(np.arange(start, start + 3))

    obs = data["steps"]["observation"]
    batched = sbu.batch_space(dataset.single_space, 3)
    assert batched["steps"]["observation"].contains(obs)
    assert set(obs) == {"robot", "hand_cam", "external_cam_0", "external_cam_1"}

    # Robot leaves are the upstream robot_state, renamed and converted.
    raw_obs = raw_data["steps"]["observation"]
    arm = obs["robot"]["arm"]
    np.testing.assert_allclose(arm["joint_torque"], raw_obs["motor_torques_measured"])
    np.testing.assert_allclose(arm["joint_motor_torque"], raw_obs["joint_torques_computed"])
    np.testing.assert_allclose(arm["eef_position"], raw_obs["cartesian_position"][:, :3], atol=1e-6)
    assert (arm["eef_quaternion"][:, 0] >= 0).all()
    for i in range(3):
        np.testing.assert_allclose(
            tr3d.quaternions.quat2mat(arm["eef_quaternion"][i].astype(np.float64)),
            tr3d.euler.euler2mat(*raw_obs["cartesian_position"][i, 3:].astype(np.float64)),
            atol=1e-5,
        )
    np.testing.assert_allclose(obs["robot"]["gripper"]["position"][:, 0], raw_obs["gripper_position"])
    np.testing.assert_array_equal(obs["external_cam_0"]["rgb"], raw_obs["exterior_image_1_left"])

    # Serials follow the camera names.
    assert set(metadata["camera_serials"]) == {"hand_cam", "external_cam_0", "external_cam_1"}
    assert list(metadata["camera_serials"]["external_cam_0"]) == list(raw_metadata["camera_serials"]["exterior_1"])
    for name, leaf in obs.items():
        if name != "robot":
            assert dataset.single_space["steps"]["observation"][name]["rgb"].metadata["source"] == name
    md = dataset.single_space["steps"]["observation"]["robot"]["arm"]["eef_quaternion"].metadata
    assert md == conventions.ARM_OBSERVATION_METADATA["eef_quaternion"]


def test_right_streams():
    raw = RawDroidDataset(ROOT, success_only=True, read_right=True)
    dataset = TransformedBatch(raw, DroidRawToEnvTransformation())
    start, _ = raw.get_trajectory_start_idx_and_length(0)
    obs = dataset.get_at(start)["steps"]["observation"]
    raw_poses = raw.get_at(start)["steps"]["observation"]["camera_extrinsics"]
    assert dataset.single_space["steps"]["observation"].contains(obs)
    for name, camera in (("hand_cam", "wrist_camera"), ("external_cam_0", "exterior_camera_1"), ("external_cam_1", "exterior_camera_2")):
        right = obs[f"{name}_right"]
        np.testing.assert_allclose(
            right["extrinsics"], conventions.extrinsics_from_camera_pose(raw_poses[f"{camera}_right"]), atol=1e-5,
        )
        # The two streams of a stereo camera: centres ~12 cm apart (ZED 2 baseline).
        centre = lambda ext: -ext[:, :3].T @ ext[:, 3]
        baseline = np.linalg.norm(centre(right["extrinsics"]) - centre(obs[name]["extrinsics"]))
        assert 0.05 < baseline < 0.2, (name, baseline)
