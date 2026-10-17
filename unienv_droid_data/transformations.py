"""Transformations from the raw DROID dataset to the ``DroidEnv`` observation schema.

:class:`~unienv_droid_data.raw.raw_data.RawDroidDataset` mirrors the DROID
recordings (``trajectory.h5`` + SVO). :class:`DroidRawToEnvTransformation`
rewrites its ``steps.observation`` into the observation of
:class:`unienv_droid.DroidEnv` (see :mod:`unienv_droid.conventions`), with the
same conversion code, so a recorded frame and a live observation match;
:class:`DroidRawToEnvMetadataTransformation` renames the per-frame camera
serials accordingly. Use them through ``unienv_data``'s ``TransformedBatch``::

    from unienv_data.batches import TransformedBatch

    dataset = TransformedBatch(
        RawDroidDataset(root_dir),
        DroidRawToEnvTransformation(),
        metadata_transformation=DroidRawToEnvMetadataTransformation(),
    )

Cameras are named by their DROID position: ``wrist`` -> ``hand_cam``,
``exterior_1`` -> ``external_cam_0``, ``exterior_2`` -> ``external_cam_1``
(``camera_names`` overrides them, keyed by position); right streams
(``read_right=True``) are ``<name>_right``. Every exterior left stream also
carries the dataset-only ``extrinsics_is_postcalibrated`` flag. The other
``steps`` fields (``action_dict``, language instructions, ``is_first`` /
``is_last``) pass through unchanged.

This module does not import ``pyzed`` / ``h5py``.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, DictSpace, Space, TextSpace
from unienv_interface.space.space_utils.metadata_utils import metadata as space_metadata
from unienv_interface.transformations.transformation import DataTransformation

from unienv_droid.conventions import (
    HAND_CAMERA_NAME,
    EXTERNAL_CAMERA_PREFIX,
    RIGHT_STREAM_SUFFIX,
    camera_observation_space,
    extrinsics_from_camera_pose,
    robot_observation,
    robot_observation_space,
)

__all__ = [
    "DEFAULT_CAMERA_NAMES",
    "DroidRawToEnvTransformation",
    "DroidRawToEnvMetadataTransformation",
]

#: ``{DROID camera position: camera name}``.
DEFAULT_CAMERA_NAMES : Dict[str, str] = {
    "wrist": HAND_CAMERA_NAME,
    "exterior_1": f"{EXTERNAL_CAMERA_PREFIX}0",
    "exterior_2": f"{EXTERNAL_CAMERA_PREFIX}1",
}

# Raw observation keys of each camera position: (image, depth, camera parameters).
_RAW_CAMERA_KEYS : Dict[str, Tuple[str, str, str]] = {
    "wrist": ("wrist_image", "wrist_depth", "wrist_camera"),
    "exterior_1": ("exterior_image_1", "exterior_depth_1", "exterior_camera_1"),
    "exterior_2": ("exterior_image_2", "exterior_depth_2", "exterior_camera_2"),
}

# Upstream ``robot_state`` key <- raw observation key (the raw dataset renames one).
_RAW_ROBOT_STATE_KEYS : Dict[str, str] = {
    "cartesian_position": "cartesian_position",
    "gripper_position": "gripper_position",
    "joint_positions": "joint_position",
    "joint_velocities": "joint_velocities",
    "joint_torques_computed": "joint_torques_computed",
    "motor_torques_measured": "motor_torques_measured",
    "prev_joint_torques_computed": "prev_joint_torques_computed",
    "prev_joint_torques_computed_safened": "prev_joint_torques_computed_safened",
    "prev_controller_latency_ms": "prev_controller_latency_ms",
    "prev_command_successful": "prev_command_successful",
}

_POSTCALIBRATED_METADATA = space_metadata(
    "state", label="the extrinsics come from the DROID calibration annotations",
)


def _camera_names(camera_names : Optional[Mapping[str, str]]) -> Dict[str, str]:
    names = dict(DEFAULT_CAMERA_NAMES)
    for position, name in (camera_names or {}).items():
        if position not in names:
            raise ValueError(f"unknown DROID camera position {position!r}; expected one of {sorted(names)}")
        names[position] = str(name)
    taken = list(names.values())
    if len(set(taken)) != len(taken):
        raise ValueError(f"duplicate camera names: {names}")
    for name in taken:
        if name == "robot" or name.endswith(RIGHT_STREAM_SUFFIX):
            raise ValueError(f"camera name {name!r} is reserved (the 'robot' key, or the '_right' stream suffix)")
    return names


class DroidRawToEnvTransformation(DataTransformation):
    """Raw DROID frame -> ``DroidEnv`` observation schema (``steps.observation`` only).

    Args:
        use_tip_as_eef: Report the EEF at the fingertip, as
            ``DroidEnv(use_tip_as_eef=True)`` (default: the flange).
        camera_names: ``{DROID camera position: name}`` overriding
            :data:`DEFAULT_CAMERA_NAMES`.
    """
    has_inverse = False

    def __init__(
        self,
        use_tip_as_eef : bool = False,
        camera_names : Optional[Mapping[str, str]] = None,
    ):
        self.use_tip_as_eef = bool(use_tip_as_eef)
        self.camera_names = _camera_names(camera_names)

    @staticmethod
    def _sides(observation_keys) -> Tuple[str, ...]:
        return ("left", "right") if "wrist_image_right" in observation_keys else ("left",)

    def _stream_name(self, position : str, side : str) -> str:
        name = self.camera_names[position]
        return name if side == "left" else name + RIGHT_STREAM_SUFFIX

    def get_target_space_from_source(self, source_space : Space) -> DictSpace:
        """The target space of a single-frame ``RawDroidDataset`` space."""
        steps = source_space["steps"]
        observation = steps["observation"]
        height, width = observation["wrist_image_left"].shape[-3:-1]
        spaces : Dict[str, Any] = {"robot": robot_observation_space()}
        for side in self._sides(observation.spaces):
            for position in _RAW_CAMERA_KEYS:
                name = self._stream_name(position, side)
                camera = camera_observation_space(name, height, width)
                if position != "wrist" and side == "left":
                    camera = DictSpace(NumpyComputeBackend, {
                        **camera.spaces,
                        "extrinsics_is_postcalibrated": BinarySpace(
                            NumpyComputeBackend, shape=(), metadata=_POSTCALIBRATED_METADATA,
                        ),
                    })
                spaces[name] = camera
        new_steps = DictSpace(
            NumpyComputeBackend,
            {**steps.spaces, "observation": DictSpace(NumpyComputeBackend, spaces)},
            metadata=steps.metadata,
        )
        return DictSpace(
            NumpyComputeBackend, {**source_space.spaces, "steps": new_steps}, metadata=source_space.metadata,
        )

    def transform(self, source_space : Space, data : Dict[str, Any]) -> Dict[str, Any]:
        observation = data["steps"]["observation"]
        robot_state = {key: observation[raw_key] for key, raw_key in _RAW_ROBOT_STATE_KEYS.items()}
        new_observation : Dict[str, Any] = {
            "robot": robot_observation(robot_state, use_tip_as_eef=self.use_tip_as_eef),
        }
        intrinsics, extrinsics = observation["camera_intrinsics"], observation["camera_extrinsics"]
        for side in self._sides(observation.keys()):
            for position, (image_key, depth_key, camera_key) in _RAW_CAMERA_KEYS.items():
                camera = {
                    "rgb": np.asarray(observation[f"{image_key}_{side}"], dtype=np.uint8),
                    "depth": np.asarray(observation[f"{depth_key}_{side}"], dtype=np.float32),
                    "intrinsics": np.asarray(intrinsics[f"{camera_key}_{side}"], dtype=np.float32),
                    "extrinsics": extrinsics_from_camera_pose(extrinsics[f"{camera_key}_{side}"]).astype(np.float32),
                }
                if position != "wrist" and side == "left":
                    camera["extrinsics_is_postcalibrated"] = np.asarray(
                        extrinsics[f"{camera_key}_is_postcalibrated"], dtype=bool,
                    )
                new_observation[self._stream_name(position, side)] = camera
        return {**data, "steps": {**data["steps"], "observation": new_observation}}

    def serialize(self, source_space : Optional[Space] = None) -> Dict[str, Any]:
        return {"use_tip_as_eef": self.use_tip_as_eef, "camera_names": dict(self.camera_names)}

    @classmethod
    def deserialize_from(
        cls, json_data : Dict[str, Any], source_space : Optional[Space] = None,
    ) -> "DroidRawToEnvTransformation":
        return cls(use_tip_as_eef=json_data["use_tip_as_eef"], camera_names=json_data.get("camera_names"))


class DroidRawToEnvMetadataTransformation(DataTransformation):
    """``RawDroidDataset`` per-frame metadata -> camera serials keyed by camera name.

    ``camera_serials`` becomes ``{camera name: serial}`` (``hand_cam``,
    ``external_cam_0``, ``external_cam_1``; a right stream has its camera's
    serial). ``camera_names`` must match the data transformation's.
    """
    has_inverse = False

    def __init__(self, camera_names : Optional[Mapping[str, str]] = None):
        self.camera_names = _camera_names(camera_names)

    def get_target_space_from_source(self, source_space : Space) -> DictSpace:
        serials = DictSpace(NumpyComputeBackend, {
            name: TextSpace(
                NumpyComputeBackend, max_length=64,
                metadata=space_metadata("state", label=f"{name} camera serial"),
            )
            for name in self.camera_names.values()
        })
        return DictSpace(
            NumpyComputeBackend, {**source_space.spaces, "camera_serials": serials}, metadata=source_space.metadata,
        )

    def transform(self, source_space : Space, data : Dict[str, Any]) -> Dict[str, Any]:
        serials = data["camera_serials"]
        return {
            **data,
            "camera_serials": {name: serials[position] for position, name in self.camera_names.items()},
        }

    def serialize(self, source_space : Optional[Space] = None) -> Dict[str, Any]:
        return {"camera_names": dict(self.camera_names)}

    @classmethod
    def deserialize_from(
        cls, json_data : Dict[str, Any], source_space : Optional[Space] = None,
    ) -> "DroidRawToEnvMetadataTransformation":
        return cls(camera_names=json_data.get("camera_names"))
