"""DROID conventions shared by :class:`~unienv_droid.DroidEnv` and the raw dataset.

Element names and the unienv space metadata (``space.metadata``) that describe
what the DROID robot-state, camera and action leaves mean. Both
:mod:`unienv_droid.env_wrapper` and :mod:`unienv_droid_data.raw` build their
spaces from these, so the two describe shared quantities identically.

Conventions (upstream ``droid``):

* EEF poses and actions are in the **robot base frame**; the euler triple is
  extrinsic XYZ (scipy ``"xyz"`` == transforms3d ``"sxyz"``) in radians.
* The quaternion is ``(w, x, y, z)``; the 6D rotation is the first two **rows**
  of the rotation matrix.
* Gripper: ``0`` open, ``1`` closed.
* DROID runs at 15 Hz (``RobotEnv.control_hz``).
"""

from typing import Any, Dict, Tuple

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    depth_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)

__all__ = [
    "DROID_CONTROL_HZ",
    "FR3_JOINT_NAMES",
    "EEF_POSITION_NAMES",
    "EULER_NAMES",
    "QUATERNION_NAMES",
    "ROTATION_6D_NAMES",
    "CARTESIAN_POSITION_NAMES",
    "CARTESIAN_POSITION_UNITS",
    "GRIPPER_LABEL",
    "cartesian_position_metadata",
    "cartesian_action_metadata",
    "camera_intrinsics_metadata",
    "camera_leaf_metadata",
]

#: Control / capture rate of DROID (``droid.robot_env.RobotEnv.control_hz``).
DROID_CONTROL_HZ = 15

FR3_JOINT_NAMES : Tuple[str, ...] = tuple(f"fr3_joint{i}" for i in range(1, 8))
EEF_POSITION_NAMES : Tuple[str, ...] = ("x", "y", "z")
EULER_NAMES : Tuple[str, ...] = ("rx", "ry", "rz")
QUATERNION_NAMES : Tuple[str, ...] = ("qw", "qx", "qy", "qz")
ROTATION_6D_NAMES : Tuple[str, ...] = ("r00", "r01", "r02", "r10", "r11", "r12")

CARTESIAN_POSITION_NAMES : Tuple[str, ...] = EEF_POSITION_NAMES + EULER_NAMES
CARTESIAN_POSITION_UNITS : Tuple[str, ...] = ("m",) * 3 + ("rad",) * 3

GRIPPER_LABEL = "gripper position (0 = open, 1 = closed)"
INTRINSICS_LABEL = "OpenCV camera matrix K"
EXTRINSICS_LABEL = "base -> camera (OpenCV) transform [R | t]"


def cartesian_position_metadata() -> Dict[str, Any]:
    """Proprio metadata of a ``(6,)`` EEF pose ``[x, y, z, rx, ry, rz]``."""
    return proprio_metadata(
        units=list(CARTESIAN_POSITION_UNITS),
        names=list(CARTESIAN_POSITION_NAMES),
        frame="base",
        rotation="euler_sxyz",
    )


def cartesian_action_metadata(
    action_space : str,
    use_6d_rotation_in_action : bool = False,
) -> Dict[str, Any]:
    """Metadata of a ``[pos (3), rotation (3 euler | 6 rows), gripper]`` action.

    ``cartesian_position``: an absolute EEF pose (metres, radians or 6D rows)
    and an absolute gripper position. ``cartesian_velocity``: normalized
    (``[-1, 1]``) velocities; their physical scale is set by the robot side.
    """
    rotation_names = ROTATION_6D_NAMES if use_6d_rotation_in_action else EULER_NAMES
    names = [*EEF_POSITION_NAMES, *rotation_names, "gripper"]
    rotation = "rot6d" if use_6d_rotation_in_action else "euler_sxyz"
    if action_space == "cartesian_velocity":
        return action_metadata(
            control="eef_velocity",
            mode="velocity",
            units="normalized",
            normalized=True,
            names=names,
            frame="base",
            rotation=rotation,
        )
    if action_space == "cartesian_position":
        rotation_unit = "normalized" if use_6d_rotation_in_action else "rad"
        return action_metadata(
            control="eef_pose",
            mode="absolute",
            units=["m"] * 3 + [rotation_unit] * len(rotation_names) + ["normalized"],
            names=names,
            frame="base",
            rotation=rotation,
        )
    raise ValueError(f"unsupported cartesian action space {action_space!r}")


def camera_intrinsics_metadata(source : str) -> Dict[str, Any]:
    """Metadata of a ``(3, 3)`` OpenCV camera matrix (pixels)."""
    return metadata("intrinsics", source=source, frame="camera_cv", units="px", label=INTRINSICS_LABEL)


def camera_leaf_metadata(source : str, fps : float = DROID_CONTROL_HZ) -> Dict[str, Dict[str, Any]]:
    """Metadata of the ``rgb`` / ``depth`` / ``intrinsics`` / ``extrinsics`` leaves of a camera.

    ``rgb`` is ``(H, W, 3)`` uint8 RGB, ``depth`` ``(H, W)`` float32 metres,
    ``intrinsics`` the OpenCV camera matrix and ``extrinsics`` the ``(3, 4)``
    base (world) -> camera (OpenCV axes) transform.
    """
    return {
        "rgb": image_metadata(format="rgb", layout="HWC", source=source, fps=fps),
        "depth": depth_metadata(units="m", layout="HW", source=source, fps=fps),
        "intrinsics": camera_intrinsics_metadata(source),
        "extrinsics": metadata("extrinsics", source=source, frame="camera_cv", label=EXTRINSICS_LABEL),
    }
