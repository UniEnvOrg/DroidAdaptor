"""DROID conventions shared by :class:`~unienv_droid.DroidEnv` and the dataset transformation.

The observation schema, its unienv space metadata (``space.metadata``) and the
math that turns an upstream DROID ``robot_state`` / camera pose into it.
:class:`~unienv_droid.DroidEnv` and
:class:`~unienv_droid_data.transformations.DroidRawToEnvTransformation` both
build their observations with :func:`robot_observation` /
:func:`extrinsics_from_camera_pose` and their spaces with
:func:`robot_observation_space` / :func:`camera_observation_space`, so a
recorded frame and a live observation are identical.

Observation schema::

    {
      "robot": {
        "arm": {joint_position, joint_velocity, joint_torque, joint_motor_torque,
                eef_position, eef_quaternion,
                prev_joint_torques_computed, prev_joint_torques_computed_safened,
                prev_controller_latency_ms, prev_command_successful},
        "gripper": {position},
      },
      "<camera name>": {rgb, depth, intrinsics, extrinsics},   # hand_cam, external_cam_<i>
    }

Conventions (upstream ``droid``):

* EEF poses are in the **robot base frame**; upstream reports the flange pose
  as ``cartesian_position = [x, y, z, rx, ry, rz]`` with an extrinsic-XYZ
  euler triple (scipy ``"xyz"`` == transforms3d ``"sxyz"``) in radians. The
  observation reports it as ``eef_position`` and a ``(w, x, y, z)``
  ``eef_quaternion`` with ``w >= 0``.
* ``joint_torque`` is the measured joint torque (upstream
  ``motor_torques_measured``); ``joint_motor_torque`` the torque the
  controller commands (upstream ``joint_torques_computed``).
* Gripper: ``0`` open, ``1`` closed.
* Cameras: ``hand_cam`` (wrist) and ``external_cam_<i>`` (exterior); the
  right stream of a stereo camera is ``<name>_right``. ``extrinsics`` is the
  ``(3, 4)`` base -> camera (OpenCV axes) transform; upstream reports the
  camera pose (camera -> base) as position + extrinsic-XYZ euler.
* DROID runs at 15 Hz (``RobotEnv.control_hz``).
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BinarySpace, BoxSpace, DictSpace
from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    depth_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)

__all__ = [
    "DROID_CONTROL_HZ",
    "TIP_EEF_OFFSET",
    "FR3_JOINT_NAMES",
    "EEF_POSITION_NAMES",
    "EULER_NAMES",
    "QUATERNION_NAMES",
    "ROTATION_6D_NAMES",
    "CARTESIAN_POSITION_NAMES",
    "CARTESIAN_POSITION_UNITS",
    "GRIPPER_LABEL",
    "HAND_CAMERA_NAME",
    "EXTERNAL_CAMERA_PREFIX",
    "RIGHT_STREAM_SUFFIX",
    "ARM_OBSERVATION_METADATA",
    "GRIPPER_OBSERVATION_METADATA",
    "cartesian_position_metadata",
    "cartesian_action_metadata",
    "camera_intrinsics_metadata",
    "camera_leaf_metadata",
    "robot_observation_space",
    "camera_observation_space",
    "default_camera_names",
    "euler_to_quaternion",
    "canonical_quaternion",
    "quaternion_to_rotation_matrix",
    "shift_along_tool_z",
    "extrinsics_from_camera_pose",
    "robot_observation",
]

#: Control / capture rate of DROID (``droid.robot_env.RobotEnv.control_hz``).
DROID_CONTROL_HZ = 15

#: Distance (m) from the FR3 flange origin to the gripper fingertip along the
#: tool (+z) axis: the mujoco_menagerie FR3 + Robotiq 2F-85 CAD model
#: (``g_base`` -> fingertip). ``use_tip_as_eef`` reports / actuates this point.
TIP_EEF_OFFSET = 0.159

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

HAND_CAMERA_NAME = "hand_cam"
EXTERNAL_CAMERA_PREFIX = "external_cam_"
RIGHT_STREAM_SUFFIX = "_right"

_JOINTS = list(FR3_JOINT_NAMES)

#: Metadata of the ``robot.arm`` leaves.
ARM_OBSERVATION_METADATA : Dict[str, Dict[str, Any]] = {
    "joint_position": proprio_metadata(units="rad", names=_JOINTS),
    "joint_velocity": proprio_metadata(units="rad/s", names=_JOINTS),
    "joint_torque": proprio_metadata(units="N*m", names=_JOINTS, label="joint torque"),
    "joint_motor_torque": proprio_metadata(units="N*m", names=_JOINTS, label="commanded actuator torque"),
    "eef_position": proprio_metadata(units="m", names=list(EEF_POSITION_NAMES), frame="base"),
    "eef_quaternion": proprio_metadata(
        units="normalized", names=list(QUATERNION_NAMES), frame="base", rotation="quat_wxyz",
    ),
    "prev_joint_torques_computed": proprio_metadata(
        units="N*m", names=_JOINTS, label="controller torques of the previous command",
    ),
    "prev_joint_torques_computed_safened": proprio_metadata(
        units="N*m", names=_JOINTS, label="previous command's torques after safety clipping",
    ),
    "prev_controller_latency_ms": metadata("state", units="ms", label="controller round-trip latency"),
    "prev_command_successful": metadata("state", label="the previous command succeeded"),
}

#: Metadata of the ``robot.gripper`` leaves.
GRIPPER_OBSERVATION_METADATA : Dict[str, Dict[str, Any]] = {
    "position": proprio_metadata(units="normalized", names=["gripper"], label=GRIPPER_LABEL),
}


def cartesian_position_metadata() -> Dict[str, Any]:
    """Proprio metadata of a ``(6,)`` EEF pose ``[x, y, z, rx, ry, rz]`` (upstream ``cartesian_position``)."""
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


def camera_intrinsics_metadata(source : str, **extra : Any) -> Dict[str, Any]:
    """Metadata of a ``(3, 3)`` OpenCV camera matrix (pixels)."""
    return metadata("intrinsics", source=source, frame="camera_cv", units="px", label=INTRINSICS_LABEL, **extra)


def camera_leaf_metadata(
    name : str,
    fps : float = DROID_CONTROL_HZ,
    serial_number : Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Metadata of the ``rgb`` / ``depth`` / ``intrinsics`` / ``extrinsics`` leaves of a camera.

    ``source`` is the camera name (``hand_cam``, ``external_cam_<i>``, with
    ``_right`` for a right stream) and ``serial_number``, when given, the
    camera serial. ``rgb`` is ``(H, W, 3)`` uint8 RGB, ``depth`` ``(H, W)``
    float32 metres, ``intrinsics`` the OpenCV camera matrix and
    ``extrinsics`` the ``(3, 4)`` base -> camera (OpenCV axes) transform.
    """
    extra = {} if serial_number is None else {"serial_number": str(serial_number)}
    return {
        "rgb": image_metadata(format="rgb", layout="HWC", source=name, fps=fps, **extra),
        "depth": depth_metadata(units="m", layout="HW", source=name, fps=fps, **extra),
        "intrinsics": camera_intrinsics_metadata(name, **extra),
        "extrinsics": metadata("extrinsics", source=name, frame="camera_cv", label=EXTRINSICS_LABEL, **extra),
    }


def _box(shape, low=-np.inf, high=np.inf, metadata=None, dtype=np.float32) -> BoxSpace:
    return BoxSpace(NumpyComputeBackend, low=low, high=high, dtype=dtype, shape=shape, metadata=metadata)


def robot_observation_space() -> DictSpace:
    """The (unbatched) ``robot`` observation space: ``{"arm": {...}, "gripper": {...}}``."""
    arm_md = ARM_OBSERVATION_METADATA
    joints = (len(FR3_JOINT_NAMES),)
    arm = DictSpace(NumpyComputeBackend, {
        "joint_position": _box(joints, metadata=arm_md["joint_position"]),
        "joint_velocity": _box(joints, metadata=arm_md["joint_velocity"]),
        "joint_torque": _box(joints, metadata=arm_md["joint_torque"]),
        "joint_motor_torque": _box(joints, metadata=arm_md["joint_motor_torque"]),
        "eef_position": _box((3,), metadata=arm_md["eef_position"]),
        "eef_quaternion": _box((4,), low=-1.0, high=1.0, metadata=arm_md["eef_quaternion"]),
        "prev_joint_torques_computed": _box(joints, metadata=arm_md["prev_joint_torques_computed"]),
        "prev_joint_torques_computed_safened": _box(joints, metadata=arm_md["prev_joint_torques_computed_safened"]),
        "prev_controller_latency_ms": _box((), low=0.0, metadata=arm_md["prev_controller_latency_ms"]),
        "prev_command_successful": BinarySpace(
            NumpyComputeBackend, shape=(), metadata=arm_md["prev_command_successful"],
        ),
    })
    gripper = DictSpace(NumpyComputeBackend, {
        "position": _box((1,), low=0.0, high=1.0, metadata=GRIPPER_OBSERVATION_METADATA["position"]),
    })
    return DictSpace(NumpyComputeBackend, {"arm": arm, "gripper": gripper})


def camera_observation_space(
    name : str,
    height : int,
    width : int,
    fps : float = DROID_CONTROL_HZ,
    serial_number : Optional[str] = None,
) -> DictSpace:
    """The (unbatched) observation space of one camera stream (see :func:`camera_leaf_metadata`)."""
    md = camera_leaf_metadata(name, fps=fps, serial_number=serial_number)
    return DictSpace(NumpyComputeBackend, {
        "rgb": _box((height, width, 3), low=0, high=255, dtype=np.uint8, metadata=md["rgb"]),
        "depth": _box((height, width), low=0.0, metadata=md["depth"]),
        "intrinsics": _box((3, 3), metadata=md["intrinsics"]),
        "extrinsics": _box((3, 4), metadata=md["extrinsics"]),
    })


def default_camera_names(
    camera_types : Mapping[str, int],
    camera_names : Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """``{serial: camera name}`` of a rig.

    The wrist camera (``camera_type == 0``) is ``hand_cam`` and the exterior
    cameras are ``external_cam_0``, ``external_cam_1``, ... in sorted-serial
    order. ``camera_names`` (``{serial: name}``) overrides any of them
    (serials not on the rig are ignored, so one mapping can describe several
    camera subsets). Raises ``ValueError`` on duplicate or reserved names.
    """
    names : Dict[str, str] = {}
    exterior = sorted(serial for serial, camera_type in camera_types.items() if int(camera_type) != 0)
    wrist = [serial for serial, camera_type in camera_types.items() if int(camera_type) == 0]
    for i, serial in enumerate(wrist):
        names[serial] = HAND_CAMERA_NAME if i == 0 else f"{HAND_CAMERA_NAME}_{i}"
    for i, serial in enumerate(exterior):
        names[serial] = f"{EXTERNAL_CAMERA_PREFIX}{i}"
    names.update({str(serial): str(name) for serial, name in (camera_names or {}).items() if serial in names})
    taken = list(names.values())
    if len(set(taken)) != len(taken):
        raise ValueError(f"duplicate camera names: {names}")
    for name in taken:
        if name == "robot" or name.endswith(RIGHT_STREAM_SUFFIX):
            raise ValueError(f"camera name {name!r} is reserved (the 'robot' key, or the '_right' stream suffix)")
    return names


# ---------------------------------------------------------------------------
# Math (numpy, any leading dimensions)
# ---------------------------------------------------------------------------

def euler_to_quaternion(euler : np.ndarray) -> np.ndarray:
    """``(..., 3)`` extrinsic-XYZ euler (transforms3d ``"sxyz"``) -> ``(..., 4)`` ``(w, x, y, z)``.

    Same values and sign as ``transforms3d.euler.euler2quat(*euler)``.
    """
    euler = np.asarray(euler, dtype=np.float64)
    half = euler / 2.0
    ci, cj, ck = np.cos(half[..., 0]), np.cos(half[..., 1]), np.cos(half[..., 2])
    si, sj, sk = np.sin(half[..., 0]), np.sin(half[..., 1]), np.sin(half[..., 2])
    cc, cs, sc, ss = ci * ck, ci * sk, si * ck, si * sk
    return np.stack([
        cj * cc + sj * ss,
        cj * sc - sj * cs,
        cj * ss + sj * cc,
        cj * cs - sj * sc,
    ], axis=-1)


def canonical_quaternion(quaternion : np.ndarray) -> np.ndarray:
    """``(..., 4)`` ``(w, x, y, z)`` with the sign chosen so that ``w >= 0``."""
    quaternion = np.asarray(quaternion)
    return np.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def quaternion_to_rotation_matrix(quaternion : np.ndarray) -> np.ndarray:
    """``(..., 4)`` unit ``(w, x, y, z)`` -> ``(..., 3, 3)`` rotation matrix."""
    w, x, y, z = (np.asarray(quaternion, dtype=np.float64)[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def shift_along_tool_z(position : np.ndarray, quaternion : np.ndarray, offset : float) -> np.ndarray:
    """``position`` moved ``offset`` metres along the tool +z axis of ``quaternion``."""
    tool_z = quaternion_to_rotation_matrix(quaternion)[..., :, 2]
    return np.asarray(position, dtype=np.float64) + offset * tool_z


def extrinsics_from_camera_pose(pose : np.ndarray) -> np.ndarray:
    """``(..., 6)`` camera pose (camera -> base: position + extrinsic-XYZ euler) -> ``(..., 3, 4)`` base -> camera."""
    pose = np.asarray(pose, dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(euler_to_quaternion(pose[..., 3:6]))
    rotation_t = np.swapaxes(rotation, -1, -2)
    translation = -np.einsum("...ij,...j->...i", rotation_t, pose[..., :3])
    return np.concatenate([rotation_t, translation[..., None]], axis=-1)


def robot_observation(robot_state : Mapping[str, Any], use_tip_as_eef : bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """The ``robot`` observation from an upstream DROID ``robot_state`` (any leading dimensions).

    ``robot_state`` has the upstream keys (``cartesian_position``,
    ``gripper_position``, ``joint_positions``, ``joint_velocities``,
    ``joint_torques_computed``, ``prev_joint_torques_computed``,
    ``prev_joint_torques_computed_safened``, ``motor_torques_measured``,
    ``prev_controller_latency_ms``, ``prev_command_successful``). With
    ``use_tip_as_eef`` the EEF position is the fingertip
    (:data:`TIP_EEF_OFFSET` along the tool z axis) instead of the flange.
    """
    def f32(key):
        return np.asarray(robot_state[key], dtype=np.float32)

    cartesian = np.asarray(robot_state["cartesian_position"], dtype=np.float64)
    quaternion = canonical_quaternion(euler_to_quaternion(cartesian[..., 3:6]))
    position = cartesian[..., :3]
    if use_tip_as_eef:
        position = shift_along_tool_z(position, quaternion, TIP_EEF_OFFSET)
    return {
        "arm": {
            "joint_position": f32("joint_positions"),
            "joint_velocity": f32("joint_velocities"),
            "joint_torque": f32("motor_torques_measured"),
            "joint_motor_torque": f32("joint_torques_computed"),
            "eef_position": position.astype(np.float32),
            "eef_quaternion": quaternion.astype(np.float32),
            "prev_joint_torques_computed": f32("prev_joint_torques_computed"),
            "prev_joint_torques_computed_safened": f32("prev_joint_torques_computed_safened"),
            "prev_controller_latency_ms": f32("prev_controller_latency_ms"),
            "prev_command_successful": np.asarray(robot_state["prev_command_successful"], dtype=bool),
        },
        "gripper": {
            "position": f32("gripper_position")[..., None],
        },
    }
