from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union, Optional, List, Mapping
import numpy as np
import transforms3d as tr3d
from dataclasses import dataclass

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.env_base import Env

from .conventions import (
    DROID_CONTROL_HZ,
    RIGHT_STREAM_SUFFIX,
    TIP_EEF_OFFSET,
    cartesian_action_metadata,
    camera_observation_space,
    default_camera_names,
    extrinsics_from_camera_pose,
    robot_observation,
    robot_observation_space,
)

if TYPE_CHECKING:
    # Only needed for annotations: importing ``droid`` pulls in the camera
    # stack, which is not needed to wrap a compatible stand-in.
    from droid.robot_env import RobotEnv

def get_intrinsics_for_180_rotated_image(
    intrinsics : np.ndarray,
    image_height : int,
    image_width : int
) -> np.ndarray:
    """
    Adjust intrinsics matrix for a 180-degree rotated image.
    When rotating image 180 degrees (equivalent to flipping both X and Y):
    - The principal point changes: cx' = width - 1 - cx, cy' = height - 1 - cy
    - fx and fy remain unchanged (axis flipping is handled by extrinsics)

    This works in conjunction with get_extrinsics_for_180_rotated_image which
    negates the first two rows of the extrinsics matrix.

    Note: If cx ≈ width/2 and cy ≈ height/2, the principal point adjustment is minimal.
    """
    new_intrinsics = intrinsics.copy()
    # cx becomes (image_width - 1 - cx)
    new_intrinsics[0, 2] = image_width - 1 - intrinsics[0, 2]
    # cy becomes (image_height - 1 - cy)
    new_intrinsics[1, 2] = image_height - 1 - intrinsics[1, 2]
    return new_intrinsics

def get_extrinsics_for_180_rotated_image(
    extrinsics : np.ndarray
) -> np.ndarray:
    """
    Adjust extrinsics matrix (3x4, world -> camera) for a 180-degree rotated image.

    Rotating the image 180 degrees is equivalent to rotating the camera coordinate
    system 180 degrees around the optical axis (Z). This negates both X_cam and Y_cam,
    which is achieved by negating the first two rows of the extrinsics matrix.

    This works in conjunction with get_intrinsics_for_180_rotated_image which
    adjusts cx and cy accordingly.
    """
    new_extrinsics = extrinsics.copy()
    # Negate the first row (X-axis) and second row (Y-axis) of the extrinsics
    new_extrinsics[0, :] = -new_extrinsics[0, :]
    new_extrinsics[1, :] = -new_extrinsics[1, :]
    return new_extrinsics

@dataclass(frozen=True)
class DroidCamera:
    """One camera stream exposed by :class:`DroidEnv`.

    Attributes:
        name: Observation key of the stream: ``hand_cam`` / ``external_cam_<i>``
            (or the ``camera_names`` override) for a mono (or left stereo)
            stream, and ``<name>_right`` for the right stream of a stereo
            camera.
        source_key: Key of the stream in the upstream ``RobotEnv`` image /
            depth / intrinsics dicts (e.g. ``"<serial>_left"``).
        serial: Camera serial (the key of upstream ``camera_type``).
        camera_type: Upstream camera type (``0`` = hand / wrist, otherwise
            exterior).
    """
    name : str
    source_key : str
    serial : str
    camera_type : int

    @property
    def is_ext(self) -> bool:
        return self.camera_type != 0

    @property
    def is_right(self) -> bool:
        return self.source_key.endswith("_right")

__all__ = ['DroidEnv', 'DroidCamera']

class DroidEnv(Env[
    NumpyArrayType,
    None,
    Dict[str, Any],
    NumpyArrayType,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
]):
    """A UniEnv ``Env`` over an upstream DROID ``RobotEnv``.

    Observations (see :mod:`unienv_droid.conventions`)::

        {"robot": {"arm": {...}, "gripper": {"position"}},
         "hand_cam": {...}, "external_cam_0": {...}, ...}

    Actions are the flat upstream ``[pos (3), rotation (3 euler | 6 rows),
    gripper]`` vector of the ``RobotEnv``'s ``action_space``.
    """

    metadata : Dict[str, Any] = {
        "render_modes": ["rgb_array"],
    }

    render_mode = "rgb_array"
    render_fps = DROID_CONTROL_HZ

    backend = NumpyComputeBackend
    device = None

    batch_size = None
    context_space = None

    #: Factor converting upstream depth in each supported unit to metres.
    DEPTH_UNIT_SCALES = {"mm": 1e-3, "m": 1.0}

    SUPPORTED_ACTION_SPACES = ("cartesian_position", "cartesian_velocity")

    # Distance (m) from the FR3 flange origin to the gripper fingertip along
    # the tool (+z) axis (``conventions.TIP_EEF_OFFSET``, the mujoco_menagerie
    # FR3 + Robotiq 2F-85 CAD value), so sim and real share one EEF convention.
    TIP_EEF_OFFSET = TIP_EEF_OFFSET

    @staticmethod
    def action_bounds(
        action_space : str,
        use_6d_rotation_in_action : bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` of the action Box for an upstream ``action_space``.

        ``cartesian_velocity`` actions are normalized velocities, which
        upstream ``RobotEnv.step`` asserts to lie in ``[-1, 1]``.
        ``cartesian_position`` actions are an absolute pose that upstream does
        not normalize: position in metres (``[-1, 1]``, the bounds the DROID
        dataset declares for ``cartesian_position``), euler angles in radians
        (``[-pi, pi]``) or 6D rotation rows (``[-1, 1]``), and a gripper
        position, which upstream clips to ``[0, 1]`` (``0`` open, ``1`` closed).
        """
        n_rot = 6 if use_6d_rotation_in_action else 3
        if action_space == "cartesian_velocity":
            low = np.full(3 + n_rot + 1, -1.0)
            high = np.full(3 + n_rot + 1, 1.0)
        else:
            rot_bound = 1.0 if use_6d_rotation_in_action else np.pi
            low = np.concatenate([[-1.0] * 3, [-rot_bound] * n_rot, [0.0]])
            high = np.concatenate([[1.0] * 3, [rot_bound] * n_rot, [1.0]])
        return low.astype(np.float32), high.astype(np.float32)

    @classmethod
    def build_action_space(
        cls,
        action_space : str,
        use_6d_rotation_in_action : bool = False
    ) -> BoxSpace:
        """The action space for an upstream ``action_space`` (see :meth:`action_bounds`).

        Its metadata (``conventions.cartesian_action_metadata``) names every
        element and gives the mode, units, frame and rotation representation.
        """
        low, high = cls.action_bounds(action_space, use_6d_rotation_in_action)
        return BoxSpace(
            NumpyComputeBackend,
            low,
            high,
            dtype=np.float32,
            shape=low.shape,
            metadata=cartesian_action_metadata(action_space, use_6d_rotation_in_action)
        )

    def __init__(
        self,
        robotenv : RobotEnv,
        use_6d_rotation_in_action : bool = False,
        use_tip_as_eef : bool = False,
        flip_ext_cams_y : Union[bool, Mapping[str, bool]] = False,
        include_right_images : bool = False,
        depth_units : str = "mm",
        camera_names : Optional[Mapping[str, str]] = None,
    ):
        """
        Args:
            robotenv: The upstream ``droid.robot_env.RobotEnv`` (or a compatible
                stand-in exposing ``action_space``, ``reset``, ``step`` and
                ``get_observation``).
            use_6d_rotation_in_action: ``(10,)`` actions with a 6D rotation
                (the first two rows of the rotation matrix) instead of ``(7,)``
                actions with euler angles.
            use_tip_as_eef: Report / actuate the EEF at the fingertip
                (``TIP_EEF_OFFSET`` along tool-z) instead of the flange.
            flip_ext_cams_y: Rotate camera outputs by 180 degrees. ``True``
                flips every **exterior** camera (the wrist camera is never
                flipped by the bool form); a mapping selects cameras by
                camera name (``external_cam_0``, ...) or serial.
            include_right_images: Also expose the right stream of stereo
                cameras, as ``<name>_right``. By default only the left stream
                is exposed, as ``<name>``.
            depth_units: Units of the depth maps the upstream ``RobotEnv``
                returns; ``DroidEnv`` always reports depth in **metres**.
                ``"mm"`` (default) matches upstream DROID's ZED reader, which
                opens the camera without setting ``coordinate_units`` (ZED
                default: millimetres) — its depth retrieval is commented out
                upstream, so forks that re-enable it get millimetres. Pass
                ``"m"`` for a stack that already returns metres (e.g. a reader
                opened with ``coordinate_units = sl.UNIT.METER``).
            camera_names: ``{serial: name}`` overriding the camera names. By
                default the wrist camera is ``hand_cam`` and the exterior
                cameras are ``external_cam_0``, ``external_cam_1``, ... in
                sorted-serial order (``conventions.default_camera_names``).
        """
        if depth_units not in self.DEPTH_UNIT_SCALES:
            raise ValueError(
                f"depth_units must be one of {tuple(self.DEPTH_UNIT_SCALES)}, got {depth_units!r}"
            )
        if robotenv.action_space not in self.SUPPORTED_ACTION_SPACES:
            raise ValueError(
                f"DroidEnv supports a RobotEnv with action_space in {self.SUPPORTED_ACTION_SPACES}, "
                f"got {robotenv.action_space!r}"
            )
        self.action_space = self.build_action_space(
            robotenv.action_space,
            use_6d_rotation_in_action
        )
        self.use_6d_rotation_in_action = use_6d_rotation_in_action
        self.use_tip_as_eef = use_tip_as_eef
        self.flip_ext_cams_y = flip_ext_cams_y
        self.include_right_images = include_right_images
        self.depth_units = depth_units
        self.robotenv = robotenv

        robotenv_obs = robotenv.get_observation()
        self.cameras : List[DroidCamera] = self.discover_cameras(
            robotenv_obs,
            include_right_images=include_right_images,
            camera_names=camera_names,
        )

        spaces : Dict[str, Any] = {"robot": robot_observation_space()}
        for cam in self.cameras:
            height, width = robotenv_obs['image'][cam.source_key].shape[:2]
            spaces[cam.name] = camera_observation_space(
                cam.name, height, width, fps=self.render_fps, serial_number=cam.serial,
            )
        self.observation_space = DictSpace(NumpyComputeBackend, spaces)

        self.rng = np.random.default_rng()
        self._render_frame = None

    @staticmethod
    def discover_cameras(
        robotenv_obs : Dict[str, Any],
        include_right_images : bool = False,
        camera_names : Optional[Mapping[str, str]] = None,
    ) -> List[DroidCamera]:
        """
        List the camera streams of an upstream observation.

        Real ZED cameras report ``<serial>_left`` / ``<serial>_right`` streams,
        mono cameras (e.g. a simulator) report ``<serial>``. Each camera is
        named by ``conventions.default_camera_names`` (``hand_cam``,
        ``external_cam_<i>``, or the ``camera_names`` override); an included
        right stream is ``<name>_right``.
        """
        camera_types = {
            serial: int(camera_type)
            for serial, camera_type in robotenv_obs['camera_type'].items()
            if len(serial) > 0
        }
        names = default_camera_names(camera_types, camera_names)
        cameras = []
        for serial, camera_type in camera_types.items():
            name = names[serial]
            if serial in robotenv_obs['image']:
                cameras.append(DroidCamera(name, serial, serial, camera_type))
            elif f"{serial}_left" in robotenv_obs['image']:
                cameras.append(DroidCamera(name, f"{serial}_left", serial, camera_type))
            if include_right_images and f"{serial}_right" in robotenv_obs['image']:
                cameras.append(DroidCamera(name + RIGHT_STREAM_SUFFIX, f"{serial}_right", serial, camera_type))
        return cameras

    @property
    def camera_names(self) -> Tuple[str, ...]:
        """Observation keys of the exposed camera streams."""
        return tuple(cam.name for cam in self.cameras)

    @property
    def camera_serials(self) -> Dict[str, str]:
        """``{camera name: serial}`` of the exposed camera streams."""
        return {cam.name: cam.serial for cam in self.cameras}

    def _should_flip(self, cam : DroidCamera) -> bool:
        if isinstance(self.flip_ext_cams_y, Mapping):
            return bool(
                self.flip_ext_cams_y.get(cam.name, self.flip_ext_cams_y.get(cam.serial, False))
            )
        return bool(self.flip_ext_cams_y) and cam.is_ext

    def extract_obs(
        self,
        robotenv_obs : Dict[str, Any]
    ) -> Dict[str, Any]:
        """The observation of an upstream ``RobotEnv.get_observation()`` output."""
        ret : Dict[str, Any] = {
            "robot": robot_observation(robotenv_obs['robot_state'], use_tip_as_eef=self.use_tip_as_eef),
        }
        for cam in self.cameras:
            rgb : np.ndarray = robotenv_obs['image'][cam.source_key]
            rgb = rgb[..., [2,1,0]].astype(np.uint8) # For ZED we get BGRA images, convert to RGB and drop alpha if present
            depth : np.ndarray = robotenv_obs['depth'][cam.source_key]
            depth = depth.astype(np.float32)
            depth_scale = self.DEPTH_UNIT_SCALES[self.depth_units]
            if depth_scale != 1.0:
                depth = depth * np.float32(depth_scale) # -> metres
            intrinsics = np.asarray(robotenv_obs['camera_intrinsics'][cam.source_key], dtype=np.float32)
            camera_extrinsics = robotenv_obs['camera_extrinsics']
            camera_pose = camera_extrinsics[cam.source_key] if cam.source_key in camera_extrinsics else camera_extrinsics[cam.serial]
            extrinsics = extrinsics_from_camera_pose(camera_pose)
            if self._should_flip(cam):
                rgb = rgb[::-1, ::-1]
                depth = depth[::-1, ::-1]
                intrinsics = get_intrinsics_for_180_rotated_image(
                    intrinsics,
                    rgb.shape[0],
                    rgb.shape[1]
                )
                extrinsics = get_extrinsics_for_180_rotated_image(
                    extrinsics
                )
            ret[cam.name] = {
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "depth": np.ascontiguousarray(depth, dtype=np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "extrinsics": extrinsics.astype(np.float32),
            }
        return ret

    def _update_render_frame(self, obs : Dict[str, Any]) -> None:
        self._render_frame = [
            obs[cam.name]['rgb'] for cam in self.cameras if not cam.is_right
        ]

    def reset(
        self,
        mask : Optional[NumpyArrayType] = None,
        seed : Optional[int] = None
    ) -> Tuple[None, Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.robotenv.reset()
        obs = self.extract_obs(self.robotenv.get_observation())

        self._update_render_frame(obs)
        return None, obs, {}

    def step(self, action : np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.use_6d_rotation_in_action:
            action_euler = self.rotation_6d_to_euler(action[3:9])
            if self.robotenv.action_space == 'cartesian_velocity':
                action_euler /= np.pi
            action_gripper = action[9:]
        else:
            action_euler = action[3:6]
            action_gripper = action[6:]

        action_pos = action[:3]
        action = np.concatenate([
            action_pos,
            action_euler,
            action_gripper
        ])

        if self.use_tip_as_eef and self.robotenv.action_space == 'cartesian_position':
            action = self.transform_tip_action_to_droid(action)

        self.robotenv.step(action)
        obs = self.extract_obs(self.robotenv.get_observation())

        self._update_render_frame(obs)
        return obs, 0, False, False, {}

    def render(self):
        return self._render_frame

    @staticmethod
    def rotation_6d_to_euler(
        rotation_6d: np.ndarray
    ) -> np.ndarray:
        # float64 throughout: transforms3d's ``mat2euler`` calls
        # ``np.array(..., copy=False)``, which NumPy 2 rejects when a dtype
        # conversion (e.g. from float32) would be needed.
        rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
        vec1 = rotation_6d[:3]
        vec2 = rotation_6d[3:]
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 - np.dot(vec1, vec2) * vec1
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = np.cross(vec1, vec2)
        rotation_matrix = np.stack([vec1, vec2, vec3], axis=0)
        return np.asarray(tr3d.euler.mat2euler(rotation_matrix), dtype=np.float32)

    @staticmethod
    def transform_tip_action_to_droid(
        action: np.ndarray
    ) -> np.ndarray:
        """An absolute ``[tip pos, euler, gripper]`` action -> the flange action upstream expects."""
        eef_euler = action[3:6]
        eef_rotation_matrix = tr3d.euler.euler2mat(*eef_euler)
        return np.concatenate([
            action[:3] - eef_rotation_matrix[:, 2] * DroidEnv.TIP_EEF_OFFSET,
            action[3:]
        ])
