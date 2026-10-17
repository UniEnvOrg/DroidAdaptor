from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union, Optional, List, Callable, Mapping
import numpy as np
import transforms3d as tr3d
import copy
from dataclasses import dataclass

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import BoxSpace, DictSpace, BinarySpace
from unienv_interface.env_base import Env

if TYPE_CHECKING:
    # Only needed for annotations: importing ``droid`` pulls in the camera
    # stack, which is not needed to wrap a compatible stand-in.
    from droid.robot_env import RobotEnv

def extrinsic_from_pose(
    pose : np.ndarray
) -> np.ndarray:
    """
    Get opencv extrinsics matrix (3x4) from pose (7,) that contains position and euler angles
    """
    pos = pose[:3]
    euler = pose[3:]
    rotation_matrix = tr3d.euler.euler2mat(*euler)
    affine_matrix = tr3d.affines.compose(pos, rotation_matrix, np.ones(3))
    inverse_affine_matrix = np.linalg.inv(affine_matrix)
    return inverse_affine_matrix[:3]

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
        name: Observation name; the stream is emitted under ``cam_<name>``.
            The bare camera serial for a mono (or left stereo) stream, and
            ``<serial>_right`` for the right stream of a stereo camera.
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
    # https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py#L158
    # https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/protos/polymetis.proto#L120
    ROBOT_STATE_OBSERVATION_SPACE = DictSpace(
        NumpyComputeBackend,
        {
            "cartesian_position": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(6,)
            ), # end-effector position (x, y, z) and orientation (roll, pitch, yaw)
            "gripper_position": BoxSpace(
                NumpyComputeBackend,
                low=0,
                high=1,
                dtype=np.float32,
                shape=()
            ),
            "joint_positions": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "joint_velocities": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "joint_torques_computed": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "prev_joint_torques_computed": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "prev_joint_torques_computed_safened": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "motor_torques_measured": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(7,)
            ),
            "prev_controller_latency_ms": BoxSpace(
                NumpyComputeBackend,
                low=0,
                high=np.inf,
                dtype=np.float32,
                shape=()
            ),
            "prev_command_successful": BinarySpace(
                NumpyComputeBackend,
                shape=()
            )
        }
    )
    ADDITIONAL_STATE_OBSERVATION_SPACE = DictSpace(
        NumpyComputeBackend,
        {
            "end_effector_pos": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(3,)
            ), # end-effector position (x, y, z)
            "end_effector_euler": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf,
                high=np.inf,
                dtype=np.float32,
                shape=(3,)
            ), # end-effector orientation (roll, pitch, yaw)
            "end_effector_quaternion": BoxSpace(
                NumpyComputeBackend,
                low=-1,
                high=1,
                dtype=np.float32,
                shape=(4,)
            ),
            "end_effector_6d_rotation": BoxSpace(
                NumpyComputeBackend,
                low=-1,
                high=1,
                dtype=np.float32,
                shape=(6,)
            ),
        }
    )

    metadata : Dict[str, Any] = {
        "render_modes": ["rgb_array"],
    }

    render_mode = "rgb_array"
    render_fps = 15

    backend = NumpyComputeBackend
    device = None

    batch_size = None
    context_space = None
    
    #: Factor converting upstream depth in each supported unit to metres.
    DEPTH_UNIT_SCALES = {"mm": 1e-3, "m": 1.0}

    SUPPORTED_ACTION_SPACES = ("cartesian_position", "cartesian_velocity")

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
        """The action space for an upstream ``action_space`` (see :meth:`action_bounds`)."""
        low, high = cls.action_bounds(action_space, use_6d_rotation_in_action)
        return BoxSpace(
            NumpyComputeBackend,
            low,
            high,
            dtype=np.float32,
            shape=low.shape
        )

    def __init__(
        self,
        robotenv : RobotEnv,
        use_6d_rotation_in_action : bool = False,
        use_tip_as_eef : bool = False,
        flip_ext_cams_y : Union[bool, Dict[str, bool]] = False,
        include_right_images : bool = False,
        depth_units : str = "mm",
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
                observation name (``<serial>``) or upstream key
                (``<serial>_left``).
            include_right_images: Also expose the right stream of stereo
                cameras, as ``cam_<serial>_right``. By default only the left
                stream is exposed, as ``cam_<serial>``.
            depth_units: Units of the depth maps the upstream ``RobotEnv``
                returns; ``DroidEnv`` always reports depth in **metres**.
                ``"mm"`` (default) matches upstream DROID's ZED reader, which
                opens the camera without setting ``coordinate_units`` (ZED
                default: millimetres) — its depth retrieval is commented out
                upstream, so forks that re-enable it get millimetres. Pass
                ``"m"`` for a stack that already returns metres (e.g. a reader
                opened with ``coordinate_units = sl.UNIT.METER``).
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
            include_right_images=include_right_images
        )
        # Kept for backwards compatibility: observation names and exterior flags.
        self.all_cam_names = [cam.name for cam in self.cameras]
        self.all_cam_is_ext = [cam.is_ext for cam in self.cameras]

        self.observation_space = copy.copy(self.ROBOT_STATE_OBSERVATION_SPACE)
        self.observation_space.spaces.update(self.ADDITIONAL_STATE_OBSERVATION_SPACE.spaces)
        for cam in self.cameras:
            cam_size = tuple(robotenv_obs['image'][cam.source_key].shape[:2])
            self.observation_space['cam_' + cam.name] = DictSpace(
                NumpyComputeBackend,
                {
                    "rgb": BoxSpace(
                        NumpyComputeBackend,
                        0,
                        255,
                        dtype=np.uint8,
                        shape=(*cam_size, 3)
                    ),
                    "depth": BoxSpace(
                        NumpyComputeBackend,
                        0,
                        np.inf,
                        dtype=np.float32,
                        shape=cam_size
                    ),
                    "intrinsics": BoxSpace(
                        NumpyComputeBackend,
                        low=-np.inf,
                        high=np.inf,
                        dtype=np.float32,
                        shape=(3, 3)
                    ),
                    "extrinsics": BoxSpace(
                        NumpyComputeBackend,
                        low=-np.inf,
                        high=np.inf,
                        dtype=np.float32,
                        shape=(3, 4)
                    ),
                }
            )

        self.rng = np.random.default_rng()
        self._render_frame = None

    @staticmethod
    def discover_cameras(
        robotenv_obs : Dict[str, Any],
        include_right_images : bool = False,
    ) -> List[DroidCamera]:
        """
        List the camera streams of an upstream observation.

        Real ZED cameras report ``<serial>_left`` / ``<serial>_right`` streams,
        mono cameras (e.g. a simulator) report ``<serial>``. Every stream is
        named after its camera serial so the observation key is ``cam_<serial>``
        on both, with ``cam_<serial>_right`` for an included right stream.
        """
        cameras = []
        for serial, camera_type in robotenv_obs['camera_type'].items():
            if len(serial) == 0:
                continue
            camera_type = int(camera_type)
            if serial in robotenv_obs['image']:
                cameras.append(DroidCamera(serial, serial, serial, camera_type))
            elif f"{serial}_left" in robotenv_obs['image']:
                cameras.append(DroidCamera(serial, f"{serial}_left", serial, camera_type))
            if include_right_images and f"{serial}_right" in robotenv_obs['image']:
                cameras.append(DroidCamera(f"{serial}_right", f"{serial}_right", serial, camera_type))
        return cameras

    @property
    def camera_serials(self) -> Tuple[str, ...]:
        """Observation names of the exposed cameras (``cam_<name>`` keys)."""
        return tuple(cam.name for cam in self.cameras)

    @property
    def camera_types(self) -> Dict[str, int]:
        """``{camera name: camera_type}`` (``0`` = wrist, ``1`` = exterior)."""
        return {cam.name: cam.camera_type for cam in self.cameras}

    def _should_flip(self, cam : DroidCamera) -> bool:
        if isinstance(self.flip_ext_cams_y, Mapping):
            return bool(
                self.flip_ext_cams_y.get(cam.name, self.flip_ext_cams_y.get(cam.source_key, False))
            )
        return bool(self.flip_ext_cams_y) and cam.is_ext

    def extract_obs(
        self,
        robotenv_obs : Dict[str, Any]
    ):
        robot_state = robotenv_obs['robot_state']
        ret = {
            key: np.asarray(robot_state[key], dtype=space.dtype)
            for key, space in self.ROBOT_STATE_OBSERVATION_SPACE.spaces.items()
        }

        eef_euler = ret['cartesian_position'][3:]
        eef_quaternion = tr3d.euler.euler2quat(*eef_euler)
        eef_rotation_matrix = tr3d.quaternions.quat2mat(eef_quaternion)

        ret['end_effector_euler'] = eef_euler
        ret['end_effector_quaternion'] = np.asarray(eef_quaternion, dtype=np.float32)
        ret['end_effector_6d_rotation'] = np.asarray(eef_rotation_matrix[:2, :].flatten(), dtype=np.float32)

        eef_pos = ret['cartesian_position'][:3]
        if self.use_tip_as_eef:
            eef_pos = self.transform_droid_eef_location_to_tip(
                eef_pos,
                eef_rotation_matrix
            )
        eef_pos = np.asarray(eef_pos, dtype=np.float32)
        ret['end_effector_pos'] = eef_pos
        ret['cartesian_position'] = np.concatenate([eef_pos, eef_euler]).astype(np.float32)

        intrinsics = extrinsics = None
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
            extrinsics = extrinsic_from_pose(np.asarray(camera_pose, dtype=np.float64))
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
            intrinsics = intrinsics.astype(np.float32)
            extrinsics = extrinsics.astype(np.float32)
            ret['cam_' + cam.name] = {
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "depth": np.ascontiguousarray(depth, dtype=np.float32),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics
            }

        return ret, intrinsics, extrinsics

    def _update_render_frame(self, obs : Dict[str, Any]) -> None:
        self._render_frame = [
            obs['cam_' + cam.name]['rgb'] for cam in self.cameras if not cam.is_right
        ]

    def reset(
        self,
        mask : Optional[NumpyArrayType] = None,
        seed : Optional[int] = None
    ) -> Tuple[None, Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.robotenv.reset()
        robotenv_obs = self.robotenv.get_observation()
        obs, intrinsics, extrinsics = self.extract_obs(robotenv_obs)

        self._update_render_frame(obs)
        return None, obs, {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }

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
        robotenv_obs = self.robotenv.get_observation()
        obs, intrinsics, extrinsics = self.extract_obs(robotenv_obs)

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

    # Distance (m) from the FR3 flange origin to the gripper fingertip along
    # the tool (+z) axis. 0.159 m matches the mujoco_menagerie
    # franka_fr3_robotiq_2f85 CAD model (g_base -> fingertip), so sim and real
    # share one EEF convention. (DROID's historical measured value was 0.152 m.)
    TIP_EEF_OFFSET = 0.159

    @staticmethod
    def transform_tip_action_to_droid(
        action: np.ndarray
    ) -> np.ndarray:
        eef_euler = action[3:6]
        eef_rotation_matrix = tr3d.euler.euler2mat(*eef_euler)
        return np.concatenate([
            action[:3] - eef_rotation_matrix[:, 2] * DroidEnv.TIP_EEF_OFFSET,
            action[3:]
        ])

    @staticmethod
    def transform_droid_eef_location_to_tip(
        action: np.ndarray,
        eef_rotation_matrix: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([
            action[:3] + eef_rotation_matrix[:, 2] * DroidEnv.TIP_EEF_OFFSET,
            action[3:]
        ])

