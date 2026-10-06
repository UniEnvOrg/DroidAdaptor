from typing import Any, Dict, Tuple, Union, Optional, List, Callable, Mapping, Sequence
import numpy as np
import transforms3d as tr3d
import torch
import copy
from dataclasses import dataclass

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import BoxSpace, DictSpace, BinarySpace
from unienv_interface.env_base import Env

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

def _map_transform_to_array(data):
    if isinstance(data, Mapping):
        ret = {}
        for key, value in data.items():
            ret[key] = _map_transform_to_array(value)
        return ret
    elif isinstance(data, Sequence) and not isinstance(data, str):
        ret_array = np.asarray(data)
        if ret_array.dtype == np.float64:
            ret_array = ret_array.astype(np.float32)
        return ret_array
    elif isinstance(data, int):
        return np.array(data, dtype=np.int32)
    elif isinstance(data, float):
        return np.array(data, dtype=np.float32)
    elif isinstance(data, bool):
        return np.array(data, dtype=np.bool_)
    else:
        return data

__all__ = ['DroidEnv']

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
            "prev_joint_torques_applied_safened": BoxSpace(
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
    
    def __init__(
        self,
        robotenv : RobotEnv,
        use_6d_rotation_in_action : bool = False,
        use_tip_as_eef : bool = False,
        flip_ext_cams_y : Union[bool, Dict[str, bool]] = False,
    ):
        if use_6d_rotation_in_action:
            self.action_space = BoxSpace(
                NumpyComputeBackend,
                -1.0,
                1.0,
                dtype=np.float32,
                shape=(10,)
            )
        else:
            self.action_space = BoxSpace(
                NumpyComputeBackend,
                -1.0,
                1.0,
                dtype=np.float32,
                shape=(7,)
            )
        self.use_6d_rotation_in_action = use_6d_rotation_in_action
        self.use_tip_as_eef = use_tip_as_eef
        self.flip_ext_cams_y = flip_ext_cams_y
        self.robotenv = robotenv

        robotenv_obs = robotenv.get_observation()
        self.all_cam_names = []
        self.all_cam_is_ext = []
        for serial, is_ext in robotenv_obs['camera_type'].items():
            is_ext = is_ext != 0
            if len(serial) == 0:
                continue
            if f"{serial}_left" in robotenv_obs['image']:
                self.all_cam_names.append(f"{serial}_left")
                self.all_cam_is_ext.append(is_ext)
            if f"{serial}_right" in robotenv_obs['image']:
                self.all_cam_names.append(f"{serial}_right")
                self.all_cam_is_ext.append(is_ext)
            if serial in robotenv_obs['image']:
                self.all_cam_names.append(serial)
                self.all_cam_is_ext.append(is_ext)
        
        def _get_cam_size(cam_name):
            return tuple(robotenv_obs['image'][cam_name].shape[:2])
        
        self.observation_space = copy.copy(self.ROBOT_STATE_OBSERVATION_SPACE)
        self.observation_space.spaces.update(self.ADDITIONAL_STATE_OBSERVATION_SPACE.spaces)
        for cam_name, is_ext in zip(self.all_cam_names, self.all_cam_is_ext):
            cam_size = _get_cam_size(cam_name)
            self.observation_space['cam_' + cam_name] = DictSpace(
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

    def extract_obs(
        self,
        robotenv_obs : Dict[str, Any]
    ):
        ret = _map_transform_to_array(robotenv_obs['robot_state'])
        
        
        eef_euler = ret['cartesian_position'][3:]
        eef_quaternion = tr3d.euler.euler2quat(*eef_euler)
        eef_rotation_matrix = tr3d.quaternions.quat2mat(eef_quaternion)
        
        ret['end_effector_euler'] = eef_euler
        ret['end_effector_quaternion'] = eef_quaternion
        ret['end_effector_6d_rotation'] = eef_rotation_matrix[:2, :].flatten()

        eef_pos = ret['cartesian_position'][:3]
        if self.use_tip_as_eef:
            eef_pos = self.transform_droid_eef_location_to_tip(
                eef_pos,
                eef_rotation_matrix
            )
        ret['end_effector_pos'] = eef_pos
        ret['cartesian_position'] = np.concatenate([eef_pos, eef_euler])

        for cam_name, is_ext in zip(self.all_cam_names, self.all_cam_is_ext):
            serial = cam_name.split('_')[0]
            rgb : np.ndarray = robotenv_obs['image'][cam_name]
            rgb = rgb[..., [2,1,0]].astype(np.uint8) # For ZED we get BGRA images, convert to RGB and drop alpha if present
            depth : np.ndarray = robotenv_obs['depth'][cam_name]
            depth = depth.astype(np.float32)
            intrinsics = np.asarray(robotenv_obs['camera_intrinsics'][cam_name], dtype=np.float32)
            extrinsics = np.asarray(robotenv_obs['camera_extrinsics'][cam_name], dtype=np.float32) if cam_name in robotenv_obs['camera_extrinsics'] else np.asarray(robotenv_obs['camera_extrinsics'][serial], dtype=np.float32)
            extrinsics = extrinsic_from_pose(extrinsics)
            if (
                (isinstance(self.flip_ext_cams_y, bool) and self.flip_ext_cams_y) or 
                (isinstance(self.flip_ext_cams_y, Mapping) and self.flip_ext_cams_y.get(cam_name, False))
            ):
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
            ret['cam_' + serial] = {
                "rgb": rgb.astype(np.uint8),
                "depth": depth.astype(np.float32),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics
            }

        return ret, intrinsics, extrinsics

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
        
        self._render_frame = [obs['cam_' + cam_name]['rgb'] for cam_name in self.all_cam_names if not cam_name.endswith('right')]
        return None, obs, {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }

    def step(self, action : np.ndarray):
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
        
        self._render_frame = [obs['cam_' + cam_name]['rgb'] for cam_name in self.all_cam_names if not cam_name.endswith('right')]
        return obs, 0, False, False, {}

    def render(self):
        return self._render_frame

    @staticmethod
    def rotation_6d_to_euler(
        rotation_6d: np.ndarray
    ) -> np.ndarray:
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

