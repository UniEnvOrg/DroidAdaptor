# DROID Dataset dataloader class.
from typing import List, Literal, Union, Dict, Optional, Any, Tuple, Callable
import pickle
import torch
from unienv_interface.space import Space, DictSpace, BoxSpace, TextSpace, BinarySpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_data.base import BatchBase, IndexableType
from functools import partial
import os
import torch
import json
import h5py
import numpy as np
import glob
from pyzed import sl

from .metadata import DroidDatasetEpisodicMetadata
from .data_index import DroidDatasetIndexEntry, DroidDatasetIndexMetadata, build_index_metadata
from .svo_file import *

__all__ = [
    "RawDroidDataset"
]

NumpyBox = partial(BoxSpace, NumpyComputeBackend)
NumpyDict = partial(DictSpace, NumpyComputeBackend)
NumpyBinarySpace = partial(BinarySpace, NumpyComputeBackend)

# Joint Limits are Read from Mujoco Menagerie's FR3 Arm MJCF
_FR3_JNT_POSITION_LIMITS = np.array([
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
])
# Actually if we want more accurate limits, we should use the limits from the hardware config file
# https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/config/fr3/franka_hardware.yaml#L43

_FR3_JNT_POSITION_BOX = NumpyBox(
    low=_FR3_JNT_POSITION_LIMITS[0],
    high=_FR3_JNT_POSITION_LIMITS[1],
    dtype=np.float32,
    shape=(7,)
)

def _get_rgb_image_space(
    height : int,
    width : int,
    channels : int = 3
) -> BoxSpace:
    return NumpyBox(
        low=0,
        high=255,
        dtype=np.uint8,
        shape=(height, width, channels)
    )

def _get_depth_image_space(
    height : int,
    width : int,
) -> BoxSpace:
    return NumpyBox(
        low=0,
        high=np.inf,
        dtype=np.float32,
        shape=(height, width)
    )

def pad_images_to(
    images : np.ndarray,
    target : int
) -> np.ndarray:
    n_pad = target - images.shape[0]
    if n_pad == 0:
        return images
    if n_pad < 0:
        return images[:n_pad]
    last_image = images[-1:]
    padding = np.repeat(last_image, n_pad, axis=0)
    return np.concatenate([images, padding], axis=0)

def _get_frame_space(
    height : int,
    width : int,
    read_right : bool = False
):
    ret_space = NumpyDict(
        {
            "steps": NumpyDict({
                "action_dict": NumpyDict({
                    # cartesian_position is not normalized, its just that its impossible to exceed the limits of [-1, 1]
                    "cartesian_position": NumpyBox(low=np.array([-1.0]*3 + [-np.pi]*3), high=np.array([1.0] * 3 + [np.pi] * 3), dtype=np.float32, shape=(6,)),
                    "cartesian_velocity": NumpyBox(low=-1.0, high=1.0, dtype=np.float32, shape=(6,)),
                    # 0 when open, 1 when closed
                    "gripper_position": NumpyBox(low=0.0, high=1.0, dtype=np.float32, shape=()),
                    "gripper_velocity": NumpyBox(low=-1.0, high=1.0, dtype=np.float32, shape=()),
                    "joint_position": _FR3_JNT_POSITION_BOX,
                    "joint_velocity": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(7,)),
                }),
                "observation": NumpyDict({
                    # Robot States
                    "cartesian_position": NumpyBox(low=np.array([-1.0]*3 + [-np.pi]*3), high=np.array([1.0] * 3 + [np.pi] * 3), dtype=np.float32, shape=(6,)),
                    "gripper_position": NumpyBox(low=0.0, high=1.0, dtype=np.float32, shape=()),
                    "joint_position": _FR3_JNT_POSITION_BOX,
                    "joint_torques_computed": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(7,)),
                    "joint_velocities": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(7,)),
                    "motor_torques_measured": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(7,)),
                    # Camera Observations
                    "exterior_image_1_left": _get_rgb_image_space(height, width),
                    "exterior_depth_1_left": _get_depth_image_space(height, width),
                    "exterior_image_2_left": _get_rgb_image_space(height, width),
                    "exterior_depth_2_left": _get_depth_image_space(height, width),
                    "wrist_image_left": _get_rgb_image_space(height, width),
                    "wrist_depth_left": _get_depth_image_space(height, width),
                    # Camera Parameters
                    "camera_intrinsics": NumpyDict({
                        "wrist_camera_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
                        "exterior_camera_1_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
                        "exterior_camera_2_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
                    }),
                    "camera_extrinsics": NumpyDict({
                        "wrist_camera_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(6,)),
                        "exterior_camera_1_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(6,)),
                        "exterior_camera_1_is_postcalibrated": NumpyBinarySpace(shape=()),
                        "exterior_camera_2_left": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(6,)),
                        "exterior_camera_2_is_postcalibrated": NumpyBinarySpace(shape=()),
                    }),
                }),
                "language_instruction_1": TextSpace(NumpyComputeBackend, max_length=512),
                "language_instruction_2": TextSpace(NumpyComputeBackend, max_length=512),
                "language_instruction_3": TextSpace(NumpyComputeBackend, max_length=512),
                "language_instruction_is_post_annotated": BinarySpace(NumpyComputeBackend, shape=()),
                "is_first": NumpyBox(low=0, high=1, dtype=np.uint8, shape=()),
                "is_last": NumpyBox(low=0, high=1, dtype=np.uint8, shape=()),
            })
        }
    )
    if read_right:
        ret_space['steps']['observation'].spaces.update({
            "exterior_image_1_right": _get_rgb_image_space(height, width),
            "exterior_depth_1_right": _get_depth_image_space(height, width),
            "exterior_image_2_right": _get_rgb_image_space(height, width),
            "exterior_depth_2_right": _get_depth_image_space(height, width),
            "wrist_image_right": _get_rgb_image_space(height, width),
            "wrist_depth_right": _get_depth_image_space(height, width),
        })
        ret_space['steps']['observation']['camera_intrinsics'].spaces.update({
            "wrist_camera_right": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
            "exterior_camera_1_right": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
            "exterior_camera_2_right": NumpyBox(low=-np.inf, high=np.inf, dtype=np.float32, shape=(3, 3)),
        })
        # No extrinsics for right cameras!
    return ret_space

def _get_metadata_space() -> DictSpace:
    return NumpyDict({
        "file_path": TextSpace(NumpyComputeBackend, max_length=1024)
    })

class RawDroidDataset(BatchBase[Dict[str, Any], NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    is_mutable = False

    def __init__(
        self, 
        root_dir : Union[str, os.PathLike],
        droid_annotations_path: Optional[Union[str, os.PathLike]] = None,
        target_resolution : sl.Resolution = sl.Resolution(672, 376),
        success_only: bool = True,
        read_right: bool = False
    ):
        self.root_dir = root_dir
        self.read_right = read_right
        self.target_resolution = target_resolution
        self.success_only = success_only

        # Build index cache
        self.index_metadata = DroidDatasetIndexMetadata.from_dataset_root_dir(root_dir, success_only=success_only)
        self.all_episode_start_idx = np.array([entry.start_idx for entry in self.index_metadata.index_entries])

        # Read additional annotations if provided
        if droid_annotations_path is not None:
            # Load calibrated extrinsics
            with open(os.path.join(droid_annotations_path, "cam2base_extrinsics.json"), "r") as f:
                self.cam2base_extrinsics = json.load(f)

            with open(os.path.join(droid_annotations_path, "cam2base_extrinsic_superset.json"), "r") as f:
                self.cam2base_extrinsics_superset = json.load(f)

            # Load additional language annotations
            with open(f'{droid_annotations_path}/droid_language_annotations.json', "r") as f:
                self.language_anns = json.load(f)
        else:
            self.cam2base_extrinsics = {}
            self.cam2base_extrinsics_superset = {}
            self.language_anns = {}
        
        # Batch Space
        single_frame_space = _get_frame_space(target_resolution.height, target_resolution.width, read_right=read_right)
        super().__init__(
            single_frame_space,
            _get_metadata_space()
        )

    def get_camera_extrinsics(
        self, 
        episode_metadata: DroidDatasetEpisodicMetadata,
        cam_serial : str, 
        h5_file : h5py.Group, 
        index : Union[int, slice, List[int], np.ndarray],
        batch_size : Optional[int],
    ):
        is_wrist_cam = cam_serial == episode_metadata.wrist_cam_serial
        
        if not is_wrist_cam:
            ret = None
            if episode_metadata.uuid in self.cam2base_extrinsics and cam_serial in self.cam2base_extrinsics[episode_metadata.uuid]:
                ret = np.asarray([self.cam2base_extrinsics[episode_metadata.uuid][cam_serial]], dtype=np.float32).reshape(6,)
            elif episode_metadata.uuid in self.cam2base_extrinsics_superset and cam_serial in self.cam2base_extrinsics_superset[episode_metadata.uuid]:
                ret = np.asarray([self.cam2base_extrinsics_superset[episode_metadata.uuid][cam_serial]], dtype=np.float32).reshape(6,)
            
            if ret is not None:
                if batch_size is not None:
                    ret = ret[np.newaxis]
                    if batch_size > 1:
                        ret = np.repeat(ret, batch_size, axis=0)
                return (
                    np.ones((batch_size,) if batch_size is not None else (), dtype=bool),
                    ret
                )

        return (
            np.zeros((batch_size,) if batch_size is not None else (), dtype=bool),
            np.asarray(h5_file['observation']['camera_extrinsics'][f'{cam_serial}_left'][index], dtype=np.float32)
        )

    def get_language_instruction(self, episode_metadata : DroidDatasetEpisodicMetadata):
        if episode_metadata.uuid in self.language_anns:
            language_ann_raw = self.language_anns[episode_metadata.uuid]
            language_ann = {
                'language_instruction_1': language_ann_raw.get('language_instruction1', ""),
                'language_instruction_2': language_ann_raw.get('language_instruction2', ""),
                'language_instruction_3': language_ann_raw.get('language_instruction3', ""),
            }
            language_ann['language_instruction_is_post_annotated'] = True
        else:
            language_ann = {
                'language_instruction_1': episode_metadata.current_task, 
                'language_instruction_2': "",
                'language_instruction_3': "",
                'language_instruction_is_post_annotated': False
            }
        return language_ann

    def extract_frame_from_episode(
        self,
        episode_dir : Union[str, os.PathLike],
        metadata : DroidDatasetEpisodicMetadata,
        index : Optional[Union[int, slice, List[int], np.ndarray]]
    ) -> Tuple[
        Dict[str, Any], # Frames
        Dict[str, Any] # Context
    ]:
        if index is None:
            index = slice(None)
        if isinstance(index, int):
            return NumpyComputeBackend.map_fn_over_arrays(
                self.extract_frame_from_episode(episode_dir, metadata, [index]),
                lambda x: x[0]
            )

        # Convert all index to numpy array
        svo_slice = index
        if isinstance(index, slice):
            index = np.arange(*index.indices(metadata.trajectory_length))
        elif isinstance(index, list):
            index = np.array(index, dtype=np.int64)
        
        batch_size = len(index)

        h5_file = h5py.File(os.path.join(self.root_dir, episode_dir, "trajectory.h5"), 'r')

        camera_observation = read_camera_SVO(os.path.join(self.root_dir, episode_dir), metadata, frame_range=svo_slice, target_resolution=self.target_resolution, read_right=self.read_right)
        for key in camera_observation:
            if key == 'camera_intrinsics':
                continue
            # Sometimes the raw SVO recordings have less frames than the trajectory length recorded in metadata
            camera_observation[key] = pad_images_to(camera_observation[key], batch_size)
        
        camera_observation['camera_intrinsics'] = {
            key: np.repeat(value[np.newaxis, ...], batch_size, axis=0)
            for key, value in camera_observation['camera_intrinsics'].items()
        }
        exterior_camera_1_is_postcalibrated, exterior_camera_1_extrinsics_left = self.get_camera_extrinsics(metadata, metadata.ext1_cam_serial, h5_file, index, batch_size)
        exterior_camera_2_is_postcalibrated, exterior_camera_2_extrinsics_left = self.get_camera_extrinsics(metadata, metadata.ext2_cam_serial, h5_file, index, batch_size)
        camera_observation['camera_extrinsics'] = {
            "wrist_camera_left": self.get_camera_extrinsics(metadata, metadata.wrist_cam_serial, h5_file, index, batch_size)[1],
            "exterior_camera_1_left": exterior_camera_1_extrinsics_left,
            "exterior_camera_1_is_postcalibrated": exterior_camera_1_is_postcalibrated,
            "exterior_camera_2_left": exterior_camera_2_extrinsics_left,
            "exterior_camera_2_is_postcalibrated": exterior_camera_2_is_postcalibrated,
        }

        frames = {
            "steps": {}
        }
        frames['steps']['action_dict'] = {
            "cartesian_position": np.asarray(h5_file['action']['cartesian_position'][index], dtype=np.float32),
            "cartesian_velocity": np.asarray(h5_file['action']['cartesian_velocity'][index], dtype=np.float32),
            'gripper_position': np.asarray(h5_file['action']['gripper_position'][index], dtype=np.float32),
            'gripper_velocity': np.asarray(h5_file['action']['gripper_velocity'][index], dtype=np.float32),
            'joint_position': np.asarray(h5_file['action']['joint_position'][index], dtype=np.float32),
            'joint_velocity': np.asarray(h5_file['action']['joint_velocity'][index], dtype=np.float32)
        }
        frames['steps']['observation'] = {
            'cartesian_position': np.asarray(h5_file['observation']['robot_state']['cartesian_position'][index], dtype=np.float32),
            'gripper_position': np.asarray(h5_file['observation']['robot_state']['gripper_position'][index], dtype=np.float32),
            'joint_position': np.asarray(h5_file['observation']['robot_state']['joint_positions'][index], dtype=np.float32),
            'joint_torques_computed': np.asarray(h5_file['observation']['robot_state']['joint_torques_computed'][index], dtype=np.float32),
            'joint_velocities': np.asarray(h5_file['observation']['robot_state']['joint_velocities'][index], dtype=np.float32),
            'motor_torques_measured': np.asarray(h5_file['observation']['robot_state']['motor_torques_measured'][index], dtype=np.float32),
            **camera_observation
        }
        frames['steps']['is_first'] = index == 0
        frames['steps']['is_last'] = index == (metadata.trajectory_length - 1)
        
        episode_language_ann = self.get_language_instruction(metadata)
        frames['steps']['language_instruction_1'] = np.array([episode_language_ann['language_instruction_1']]*batch_size, dtype=object)
        frames['steps']['language_instruction_2'] = np.array([episode_language_ann['language_instruction_2']]*batch_size, dtype=object)
        frames['steps']['language_instruction_3'] = np.array([episode_language_ann['language_instruction_3']]*batch_size, dtype=object)
        frames['steps']['language_instruction_is_post_annotated'] = np.full((batch_size, ), episode_language_ann['language_instruction_is_post_annotated'], dtype=bool)

        metadata = {
            "file_path": np.array([str(episode_dir)]*batch_size, dtype=object)
        }

        return frames, metadata
    
    def __len__(self):
        return self.index_metadata.total_length

    @property
    def num_trajectories(self) -> int:
        return len(self.index_metadata.index_entries)

    def get_trajectory_start_idx_and_length(self, trajectory_idx : int) -> Tuple[int, int]:
        """
        Get the start and end index of the given trajectory index
        """
        assert -self.num_trajectories <= trajectory_idx < self.num_trajectories, f"Trajectory index {trajectory_idx} out of bounds for {len(self.index_metadata.index_entries)} trajectories"
        entry = self.index_metadata.index_entries[trajectory_idx]
        return entry.start_idx, entry.trajectory_length

    def _convert_single_index(self, idx : int) -> Tuple[int, int]:
        """
        Convert a single index to a tuple containing
         - the batch index
         - the index within the batch
        """
        assert -len(self) <= idx < len(self), f"Index {idx} out of bounds for batch of size {len(self)}"
        if idx < 0:
            idx += len(self)
        batch_index = int(self.backend.sum(
            idx >= self.all_episode_start_idx
        ) - 1)
        return batch_index, int(idx - self.all_episode_start_idx[batch_index])

    def _convert_index(self, idx : Union[IndexableType, NumpyArrayType]) -> Tuple[
        int, 
        List[
            Tuple[int, NumpyArrayType, NumpyArrayType]
        ]
    ]:
        """
        Convert an index for this batch to a tuple of:
         - The length of the resulting array
         - List of tuples, each containing:
             - The index of the batch
             - The index to index into the batch
             - The bool mask to index into the resulting array
        """
        if isinstance(idx, slice):
            idx_array = self.backend.arange(
                *idx.indices(len(self)),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        elif idx is Ellipsis:
            idx_array = self.backend.arange(
                len(self),
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
        elif self.backend.is_backendarray(idx):
            assert len(idx.shape) == 1, "Index must be 1D"
            assert self.backend.dtype_is_real_integer(idx.dtype) or self.backend.dtype_is_boolean(idx.dtype), \
                f"Index must be of integer or boolean type, got {idx.dtype}"
            if self.backend.dtype_is_boolean(idx.dtype):
                assert idx.shape[0] == len(self), f"Boolean index must have the same length as the batch, got {idx.shape[0]} vs {len(self)}"
                idx_array = self.backend.nonzero(idx)[0]
            else:
                idx_array = idx
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
        
        assert bool(self.backend.all(
            self.backend.logical_and(
                -len(self) <= idx_array,
                idx_array < len(self)
            )
        )), f"Index {idx} converted to {idx_array} is out of bounds for batch of size {len(self)}"
        
        # Convert negative indices to positive indices
        idx_array = self.backend.at(idx_array)[idx_array < 0].add(len(self))
        idx_array_bigger = idx_array[:, None] >= self.all_episode_start_idx[None, :] # (idx_array_shape, len(self.batches))
        idx_array_batch_idx = self.backend.sum(
            idx_array_bigger,
            axis=-1
        ) - 1 # (idx_array_shape, )
        
        result_batch_list = []
        batch_indexes = self.backend.unique_values(idx_array_batch_idx)
        for i in range(batch_indexes.shape[0]):
            batch_index = int(batch_indexes[i])
            result_mask = idx_array_batch_idx == batch_index
            index_into_batch = idx_array[result_mask] - self.all_episode_start_idx[batch_index]
            result_batch_list.append((batch_index, index_into_batch, result_mask))
        return idx_array.shape[0], result_batch_list

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]

    def get_at_with_metadata(self, idx):
        if isinstance(idx, int):
            batch_idx, index_into_batch = self._convert_single_index(idx)
            episode_dir_relative = self.index_metadata.index_entries[batch_idx].to_dir_path()
            episode_dir_absolute = os.path.join(self.root_dir, episode_dir_relative)
            dat, metadata = self.extract_frame_from_episode(
                episode_dir_relative,
                DroidDatasetEpisodicMetadata.from_episode_dir(episode_dir_absolute),
                index_into_batch
            )
            return dat, metadata
        else:
            batch_size, batch_list = self._convert_index(idx)
            result_space = sbu.batch_space(
                self.single_space,
                batch_size,
            )
            result = result_space.create_empty()
            
            metadata_space = sbu.batch_space(
                self.single_metadata_space,
                batch_size,
            )
            metadata = metadata_space.create_empty()

            for batch_index, index_into_batch, mask in batch_list:
                episode_dir_relative = self.index_metadata.index_entries[batch_index].to_dir_path()
                episode_dir_absolute = os.path.join(self.root_dir, episode_dir_relative)
                batch_result, metadata_result = self.extract_frame_from_episode(
                    episode_dir_relative,
                    DroidDatasetEpisodicMetadata.from_episode_dir(episode_dir_absolute),
                    index_into_batch
                )
                result = sbu.set_at(
                    result_space,
                    result,
                    mask,
                    batch_result,
                )
                metadata = sbu.set_at(
                    metadata_space,
                    metadata,
                    mask,
                    metadata_result,
                )
            
            return result, metadata

    def get_trajectory_at(
        self,
        trajectory_idx : int
    ) -> Dict[str, Any]:
        return self.get_trajectory_at_with_metadata(trajectory_idx)[0]

    def get_trajectory_at_with_metadata(
        self,
        trajectory_idx : int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert -self.num_trajectories <= trajectory_idx < self.num_trajectories, f"Trajectory index {trajectory_idx} out of bounds for {len(self.index_metadata.index_entries)} trajectories"
        if trajectory_idx < 0:
            trajectory_idx += self.num_trajectories
        entry = self.index_metadata.index_entries[trajectory_idx]
        episode_dir_relative = entry.to_dir_path()
        episode_dir_absolute = os.path.join(self.root_dir, episode_dir_relative)
        return self.extract_frame_from_episode(
            episode_dir_relative,
            DroidDatasetEpisodicMetadata.from_episode_dir(episode_dir_absolute),
            None
        )
