from dataclasses import dataclass
from typing import List, Literal, Union
import datetime
import numpy as np
import serde
import glob
import os
from serde.json import from_json, to_json

ALL_LAB_NAMES = [
    "AUTOLab",
    "IPRL",
    "RAD",
    "RAIL",
    "WEIRD",
    "CLVR",
    "IRIS",
    "REAL",
    "GuptaLab",
    "RPL",
    "ILIAD",
    "PennPAL",
    'TRI'
]
all_lab_names_lower_map = {
    lab.lower(): lab for lab in ALL_LAB_NAMES
}
ALL_LAB_NAMES_TYPE = Literal['AUTOLab', 'IPRL', "RAD", 'RAIL', 'WEIRD', 'CLVR', 'IRIS', 'REAL', 'GuptaLab', 'RPL', 'ILIAD', 'PennPAL', 'TRI']

@serde.serde
class DroidDatasetEpisodicMetadata:
    uuid: str
    lab : ALL_LAB_NAMES_TYPE = serde.field(
        deserializer=lambda name: all_lab_names_lower_map[name.lower()]
    )
    user: str
    user_id: str
    # 2023-07-07
    date: datetime.date = serde.field(
        serializer=lambda x: x.isoformat(),
        deserializer=lambda x: datetime.date.fromisoformat(x)
    )
    
    # 2023-07-07-09h-42m-23s
    timestamp: datetime.datetime = serde.field(
        serializer=lambda x: x.strftime('%Y-%m-%d-%Hh-%Mm-%Ss'),
        deserializer=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d-%Hh-%Mm-%Ss')
    )
    hdf5_path: str
    building: str
    scene_id: int
    success: bool
    robot_serial: str
    r2d2_version: str
    current_task: str
    trajectory_length: int
    wrist_cam_serial: str
    ext1_cam_serial: str
    ext2_cam_serial: str
    wrist_cam_extrinsics: np.ndarray # (6,)
    ext1_cam_extrinsics: np.ndarray # (6,)
    ext2_cam_extrinsics: np.ndarray # (6,)
    wrist_svo_path: str
    wrist_mp4_path: str
    ext1_svo_path: str
    ext1_mp4_path: str
    ext2_svo_path: str
    ext2_mp4_path: str
    left_mp4_path: str
    right_mp4_path: str

    @staticmethod
    def from_path(path : Union[str, os.PathLike]) -> 'DroidDatasetEpisodicMetadata':
        with open(path, 'r') as f:
            return from_json(DroidDatasetEpisodicMetadata, f.read())
    
    @staticmethod
    def from_episode_dir(episode_dir : Union[str, os.PathLike]) -> 'DroidDatasetEpisodicMetadata':
        epdir_jsons = glob.glob("*.json", root_dir=episode_dir)
        if len(epdir_jsons) == 0:
            raise FileNotFoundError("No metadata json file found in episode directory")
        metadata_path = os.path.join(episode_dir, epdir_jsons[0])
        return DroidDatasetEpisodicMetadata.from_path(metadata_path)