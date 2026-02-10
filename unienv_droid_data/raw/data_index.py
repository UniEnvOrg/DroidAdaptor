# Create a data "index" that we can use to query frames.

import os
import glob
from typing import List, Union, Optional, Any
import dataclasses
import datetime
import serde
from serde.json import from_json, to_json
from .metadata import DroidDatasetEpisodicMetadata

@serde.serde
class DroidDatasetIndexEntry:
    lab_name : str
    is_success : bool

    @staticmethod
    def parse_date(date : str) -> datetime.date:
        return datetime.date.fromisoformat(date)
    
    @staticmethod
    def to_date_str(date : datetime.date) -> str:
        return date.isoformat()
    
    # YYYY-MM-DD
    date : datetime.date = serde.field(
        serializer=to_date_str,
        deserializer=parse_date
    )

    @staticmethod
    def parse_episode_time(time : str) -> datetime.datetime:
        try:
            return datetime.datetime.strptime(time, '%a_%b__%d_%H:%M:%S_%Y')
        except:
            pass

        try:
            return datetime.datetime.strptime(time, '%a_%b_%d_%H:%M:%S_%Y')
        except:
            pass
        
        try:
            return datetime.datetime.strptime(time, '%a_%b_%d_%H_%M_%S_%Y')
        except:
            pass

        return datetime.datetime.strptime(time, '%a_%b__%d_%H_%M_%S_%Y')
        
    
    @staticmethod
    def to_episode_time_str(time : datetime.datetime) -> str:
        return time.strftime('%a_%b__%d_%H:%M:%S_%Y')

    # Fri_Jul__7_11:35:19_2023
    # Mon_Mar__6_16_15_46_2023
    episode_time : datetime.datetime = serde.field(
        serializer=to_episode_time_str,
        deserializer=parse_episode_time
    )
    episode_time_raw : str


    start_idx: int
    trajectory_length: int

    def to_dir_path(self, root_dir : Optional[Union[str, os.PathLike]] = None) -> str:
        """
        Return path to the directory containing the episode.
        The dir should have `metadata_labname+serial+datetime(YYYY-MM-DD-HHh-MMm-SSs).json` format.
        And a directory named `recordings` containing the SVO / MP4 files.
        And `trajectory.h5` containing the trajectory data.
        """
        ret = os.path.join(
            self.lab_name, 
            'success' if self.is_success else 'failure', 
            self.to_date_str(
                self.date
            ),
            self.episode_time_raw
        )
        if root_dir is not None:
            ret = os.path.join(root_dir, ret)
        return ret

@dataclasses.dataclass
class DroidDatasetIndexMetadata:
    total_length : int
    index_entries : List[DroidDatasetIndexEntry]

    @staticmethod
    def from_path(path : Union[str, os.PathLike]) -> 'DroidDatasetIndexMetadata':
        with open(path, 'r') as f:
            return from_json(
                DroidDatasetIndexMetadata,
                f.read()
            )
    
    @staticmethod
    def from_dataset_root_dir(
        root_dir : Union[str, os.PathLike], 
        success_only = True,
        progress_bar = True
    ) -> 'DroidDatasetIndexMetadata':
        metadata_path = "unienv_dataset_idx.json" if not success_only else "unienv_success_dataset_idx.json"
        index_filepath = os.path.join(root_dir, metadata_path)
        if os.path.exists(index_filepath):
            return DroidDatasetIndexMetadata.from_path(index_filepath)
        else:
            index_metadata = build_index_metadata(root_dir, success_only=success_only, progress_bar=progress_bar)
            index_metadata.to_path(index_filepath)
            return index_metadata

    def to_path(self, path : Union[str, os.PathLike]):
        with open(path, 'w') as f:
            f.write(to_json(self))

def build_index_metadata(
    root_dir : Union[str, os.PathLike], 
    success_only : bool = True,
    progress_bar = True
) -> DroidDatasetIndexMetadata:
    index_metadata = DroidDatasetIndexMetadata(
        total_length=0,
        index_entries=[]
    )
    
    root_dir_list = os.listdir(root_dir)
    if progress_bar:
        try:
            import tqdm
            root_dir_list=tqdm.tqdm(root_dir_list)
        except:
            pass
    
    print("Building Droid Dataset Index Metadata" + (" (success only)" if success_only else "") + ", it should take 5-10 minutes...")
    pointer = 0
    counter = 0
    for lab in root_dir_list:
        lab_dir = os.path.join(root_dir, lab)
        lab_counter = 0
        lab_eps = 0
        if not os.path.isdir(lab_dir):
            continue
        for success_type in (['success'] if success_only else ['success', 'failure']):
            sucess_type_dir = os.path.join(lab_dir, success_type)
            assert os.path.isdir(sucess_type_dir)
            for day in os.listdir(sucess_type_dir):
                day_dir = os.path.join(sucess_type_dir, day)
                if not os.path.isdir(day_dir):
                    continue
                for episode_time in os.listdir(day_dir):
                    episode_dir = os.path.join(day_dir, episode_time)
                    if not os.path.isdir(episode_dir) or not os.path.isdir(os.path.join(episode_dir, 'recordings', 'SVO')):
                        continue

                    episode_metadata_glob = glob.glob('*.json', root_dir=episode_dir)
                    if len(episode_metadata_glob) == 0:
                        continue
                    episode_metadata_path = os.path.join(episode_dir, episode_metadata_glob[0])
                    
                    try:
                        episode_metadata = DroidDatasetEpisodicMetadata.from_path(episode_metadata_path)
                    except:
                        continue

                    if not (os.path.isfile(
                        os.path.join(episode_dir, 'trajectory.h5')
                    ) and os.path.isfile(
                        os.path.join(episode_dir, 'recordings', 'SVO', episode_metadata.ext1_cam_serial + '.svo')
                    ) and os.path.isfile(
                        os.path.join(episode_dir, 'recordings', 'SVO', episode_metadata.ext2_cam_serial + '.svo')
                    ) and os.path.isfile(
                        os.path.join(episode_dir, 'recordings', 'SVO', episode_metadata.wrist_cam_serial + '.svo')
                    )):
                        continue

                    index_entry = DroidDatasetIndexEntry(
                        lab_name=lab,
                        is_success=success_type == 'success',
                        date=DroidDatasetIndexEntry.parse_date(day),
                        episode_time=DroidDatasetIndexEntry.parse_episode_time(episode_time),
                        episode_time_raw=episode_time,
                        start_idx=pointer,
                        trajectory_length=episode_metadata.trajectory_length
                    )
                    lab_counter += episode_metadata.trajectory_length
                    lab_eps += 1
                    pointer += episode_metadata.trajectory_length
                    index_metadata.total_length += episode_metadata.trajectory_length
                    index_metadata.index_entries.append(index_entry)
                    counter += 1
    print("Done building Droid Dataset Index Metadata")
    return index_metadata
    
