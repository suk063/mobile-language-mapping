import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mshab.utils.dataset import ClosableDataset


@dataclass
class DataConfig:
    files: list[str]
    clip_cache_files: list[str]
    load_clip_cache: bool
    mask_out_classes: list[str]
    build_config_names: Optional[list[str]]
    sensor_names: list[str]
    valid_ratio: float
    batch_size: int
    num_workers: int
    multiprocessing_context: str


class StaticMappingDataset(ClosableDataset):
    def __init__(self, cfg: DataConfig, records=None):
        super().__init__()
        self.cfg = cfg
        if self.cfg.load_clip_cache:
            if len(self.cfg.clip_cache_files) < len(self.cfg.files):
                tqdm.write("number of clip cache files is less than data files, " "loading clip cache will be disabled")
                self.cfg.load_clip_cache = False

        self.fps: list[Union[dict, h5py.File]] = []
        self.clip_cache_fps: list[Union[dict, h5py.File]] = []
        self.mask_out_ids: list[list[int]] = []
        self.records = []
        self.episode_configs: list[dict] = []
        self.scene_ids: dict[str, int] = {}

        self._open_fps()
        self._build_scene_ids()

        if records is not None:
            self.records = records
        else:
            self._create_records()

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["fps"]
        del state["clip_cache_fps"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_fps()

    def _open_fps(self):
        self.fps = []
        self.clip_cache_fps = []
        for file in self.cfg.files:
            self.fps.append(self._open_fp(file))

        if self.cfg.mask_out_classes is not None:
            mask_out_set = set(self.cfg.mask_out_classes)
            self.mask_out_ids = []
            for fp in self.fps:
                mask_out_classes = defaultdict(list)
                for seg_id, seg_class in fp["segmentation_id_map"].items():
                    if seg_class not in mask_out_set:
                        continue
                    mask_out_classes[seg_class].append(int(seg_id))
                mask_out_ids = reduce(lambda x, y: x + y, mask_out_classes.values(), [])
                mask_out_ids = sorted(mask_out_ids)
                self.mask_out_ids.append(mask_out_ids)
        else:
            self.mask_out_ids = [[]] * len(self.fps)

        if self.cfg.load_clip_cache:
            for file in self.cfg.clip_cache_files:
                if os.path.exists(file):
                    self.clip_cache_fps.append(self._open_fp(file))
                else:
                    tqdm.write(f"Clip cache file not found: {file}")
                    self.cfg.load_clip_cache = False
                    self.clip_cache_fps = []
                    break

            if self.cfg.load_clip_cache:
                if len(self.clip_cache_fps) != len(self.fps):
                    tqdm.write(
                        f"Number of clip cache files ({len(self.clip_cache_fps)}) "
                        f"does not match number of data files ({len(self.fps)}), "
                        f"disabling clip cache loading"
                    )
                    self.cfg.load_clip_cache = False
                    self.clip_cache_fps = []

            if self.cfg.load_clip_cache and self.cfg.mask_out_classes is not None:
                for mask_out_ids, cache_fp in zip(self.mask_out_ids, self.clip_cache_fps):
                    mask_out_ids_cache = cache_fp.get("mask_out_ids", [])
                    if mask_out_ids_cache != mask_out_ids:
                        tqdm.write(
                            "Mask out ids in clip cache file do not match those in "
                            "data file, disabling clip cache loading"
                        )
                        self.cfg.load_clip_cache = False
                        self.clip_cache_fps = []
                        break

    @staticmethod
    def _open_fp(file: str) -> Union[dict, h5py.File]:
        if file.endswith(".h5"):
            return h5py.File(file, "r")
        elif file.endswith(".pt"):
            return torch.load(file, mmap=True)
        else:
            raise ValueError(f"Unsupported file format: {file}")

    def _build_scene_ids(self):
        # build the scene_ids for grid_net
        init_config_names = set()
        self.episode_configs = []
        for fp in self.fps:
            if isinstance(fp, h5py.File):
                episode_configs = json.loads(fp.attrs["episode_configs"])
            else:
                episode_configs = fp["episode_configs"]
            for episode_config in episode_configs:
                init_config_names.add(episode_config["init_config_name"])
            self.episode_configs.append(
                {f"traj_{i}": episode_config for i, episode_config in enumerate(episode_configs)}
            )
        init_config_names = sorted(list(init_config_names))
        self.scene_ids = {init_config_name: i for i, init_config_name in enumerate(init_config_names)}

    def _create_records(self):
        tqdm.write("Loading data from files...")

        for fp_idx, fp in tqdm(list(enumerate(self.fps)), desc="Files", ncols=80):
            episode_configs = self.episode_configs[fp_idx]
            for traj_name in fp.keys():
                if not traj_name.startswith("traj"):
                    continue
                build_config_name = episode_configs[traj_name]["build_config_name"]
                if self.cfg.build_config_names is not None and build_config_name not in self.cfg.build_config_names:
                    tqdm.write(
                        f"Skipping trajectory {traj_name} in file {fp_idx} "
                        f"due to build config name {build_config_name} not in allowed list."
                    )
                    continue
                traj_data = fp[traj_name]
                for sensor_name in self.cfg.sensor_names:
                    sensor_data = traj_data[sensor_name]
                    n = sensor_data["rgb"].shape[0]
                    self.records += [(fp_idx, traj_name, sensor_name, i) for i in range(n)]
        if len(self.records) == 0:
            raise RuntimeError("No records found")
        if len(self.fps) > 1:
            intrinsic = np.asarray(self.fps[0]["intrinsic"][:])
            for fp in self.fps[1:]:
                if not np.array_equal(np.asarray(fp["intrinsic"][:]), intrinsic):
                    raise RuntimeError("Intrinsic matrices do not match across multiple data files.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp_idx, traj_name, sensor_name, i = self.records[idx]
        fp = self.fps[fp_idx]
        sensor_data = fp[traj_name][sensor_name]

        out = dict(
            extrinsic=self.to_tensor(sensor_data["extrinsic"][i]),
            scene_ids=torch.tensor(
                self.scene_ids[self.episode_configs[fp_idx][traj_name]["init_config_name"]],
                dtype=torch.int32,
            ),
        )

        depth = self.to_tensor(sensor_data["depth"][i])  # (h, w)
        mask_out_ids = self.mask_out_ids[fp_idx]
        if len(mask_out_ids) > 0:
            segmentation = self.to_tensor(sensor_data["segmentation"][i])
            for seg_id in mask_out_ids:
                depth[segmentation == seg_id] = 0
        out["depth"] = depth

        if self.cfg.load_clip_cache:
            out["clip"] = self.to_tensor(self.clip_cache_fps[fp_idx][traj_name][sensor_name][i])
        else:
            # (3, h, w)
            out["rgb"] = self.to_tensor(sensor_data["rgb"][i]).permute(2, 0, 1)

        return out

    @staticmethod
    def to_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise TypeError(f"Unsupported type for conversion to tensor: {type(data)}")

    @property
    def intrinsic(self):
        assert len(self.fps) > 0, "No data files loaded."
        return np.asarray(self.fps[0]["intrinsic"][:])

    def close(self):
        self.records = []
        self.fps = []
        self.clip_cache_fps = []

    def split(self, valid_ratio: float):
        n = len(self.records)
        n_valid = int(n * valid_ratio)
        indices = torch.randperm(n)
        valid_indices = indices[:n_valid].tolist()
        train_indices = indices[n_valid:].tolist()
        train_dataset = StaticMappingDataset(
            self.cfg,
            [self.records[x] for x in train_indices],
        )
        valid_dataset = StaticMappingDataset(
            self.cfg,
            [self.records[x] for x in valid_indices],
        )
        return train_dataset, valid_dataset
