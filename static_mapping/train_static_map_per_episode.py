import json
import os.path
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from functools import reduce
from typing import Optional, Union

import h5py
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from dacite import from_dict
from omegaconf import OmegaConf
from ruamel import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from lang_mapping.grid_net import GridNet
from lang_mapping.mapper import MultiVoxelHashTable
from lang_mapping.module import ImplicitDecoder
from lang_mapping.utils import get_visual_features
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.dataset import ClosableDataset
import logging

logging.basicConfig(level=logging.INFO)


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


@dataclass
class ClipModelConfig:
    model_name: str = "EVA02-L-14"
    model_pretrained: str = "merged2b_s4b_b131k"


@dataclass
class GridDefinition:
    type: str = "regular"
    feature_dim: int = 60
    init_stddev: float = 0.2
    bound: list[list[float]] = field(default_factory=lambda: [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]])
    base_cell_size: float = 0.4
    per_level_scale: float = 2.0
    n_levels: int = 2
    second_order_grid_sample: bool = False


@dataclass
class VoxelHashTableConfig:
    one_to_one: bool  # whether to use one-to-one mapping for voxel features
    resolution: float  # finest cell size (e.g. 0.12)
    num_levels: int  # pyramid depth (e.g. 2)
    level_scale: float  # ratio between levels (e.g. 2.0)
    voxel_feature_dim: int  # per-level feature width (e.g. 32)
    hash_table_size: int  # buckets per level (power of two)
    scene_bound_min: list[float]  # xyz lower corner
    scene_bound_max: list[float]  # xyz upper corner


@dataclass
class GridCfg:
    name: str = "grid_net"  # grid_net or voxel_hash_table
    spatial_dim: int = 3
    n_scenes: int = 161
    grid_net: Optional[GridDefinition] = None
    voxel_hash_table: Optional[VoxelHashTableConfig] = None

    def as_dict(self):
        out = vars(self)
        if self.grid_net is not None:
            out["grid_net"] = vars(self.grid_net)
        if self.voxel_hash_table is not None:
            out["voxel_hash_table"] = vars(self.voxel_hash_table)
        return out


@dataclass
class Config:
    seed: int
    torch_deterministic: bool
    device_clip: str
    device_decoder: str
    epochs: int
    optimizer: str
    optimizer_kwargs: dict
    data: DataConfig
    clip_model: ClipModelConfig
    grid_cfg: GridCfg
    depth_downsample_method: str  # "nearest-exact", "nearest", "avg2d", "avg3d"
    decoder_hidden_dim: int
    decoder_output_dim: int
    output_dir: str
    valid_interval: int
    ckpt_interval: int
    test_model_dir: Optional[str]

    def as_dict(self):
        out = vars(self)
        out["data"] = vars(self.data)
        out["clip_model"] = vars(self.clip_model)
        out["grid_cfg"] = self.grid_cfg.as_dict()
        return out


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


def get_3d_coordinates(
    depth: torch.Tensor,
    camera_extrinsic: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    original_size: int = 224,
):
    """
    Computes 3D coordinates from 2D feature maps and depth.
    If camera_extrinsic is provided, returns (coords_world, coords_cam).
    Otherwise, returns coords_cam.
    Args:
        depth: [B, H_feat, W_feat] or [B, 1, H_feat, W_feat].
        camera_extrinsic: [B, 1, 3, 4], world->cam transform (None if absent).
        fx, fy, cx, cy: Camera intrinsics for original_size x original_size.
        original_size: Original image size (default=224).
    Returns:
        coords_world or coords_cam: [B, 3, H_feat, W_feat].
    """
    device = depth.device

    # Adjust depth shape if needed
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    B, H, W = depth.shape

    # Scale intrinsics
    scale_x = W / float(original_size)
    scale_y = H / float(original_size)
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    # Create pixel coordinate grid
    u = torch.arange(W, device=device).view(1, -1).expand(H, W) + 0.5
    v = torch.arange(H, device=device).view(-1, 1).expand(H, W) + 0.5
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    # Camera coordinates
    x_cam = (u - cx_new) * depth / fx_new
    y_cam = (v - cy_new) * depth / fy_new
    z_cam = depth
    coords_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

    # If extrinsic is given, convert cam->world
    if camera_extrinsic is not None:
        camera_extrinsic = camera_extrinsic.squeeze(1)  # [B, 3, 4]
        ones_row = torch.tensor([0, 0, 0, 1], device=device, dtype=camera_extrinsic.dtype).view(1, 1, 4)
        ones_row = ones_row.expand(B, 1, 4)  # [B, 1, 4]
        extrinsic_4x4 = torch.cat([camera_extrinsic, ones_row], dim=1)  # [B, 4, 4]
        extrinsic_inv = torch.inverse(extrinsic_4x4)  # [B, 4, 4]

        _, _, Hf, Wf = coords_cam.shape
        ones_map = torch.ones(B, 1, Hf, Wf, device=device)
        coords_hom = torch.cat([coords_cam, ones_map], dim=1)  # [B, 4, Hf, Wf]
        coords_hom_flat = coords_hom.view(B, 4, -1)  # [B, 4, Hf*Wf]

        world_coords_hom = torch.bmm(extrinsic_inv, coords_hom_flat)  # [B, 4, Hf*Wf]
        world_coords_hom = world_coords_hom.view(B, 4, Hf, Wf)
        coords_world = world_coords_hom[:, :3, :, :]
        return coords_world, coords_cam
    else:
        return coords_cam


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

        # Load datasets
        dataset = StaticMappingDataset(self.cfg.data)
        if len(dataset.scene_ids) != self.cfg.grid_cfg.n_scenes:
            raise ValueError(
                f"Number of scenes in dataset ({len(dataset.scene_ids)}) "
                f"does not match grid_cfg.n_scenes ({self.cfg.grid_cfg.n_scenes})."
            )

        self.train_dataset, self.valid_dataset = dataset.split(self.cfg.data.valid_ratio)
        self.train_loader = ClosableDataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            pin_memory=True,
            multiprocessing_context=self.cfg.data.multiprocessing_context,
            persistent_workers=True,
            prefetch_factor=4,
        )
        self.valid_loader = ClosableDataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            pin_memory=True,
            multiprocessing_context=self.cfg.data.multiprocessing_context,
            persistent_workers=True,
            prefetch_factor=4,
        )
        intrinsic = self.train_dataset.intrinsic
        self.fx = intrinsic[0, 0].item()
        self.fy = intrinsic[1, 1].item()
        self.cx = intrinsic[0, 2].item()
        self.cy = intrinsic[1, 2].item()

        # create models
        self.clip_model = None
        if not self.cfg.data.load_clip_cache:
            self.clip_model = open_clip.create_model_and_transforms(
                self.cfg.clip_model.model_name,
                self.cfg.clip_model.model_pretrained,
            )[0].to(self.cfg.device_clip)
            self.clip_model.eval()
        if self.cfg.grid_cfg.name == "voxel_hash_table":
            assert self.cfg.grid_cfg.voxel_hash_table is not None, "voxel_hash_table config must be provided"
            vht_cfg = self.cfg.grid_cfg.voxel_hash_table
            self.grid_net = MultiVoxelHashTable(
                n_scenes=self.cfg.grid_cfg.n_scenes,
                resolution=vht_cfg.resolution,
                num_levels=vht_cfg.num_levels,
                level_scale=vht_cfg.level_scale,
                feature_dim=vht_cfg.voxel_feature_dim,
                hash_table_size=vht_cfg.hash_table_size,
                scene_bound_min=vht_cfg.scene_bound_min,
                scene_bound_max=vht_cfg.scene_bound_max,
            )
            self.implicit_decoder = ImplicitDecoder(
                voxel_feature_dim=vht_cfg.voxel_feature_dim * vht_cfg.num_levels,
                hidden_dim=self.cfg.decoder_hidden_dim,
                output_dim=self.cfg.decoder_output_dim,
            ).to(self.cfg.device_decoder)
        elif self.cfg.grid_cfg.name == "grid_net":
            assert self.cfg.grid_cfg.grid_net is not None, "grid_cfg.grid must be specified for grid_net"
            grid_cfg = self.cfg.grid_cfg.grid_net
            self.grid_net = GridNet(cfg=asdict(grid_cfg))
            self.implicit_decoder = ImplicitDecoder(
                voxel_feature_dim=grid_cfg.feature_dim * grid_cfg.n_levels,
                hidden_dim=self.cfg.decoder_hidden_dim,
                output_dim=self.cfg.decoder_output_dim,
            ).to(self.cfg.device_decoder)
        else:
            raise ValueError(f"Unknown grid_cfg.name: {self.cfg.grid_cfg.name}")

    def train(self):
        # create optimizer
        self.grid_net.distribute_to_devices()
        params = list(self.grid_net.parameters())
        params += list(self.implicit_decoder.parameters())
        optimizer = getattr(torch.optim, self.cfg.optimizer)(params, **self.cfg.optimizer_kwargs)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.cfg.output_dir = os.path.join(self.cfg.output_dir, timestamp)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=self.cfg.output_dir)
        yaml_obj = yaml.YAML()
        yaml_obj.indent(mapping=4, sequence=6, offset=4)
        with open(os.path.join(self.cfg.output_dir, "config.yaml"), "w") as f:
            yaml_obj.dump(self.cfg.as_dict(), f)
        with open(os.path.join(self.cfg.output_dir, "scene_ids.yaml"), "w") as f:
            yaml_obj.dump(self.train_dataset.scene_ids, f)

        batch_cnt = 0
        valid_loss_min = float("inf")
        for epoch in tqdm(range(self.cfg.epochs), desc="Epoch", ncols=80, position=0):
            train_loss = 0
            self.setup_model(True)
            for batch in tqdm(self.train_loader, desc="Train", ncols=80, position=1, leave=False):
                with torch.enable_grad():
                    result = self.forward_model(batch)
                    loss = result["loss"]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss = loss.item()
                    train_loss += loss
                    self.tb.add_scalar("batch/train_loss", loss, global_step=batch_cnt)

                if batch_cnt % self.cfg.valid_interval == 0 and batch_cnt > 0:
                    valid_loss = self.valid()
                    self.tb.add_scalar("batch/valid_loss", valid_loss, global_step=batch_cnt)
                    self.setup_model(True)

                batch_cnt += 1
            train_loss /= len(self.train_loader)
            self.tb.add_scalar("epoch/train_loss", train_loss, global_step=epoch)
            valid_loss = self.valid()
            self.tb.add_scalar("epoch/valid_loss", valid_loss, global_step=epoch)
            if valid_loss < valid_loss_min:  # save the best model
                valid_loss_min = valid_loss
                self.save_model("best", epoch)
                tqdm.write(f"Epoch {epoch}: new best model saved with valid loss {valid_loss:.4f}")
            if (epoch + 1) % self.cfg.ckpt_interval == 0:
                # save checkpoint
                self.save_model(f"epoch_{epoch}", epoch)
                tqdm.write(f"Epoch {epoch}: checkpoint saved with valid loss {valid_loss:.4f}")

    def valid(self):
        self.setup_model(False)
        valid_loss = 0
        for batch in tqdm(self.valid_loader, desc="Valid", ncols=80, position=2, leave=False):
            with torch.no_grad():
                result = self.forward_model(batch)

                valid_loss += result["loss"].item()

        valid_loss /= len(self.valid_loader)
        return valid_loss

    def test(self):
        assert self.cfg.test_model_dir is not None, "Test model directory must be specified."
        self.grid_net.to("cpu")
        if isinstance(self.grid_net, MultiVoxelHashTable):
            self.grid_net = MultiVoxelHashTable.load_sparse(
                os.path.join(self.cfg.test_model_dir, "hash_voxel_sparse.pt")
            )
        else:
            state = torch.load(os.path.join(self.cfg.test_model_dir, "grid_net.pt"), map_location="cpu")
            self.grid_net.load_state_dict(state["model"])

        self.grid_net.distribute_to_devices()
        self.grid_net.eval()

        state = torch.load(
            os.path.join(self.cfg.test_model_dir, "implicit_decoder.pt"),
            map_location=self.cfg.device_decoder,
        )
        self.implicit_decoder.load_state_dict(state["model"])
        self.implicit_decoder.eval()
        loss = self.valid()
        tqdm.write(f"Test completed. Validation loss: {loss:.4f}")

    def setup_model(self, training=True):
        if self.clip_model is not None:
            self.clip_model.eval()
        self.grid_net.train(training)
        self.implicit_decoder.train(training)

    def forward_model(self, batch: dict):
        depth = batch["depth"].to(self.cfg.device_decoder) / 1000.0  # Convert depth from mm to m
        extrinsic = batch["extrinsic"].to(self.cfg.device_decoder)
        scene_ids = batch["scene_ids"].to(self.cfg.device_decoder)

        with torch.no_grad():
            if self.clip_model is None:
                visual_features = batch["clip"].to(self.cfg.device_decoder)
            else:
                rgb = batch["rgb"].float().to(self.cfg.device_clip) / 255.0
                visual_features = get_visual_features(self.clip_model, rgb)
                visual_features = visual_features.to(self.cfg.device_decoder)

            coords_world = None
            original_size = depth.shape[-1]
            if self.cfg.depth_downsample_method == "avg3d":
                coords_world = get_3d_coordinates(
                    depth,
                    extrinsic,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    original_size,
                )[0]
                s = depth.shape[1] // visual_features.shape[-1]
                coords_world = F.avg_pool2d(coords_world, kernel_size=s)
            elif self.cfg.depth_downsample_method == "avg2d":
                s = depth.shape[1] // visual_features.shape[-1]
                depth = F.avg_pool2d(depth.unsqueeze(1), kernel_size=s).squeeze(1)
            else:
                s = visual_features.shape[-1]
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(s, s),
                    mode=self.cfg.depth_downsample_method,
                ).squeeze(1)
            if coords_world is None:
                coords_world = get_3d_coordinates(
                    depth,
                    extrinsic,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    original_size,
                )[0]

            h, w = coords_world.shape[-2:]
            scene_ids = scene_ids.view(-1, 1, 1).expand(-1, h, w)

            # bs, c, h, w -> n, c
            mask = (depth > 0.0).flatten()  # [bs, h, w] -> [bs * h * w]
            c = visual_features.shape[1]  # feature dim, e.g. 768 for EVA02-L-14
            visual_features = visual_features.permute(0, 2, 3, 1).reshape(-1, c)[mask]
            coords_world = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
            scene_ids = scene_ids.flatten()[mask]

        voxel_features = self.grid_net.query_feature(coords_world, scene_ids)
        decoded_features = self.implicit_decoder(voxel_features, coords_world)
        cos_sim = F.cosine_similarity(decoded_features, visual_features, dim=-1)
        loss = 1.0 - cos_sim.mean()
        return dict(
            visual_features=visual_features,
            voxel_features=voxel_features,
            decoded_features=decoded_features,
            cos_sim=cos_sim,
            loss=loss,
        )

    def save_model(self, name, epoch):
        folder = os.path.join(self.cfg.output_dir, name)
        os.makedirs(folder, exist_ok=True)

        if isinstance(self.grid_net, MultiVoxelHashTable):
            dense_path = os.path.join(folder, "hash_voxel_dense.pt")
            sparse_path = os.path.join(folder, "hash_voxel_sparse.pt")
            self.grid_net.save_dense(dense_path)
            self.grid_net.save_sparse(sparse_path)
        else:
            torch.save(
                dict(model=self.grid_net.state_dict(), epoch=epoch),
                os.path.join(folder, "grid_net.pt"),
            )

        torch.save(
            dict(model=self.implicit_decoder.state_dict(), epoch=epoch),
            os.path.join(folder, "implicit_decoder.pt"),
        )

    def run(self):
        if self.cfg.test_model_dir is not None:
            tqdm.write("Testing the model...")
            self.test()
            return
        self.train()


def main():
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    cfg = from_dict(data_class=Config, data=OmegaConf.to_container(cfg))
    if cfg.test_model_dir is not None:
        # reload the config based on the test model directory
        test_model_dir = cfg.test_model_dir
        cfg = parse_cfg(default_cfg_path=os.path.join(test_model_dir, "../config.yaml"))
        cfg = from_dict(data_class=Config, data=OmegaConf.to_container(cfg))
        cfg.test_model_dir = test_model_dir
    if cfg.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
