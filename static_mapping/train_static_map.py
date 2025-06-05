import os.path
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from dacite import from_dict
from omegaconf import OmegaConf
from ruamel import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from dataclasses import dataclass, field, asdict
from typing import List, Optional

from lang_mapping.grid_net.grid_net import GridNet
from lang_mapping.module import ImplicitDecoder
from lang_mapping.utils import get_visual_features
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.dataset import ClosableDataset


@dataclass
class DataConfig:
    files: list[str]
    clip_cache_files: list[str]
    load_clip_cache: bool
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
    type: str
    feature_dim: int
    rank: int
    init_stddev: float
    bound: List[List[float]]
    base_cell_size: float
    per_level_scale: float
    n_levels: int
    n_scenes: int

@dataclass
class GridCfg:
    name: str
    spatial_dim: int
    grid: GridDefinition

@dataclass
class Config:
    seed: int
    torch_deterministic: bool
    device: str
    epochs: int
    optimizer: str
    optimizer_kwargs: dict
    data: DataConfig
    clip_model: ClipModelConfig
    depth_downsample_method: str  # "nearest-exact", "nearest", "avg2d", "avg3d"
    decoder_hidden_dim: int
    decoder_output_dim: int
    output_dir: str
    valid_interval: int
    ckpt_interval: int
    test_model_dir: Optional[str] = None
    grid_cfg: GridCfg = field(default_factory=GridCfg)

    def as_dict(self):
        out = vars(self)
        out["data"] = vars(self.data)
        out["clip_model"] = vars(self.clip_model)
        out["grid_cfg"] = asdict(self.grid_cfg) 
        return out


class StaticMappingDataset(ClosableDataset):
    def __init__(self, cfg: DataConfig, records=None):
        super().__init__()
        self.cfg = cfg
        if self.cfg.load_clip_cache:
            if len(self.cfg.clip_cache_files) < len(self.cfg.files):
                tqdm.write(
                    "number of clip cache files is less than data files, "
                    "loading clip cache will be disabled"
                )
                self.cfg.load_clip_cache = False

        self.fps: list[dict] = []
        self.clip_cache_fps: list[dict] = []
        self.records = []
        if records is not None:
            self.records = records
            self._open_fps()
        else:
            self.load()

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
        if self.cfg.load_clip_cache:
            for file in self.cfg.clip_cache_files:
                if os.path.exists(file):
                    self.clip_cache_fps.append(self._open_fp(file))
                else:
                    tqdm.write(f"Clip cache file not found: {file}")
                    self.cfg.load_clip_cache = False
                    self.clip_cache_fps = []
                    break

    @staticmethod
    def _open_fp(file: str):
        if file.endswith(".h5"):
            return h5py.File(file, "r")
        elif file.endswith(".pt"):
            return torch.load(file, mmap=True)
        else:
            raise ValueError(f"Unsupported file format: {file}")

    def load(self):
        tqdm.write("Loading data from files...")
        self._open_fps()

        for fp_idx, fp in tqdm(list(enumerate(self.fps)), desc="Files", ncols=80):
            for traj_name in fp.keys():
                if not traj_name.startswith("traj"):
                    continue
                traj_data = fp[traj_name]
                for sensor_name in traj_data.keys():
                    sensor_data = traj_data[sensor_name]
                    n = sensor_data["rgb"].shape[0]
                    self.records += [
                        (fp_idx, traj_name, sensor_name, i) for i in range(n)
                    ]
        if len(self.records) == 0:
            raise RuntimeError("No records found")
        if len(self.fps) > 1:
            intrinsic = self.fps[0]["intrinsic"][:]
            for fp in self.fps[1:]:
                if not np.array_equal(fp["intrinsic"][:], intrinsic):
                    raise RuntimeError(
                        "Intrinsic matrices do not match across multiple data files."
                    )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp_idx, traj_name, sensor_name, i = self.records[idx]
        fp = self.fps[fp_idx]
        sensor_data = fp[traj_name][sensor_name]

        out = dict(
            depth=self.to_tensor(sensor_data["depth"][i]),  # (h, w)
            extrinsic=self.to_tensor(sensor_data["extrinsic"][i]),  # (3, 4)
        )

        if self.cfg.load_clip_cache:
            out["clip"] = self.to_tensor(
                self.clip_cache_fps[fp_idx][traj_name][sensor_name][i]
            )
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
        if len(self.fps) > 0:
            return self.fps[0]["intrinsic"][:]
        return None

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
        ones_row = torch.tensor(
            [0, 0, 0, 1], device=device, dtype=camera_extrinsic.dtype
        ).view(1, 1, 4)
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

        device = torch.device(cfg.device)

        # Load datasets
        dataset = StaticMappingDataset(self.cfg.data)
        self.train_dataset, self.valid_dataset = dataset.split(
            self.cfg.data.valid_ratio
        )
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
            )[0].to(device)
            self.clip_model.eval()
            
        self.hash_voxel = GridNet(cfg=asdict(self.cfg.grid_cfg), device=device)
        self.hash_voxel = self.hash_voxel.to(device)
        
        self.implicit_decoder = ImplicitDecoder(
            voxel_feature_dim=self.cfg.grid_cfg.grid.feature_dim * self.cfg.grid_cfg.grid.n_levels,
            hidden_dim=self.cfg.decoder_hidden_dim,
            output_dim=self.cfg.decoder_output_dim,
        ).to(device)

        # create optimizer
        params = list(self.hash_voxel.parameters())
        params += list(self.implicit_decoder.parameters())
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            params, **self.cfg.optimizer_kwargs
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.cfg.output_dir = os.path.join(self.cfg.output_dir, timestamp)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=self.cfg.output_dir)
        yaml_obj = yaml.YAML()
        yaml_obj.indent(mapping=4, sequence=6, offset=4)
        with open(os.path.join(self.cfg.output_dir, "config.yaml"), "w") as f:
            yaml_obj.dump(self.cfg.as_dict(), f)

    def train(self):
        batch_cnt = 0
        valid_loss_min = float("inf")
        for epoch in tqdm(range(self.cfg.epochs), desc="Epoch", ncols=80, position=0):
            train_loss = 0
            self.setup_model(True)
            for batch in tqdm(
                self.train_loader, desc="Train", ncols=80, position=1, leave=False
            ):
                with torch.enable_grad():
                    result = self.forward_model(batch)
                    loss = result["loss"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss = loss.item()
                    train_loss += loss
                    self.tb.add_scalar("batch/train_loss", loss, global_step=batch_cnt)

                if batch_cnt % self.cfg.valid_interval == 0 and batch_cnt > 0:
                    valid_loss = self.valid()
                    self.tb.add_scalar(
                        "batch/valid_loss", valid_loss, global_step=batch_cnt
                    )
                    self.setup_model(True)

                batch_cnt += 1
            train_loss /= len(self.train_loader)
            self.tb.add_scalar("epoch/train_loss", train_loss, global_step=epoch)
            valid_loss = self.valid()
            self.tb.add_scalar("epoch/valid_loss", valid_loss, global_step=epoch)
            if valid_loss < valid_loss_min:  # save the best model
                valid_loss_min = valid_loss
                self.save_model("best", epoch)
                tqdm.write(
                    f"Epoch {epoch}: new best model saved with valid loss {valid_loss:.4f}"
                )
            if (epoch + 1) % self.cfg.ckpt_interval == 0:
                # save checkpoint
                self.save_model(f"epoch_{epoch}", epoch)
                tqdm.write(
                    f"Epoch {epoch}: checkpoint saved with valid loss {valid_loss:.4f}"
                )

    def valid(self):
        self.setup_model(False)
        valid_loss = 0
        for batch in tqdm(
            self.valid_loader, desc="Valid", ncols=80, position=2, leave=False
        ):
            with torch.no_grad():
                result = self.forward_model(batch)

                valid_loss += result["loss"].item()

        valid_loss /= len(self.valid_loader)
        return valid_loss

    def test(self):
        state = torch.load(
            os.path.join(self.cfg.test_model_dir, "hash_voxel.pt"),
            map_location=self.cfg.device,
        )
        self.hash_voxel.load_state_dict(state["model"])
        self.hash_voxel.eval()
        state = torch.load(
            os.path.join(self.cfg.test_model_dir, "implicit_decoder.pt"),
            map_location=self.cfg.device,
        )
        self.implicit_decoder.load_state_dict(state["model"])
        self.implicit_decoder.eval()
        loss = self.valid()
        tqdm.write(f"Test completed. Validation loss: {loss:.4f}")

    def setup_model(self, training=True):
        if self.clip_model is not None:
            self.clip_model.eval()
        self.hash_voxel.train(training)
        self.implicit_decoder.train(training)

    def forward_model(self, batch: dict):
        depth = batch["depth"].to(self.cfg.device) / 1000.0
        extrinsic = batch["extrinsic"].to(self.cfg.device)

        with torch.no_grad():
            if self.clip_model is None:
                visual_features = batch["clip"].to(self.cfg.device)
            else:
                rgb = batch["rgb"].float().to(self.cfg.device) / 255.0
                visual_features = get_visual_features(self.clip_model, rgb)

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
        # bs, c, h, w -> bs * h * w, c
        c = visual_features.shape[1]  # feature dim, e.g. 768 for EVA02-L-14
        visual_features = visual_features.permute(0, 2, 3, 1).reshape(-1, c)
        # bs, 3, h, w -> bs * h * w, 3
        coords_world = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)

        # (Sunghwan) Always query for scene 0 for now
        scene_ids = torch.zeros(coords_world.shape[0], dtype=torch.long, device=coords_world.device)
        voxel_features = self.hash_voxel.query_feature(
            coords_world, scene_ids
        )
        
        decoded_features = self.implicit_decoder(
            voxel_features, coords_world
        )
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
        torch.save(
            dict(model=self.hash_voxel.state_dict(), epoch=epoch),
            os.path.join(folder, "hash_voxel.pt"),
        )
        torch.save(
            dict(model=self.implicit_decoder.state_dict(), epoch=epoch),
            os.path.join(folder, "implicit_decoder.pt"),
        )

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
    if cfg.test_model_dir is not None:
        tqdm.write("Testing the model...")
        pipeline.test()
        return
    pipeline.train()


if __name__ == "__main__":
    main()
