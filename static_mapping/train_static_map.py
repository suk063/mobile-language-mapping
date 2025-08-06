import os.path
import random
import sys
from datetime import datetime

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from dacite import from_dict
from dataset import StaticMappingDataset
from omegaconf import OmegaConf
from ruamel import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from train_config import TrainConfig
from utils import get_3d_coordinates

from lang_mapping.mapper.mapper import VoxelHashTable
from lang_mapping.module import ImplicitDecoder
from lang_mapping.utils import get_visual_features
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader


class Pipeline:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
        torch.backends.cudnn.benchmark = not cfg.torch_deterministic

        device = cfg.device_decoder

        # Load datasets
        dataset = StaticMappingDataset(self.cfg.data)
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
            )[0].to(device)
            self.clip_model.eval()
        assert self.cfg.grid_cfg.name == "voxel_hash_table", "Only voxel_hash_table is supported in this pipeline"
        assert self.cfg.grid_cfg.voxel_hash_table is not None, "voxel_hash_table config must be provided"
        vht_cfg = self.cfg.grid_cfg.voxel_hash_table
        self.hash_voxel = VoxelHashTable(
            one_to_one=vht_cfg.one_to_one,
            resolution=vht_cfg.resolution,
            hash_table_size=vht_cfg.hash_table_size,
            feature_dim=vht_cfg.voxel_feature_dim,
            scene_bound_min=vht_cfg.scene_bound_min,
            scene_bound_max=vht_cfg.scene_bound_max,
            device=device,
        ).to(device)
        self.implicit_decoder = ImplicitDecoder(
            voxel_feature_dim=vht_cfg.voxel_feature_dim * vht_cfg.num_levels,
            hidden_dim=self.cfg.decoder_hidden_dim,
            output_dim=self.cfg.decoder_output_dim,
        ).to(device)

        # create optimizer
        params = list(self.hash_voxel.parameters())
        params += list(self.implicit_decoder.parameters())
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(params, **self.cfg.optimizer_kwargs)

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
            for batch in tqdm(self.train_loader, desc="Train", ncols=80, position=1, leave=False):
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
        assert self.cfg.test_model_dir is not None
        state = torch.load(
            os.path.join(self.cfg.test_model_dir, "hash_voxel.pt"),
            map_location=self.cfg.device_decoder,
        )
        self.hash_voxel.load_state_dict(state["model"])
        self.hash_voxel.eval()
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
        self.hash_voxel.train(training)
        self.implicit_decoder.train(training)

    def forward_model(self, batch: dict):
        device = self.cfg.device_decoder

        depth = batch["depth"].to(device) / 1000.0
        extrinsic = batch["extrinsic"].to(device)

        with torch.no_grad():
            if self.clip_model is None:
                visual_features = batch["clip"].to(device)
            else:
                rgb = batch["rgb"].float().to(device) / 255.0
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

            # bs, c, h, w -> n, c
            mask = (depth > 0.0).flatten()  # [bs, h, w] -> [bs * h * w]
            c = visual_features.shape[1]  # feature dim, e.g. 768 for EVA02-L-14
            visual_features = visual_features.permute(0, 2, 3, 1).reshape(-1, c)[mask]
            coords_world = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)[mask]

        voxel_features = self.hash_voxel.query_voxel_feature(coords_world)
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

        # --------------------------------------------
        dense_path = os.path.join(folder, "hash_voxel_dense.pt")
        sparse_path = os.path.join(folder, "hash_voxel_sparse.pt")
        self.hash_voxel.save_dense(dense_path)  # full state_dict
        self.hash_voxel.save_sparse(sparse_path)  # only accessed voxels

        torch.save(
            dict(model=self.implicit_decoder.state_dict(), epoch=epoch),
            os.path.join(folder, "implicit_decoder.pt"),
        )


def main():
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    cfg = from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))
    if cfg.test_model_dir is not None:
        # reload the config based on the test model directory
        test_model_dir = cfg.test_model_dir
        cfg = parse_cfg(default_cfg_path=os.path.join(test_model_dir, "../config.yaml"))
        cfg = from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))
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
