import numpy as np
import torch
import torch.nn as nn
from lang_mapping.grid_net.base_net import BaseNet
from lang_mapping.grid_net.grid_modules import *
import logging
from lang_mapping.grid_net.utils import grid_interp_regular, grid_interp_VM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GridNet(BaseNet):
    def __init__(self, cfg: dict, dtype=torch.float32, initial_features=dict()):
        super(GridNet, self).__init__(cfg, "cpu", dtype)
        self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.initial_features = initial_features  # Allow to use initial guesses for features
        self.n_scenes = cfg["n_scenes"]
        self.init_grid(cfg)

    def distribute_to_devices(self):
        for scene_id in range(self.n_scenes):
            device = self.devices[scene_id % len(self.devices)]
            if self.grid_type == "regular":
                self.features[scene_id].to(device)
            elif self.grid_type == "VM":
                self.features[scene_id].to(device)
                self.bases[scene_id].to(device)

    def init_grid(self, cfg):
        self.num_levels = cfg["n_levels"]
        base_cell_size = cfg["base_cell_size"]
        scale_factor = cfg["per_level_scale"]
        self.fdim = cfg["feature_dim"]
        self.features = nn.ModuleList()
        self.bases = nn.ModuleList()
        self.grid_type = cfg["type"]
        self.cell_sizes = []
        self.ignore_level_ = [np.zeros(self.num_levels).astype(bool) for _ in range(self.n_scenes)]

        for scene_idx in range(self.n_scenes):
            scene_features = nn.ModuleList()
            scene_bases = nn.ModuleList()
            scene_cell_sizes = []  # Store cell sizes per scene if they could differ, otherwise self.cell_sizes is fine
            initial_features_scene = self.initial_features.get(scene_idx, {})  # Get initial features for this scene

            for level in range(self.num_levels):
                cell_size = base_cell_size / (scale_factor**level)
                scene_cell_sizes.append(cell_size)  # Store if needed, might be redundant if same for all scenes

                init_feature = initial_features_scene.get(level, None)  # Get initial feature for this scene and level

                if self.grid_type == "regular":
                    grid = FeatureGrid(
                        d=self.d,
                        fdim=self.fdim,
                        bound=self.bound,
                        cell_size=cell_size,
                        name=f"scene{scene_idx}-feat-{level}",  # Naming includes scene_id
                        dtype=self.dtype,
                        initial_feature=init_feature,
                        init_stddev=cfg["init_stddev"],
                    )
                    basis = None
                elif self.grid_type == "VM":
                    grid = FeatureGridVM(
                        d=self.d,
                        fdim=self.fdim,
                        bound=self.bound,
                        cell_size=cell_size,
                        name=f"{level}",
                        dtype=self.dtype,
                        init_stddev=cfg["init_stddev"],
                        rank=cfg["rank"],
                    )
                    basis = BasisVM(
                        fdim=self.fdim,
                        name=f"{level}",
                        rank=cfg["rank"],
                        init_stddev=cfg["init_stddev"],
                        dtype=self.dtype,
                        pretrained_path=None,
                        no_optimize=False,
                    )
                else:
                    raise ValueError(f"Unknown grid type: {self.grid_type}!")
                scene_features.append(grid)
                scene_bases.append(basis)

            self.features.append(scene_features)
            self.bases.append(scene_bases)

            if scene_idx == 0:  # Assuming cell sizes are the same across scenes
                self.cell_sizes = scene_cell_sizes

    def query_feature(self, x: torch.Tensor, scene_id: torch.Tensor):
        assert x.ndim == 2, f"Invalid input coords shape {x.shape}!"
        assert x.shape[-1] == self.d
        assert x.shape[0] == scene_id.shape[0], "Mismatch between number of points and scene IDs"
        assert scene_id.ndim == 1 or scene_id.shape[1] == 1, "scene_id should be (N,) or (N, 1)"

        scene_id = scene_id.squeeze()  # Ensure shape (N,)
        N = x.shape[0]
        output_dim = self.num_levels * self.fdim
        all_feats = torch.zeros(N, output_dim, device=x.device, dtype=self.dtype)

        unique_scenes = torch.unique(scene_id)

        for s_id_tensor in unique_scenes:
            s_id = s_id_tensor.item()
            if not (0 <= s_id < self.n_scenes):
                logger.error(f"Invalid scene_id {s_id} encountered in query batch.")
                continue  # Skip points with invalid scene IDs or handle as error

            mask = scene_id == s_id_tensor
            device = self.devices[s_id % len(self.devices)]

            x_scene = x[mask]

            if x_scene.shape[0] == 0:  # Skip if no points for this scene_id
                continue

            # Select the feature grids and ignore mask for the current scene
            features_scene = self.features[s_id]
            ignore_level_scene = self.ignore_level_[s_id]
            x_scene = x_scene.to(device)

            # Interpolate feature grid for the current scene
            if self.grid_type == "regular":
                feats_scene = grid_interp_regular(features_scene, x_scene, ignore_level_scene)
            elif self.grid_type == "VM":
                bases_scene = self.bases[s_id]
                feats_scene = grid_interp_VM(features_scene, bases_scene, x_scene, ignore_level_scene)
            else:
                # Handle other grid types if necessary
                raise NotImplementedError(f"Interpolation for grid type {self.grid_type} not implemented")

            # Place the results back into the main tensor
            all_feats[mask] = feats_scene.to(all_feats.device)

        return all_feats
