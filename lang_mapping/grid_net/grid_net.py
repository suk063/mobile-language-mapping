import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_net import BaseNet
from .grid_modules import *
import logging
from .utils import grid_interp_regular, grid_decode
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridNet(BaseNet):
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
        initial_features = dict()
    ):
        super(GridNet, self).__init__(cfg, device, dtype)        
        self.initial_features = initial_features  # Allow to use initial guesses for features
        self.n_scenes = cfg['grid']['n_scenes']
        self.init_grid(cfg)

    def init_grid(self, cfg):
        self.num_levels = cfg['grid']['n_levels']
        self.second_order_grid_sample = 'second_order_grid_sample' in cfg['grid'] and cfg['grid']['second_order_grid_sample']
        base_cell_size = cfg['grid']['base_cell_size']
        scale_factor = cfg['grid']['per_level_scale']
        self.fdim = cfg['grid']['feature_dim']
        self.features = nn.ModuleList()
        self.grid_type = cfg['grid']['type']
        self.cell_sizes = []
        self.ignore_level_ = [np.zeros(self.num_levels).astype(bool) for _ in range(self.n_scenes)]

        for scene_idx in range(self.n_scenes):
            scene_features = nn.ModuleList()
            scene_cell_sizes = [] # Store cell sizes per scene if they could differ, otherwise self.cell_sizes is fine
            initial_features_scene = self.initial_features.get(scene_idx, {}) # Get initial features for this scene

            for level in range(self.num_levels):
                cell_size = base_cell_size / (scale_factor**level)
                scene_cell_sizes.append(cell_size) # Store if needed, might be redundant if same for all scenes

                init_feature = initial_features_scene.get(level, None) # Get initial feature for this scene and level

                if self.grid_type == 'regular':
                    grid = FeatureGrid(
                        d = self.d,
                        fdim=self.fdim,
                        bound=self.bound,
                        cell_size=cell_size,
                        name=f"scene{scene_idx}-feat-{level}", # Naming includes scene_id
                        dtype=self.dtype,
                        initial_feature=init_feature,
                        init_stddev=cfg['grid']['init_stddev'],
                        second_order_grid_sample=self.second_order_grid_sample
                    )
                    grid = grid.to(self.device) # Move grid to the correct device
                else:
                    raise ValueError(f"Unknown grid type: {self.grid_type}!")
                scene_features.append(grid)

            self.features.append(scene_features)
            if scene_idx == 0: # Assuming cell sizes are the same across scenes
                self.cell_sizes = scene_cell_sizes
    
    def query_feature(self, x: torch.Tensor, scene_id: torch.Tensor):
        assert x.ndim == 2, f"Invalid input coords shape {x.shape}!"
        assert x.shape[-1] == self.d
        assert x.shape[0] == scene_id.shape[0], "Mismatch between number of points and scene IDs"
        assert scene_id.ndim == 1 or scene_id.shape[1] == 1, "scene_id should be (N,) or (N, 1)"
        
        scene_id = scene_id.squeeze() # Ensure shape (N,)
        N = x.shape[0]
        output_dim = self.num_levels * self.fdim
        all_feats = torch.zeros(N, output_dim, device=x.device, dtype=self.dtype)

        unique_scenes = torch.unique(scene_id)

        for s_id_tensor in unique_scenes:
            s_id = s_id_tensor.item()
            if not (0 <= s_id < self.n_scenes):
                logger.error(f"Invalid scene_id {s_id} encountered in query batch.")
                continue # Skip points with invalid scene IDs or handle as error

            mask = (scene_id == s_id_tensor)
            x_scene = x[mask]

            if x_scene.shape[0] == 0: # Skip if no points for this scene_id
                continue

            # Select the feature grids and ignore mask for the current scene
            features_scene = self.features[s_id]
            ignore_level_scene = self.ignore_level_[s_id]

            # Interpolate feature grid for the current scene
            if self.grid_type == 'regular':
                feats_scene = grid_interp_regular(features_scene, x_scene, ignore_level_scene)
            else:
                # Handle other grid types if necessary
                raise NotImplementedError(f"Interpolation for grid type {self.grid_type} not implemented")

            # Place the results back into the main tensor
            all_feats[mask] = feats_scene

        return all_feats
    
    def snapshot(self):
        """
        Take a deep copy of each scene and level feature grid onto the CPU.
        Call this method once before training to record the initial feature state.
        """
        self._init_snapshots = []
        logger.info("Creating feature snapshots on CPU...")
        with torch.no_grad():
            for s_id in range(self.n_scenes):
                scene_snapshot = []
                for lvl in range(self.num_levels):
                    grid = self.features[s_id][lvl]
                    # Store a detached, CPU clone of the feature tensor
                    scene_snapshot.append(grid.feature.detach().cpu().clone())
                self._init_snapshots.append(scene_snapshot)
        logger.info(
            f"Snapshots created for {self.n_scenes} scenes and {self.num_levels} levels."
        )

    @torch.no_grad()
    def dump_changed_centers(
        self,
        root_dir: str = "queried_cell_centers",
        eps: float = 1e-7,
    ):

        # Ensure that snapshots exist
        if not hasattr(self, "_init_snapshots") or self._init_snapshots is None:
            logger.error(
                "Snapshot not found! Call snapshot() before dump_changed_centers()."
            )
            return

        os.makedirs(root_dir, exist_ok=True)
        # Minimum bound per dimension for world coordinate conversion
        bound_min = self.bound[:, 0].cpu()

        total_changed = 0

        # Loop over each feature level
        for lvl in range(self.num_levels):
            level_out = {}
            count_msgs = []
            cell_size = self.cell_sizes[lvl]

            # Loop over each scene
            for s_id in range(self.n_scenes):
                grid = self.features[s_id][lvl]
                snap = self._init_snapshots[s_id][lvl]
                curr = grid.feature.detach().cpu()

                # Skip if the shapes do not match
                if snap.shape != curr.shape:
                    logger.warning(
                        f"Scene {s_id}, Level {lvl} shape mismatch: "
                        f"{snap.shape} vs {curr.shape}, skipping."
                    )
                    continue

                # Compute L2 norm of the feature difference along channel axis
                delta = (curr.float() - snap.float()).norm(dim=1).squeeze(0)

                # Find voxel indices where change exceeds threshold
                idx_changed = (delta > eps).nonzero(as_tuple=False)
                if idx_changed.numel() == 0:
                    continue

                # Extract spatial indices (z,y,x) or (y,x) depending on grid dimension
                voxel_idx = (
                    idx_changed
                    if idx_changed.shape[1] == self.d
                    else idx_changed[:, -self.d:]
                )

                # Convert voxel indices to world-space centers
                centers = (voxel_idx.float() + 0.5) * cell_size + bound_min
                arr = centers.numpy().astype(np.float32)

                # Group by scene and record counts
                level_out[str(s_id)] = arr
                count_msgs.append(f"scene {s_id}: {len(arr):>5d}")
                total_changed += len(arr)

            # Save or report no changes for this level
            if not level_out:
                print(f"Level {lvl:>2d}: (no changes)")
            else:
                filename = os.path.join(
                    root_dir, f"level{lvl}_centers.npz"
                )
                np.savez_compressed(filename, **level_out)
                print(
                    f"Level {lvl:>2d} → " + ", ".join(count_msgs)
                )

        # Summary of all changes
        print(
            f"[GridNet] {total_changed} changed‑centers saved to '{root_dir}'."
        )