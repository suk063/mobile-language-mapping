import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_net import BaseNet
from .grid_modules import *
import logging
from typing import Optional
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
        self.init_grid(cfg)

    def init_grid(self, cfg):
        self.num_levels = cfg['grid']['n_levels']
        self.second_order_grid_sample = 'second_order_grid_sample' in cfg['grid'] and cfg['grid']['second_order_grid_sample']
        base_cell_size = cfg['grid']['base_cell_size']
        scale_factor = cfg['grid']['per_level_scale']
        self.fdim = cfg['grid']['feature_dim']
        self.features = nn.ModuleList()
        self.grid_type = cfg['grid']['type']
        n_scenes = cfg['grid']['n_scenes']
        self.cell_sizes = []

        for level in range(self.num_levels):
            cell_size = base_cell_size / (scale_factor**level)
            self.cell_sizes.append(cell_size)
            if level in self.initial_features.keys():
                init_feature = self.initial_features[level]
            else:
                init_feature = None
            if self.grid_type == 'regular':
                grid = FeatureGrid(
                    d = self.d,
                    fdim=self.fdim,
                    bound=self.bound,
                    cell_size=cell_size,
                    n_scenes=n_scenes,
                    name=f"feat-{level}",
                    dtype=self.dtype,
                    initial_feature=init_feature,
                    init_stddev=cfg['grid']['init_stddev'],
                    second_order_grid_sample=self.second_order_grid_sample
                )
            else:
                raise ValueError(f"Unknown grid type: {self.grid_type}!")
            self.features.append(grid)
        self.ignore_level_ = np.zeros(self.num_levels).astype(bool)

    def ignore_level(self, l):
        """Ignoring a feature level. The corresponding contribution from this level to the decoder will be set to zero.
        """
        self.ignore_level_[l] = True
        logger.warning(f"Ignore level: {self.ignore_level_}")

    def include_level(self, l):
        self.ignore_level_[l] = False
        logger.warning(f"Ignore level: {self.ignore_level_}")

    def lock_level(self, l):
        """Locking (fixing) the features at level l at the current value.
        """
        self.features[l].lock()

    def unlock_level(self, l):
        self.features[l].unlock()

    def lock_feature(self):
        for level in range(self.num_levels):
            self.lock_level(level)
    
    def unlock_feature(self):
        for level in range(self.num_levels):
            self.unlock_level(level)
    
    def print_feature_info(self):
        for level in range(self.num_levels):
            logger.info(f"Level {level} norm: {self.features[level].norm():.2f}")

    def zero_features(self):
        for grid in self.features:
            grid.zero_features()
    
    def randn_features(self, std):
        for grid in self.features:
            grid.randn_features(std)
    
    def query_feature(
        self,
        x: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
    ):
        assert x.ndim == 2, f"Invalid input coords shape {x.shape}!"
        assert x.size(1) in {self.d, self.d + 1}

        if scene_ids is not None:
            if x.size(1) != self.d:
                raise ValueError("When scene_ids is given, x must be xyz only")
            x = torch.cat([scene_ids, x], dim=1)
        else:
            if x.size(1) == self.d:
                zeros = x.new_zeros(x.size(0), 1, dtype=torch.long)
                x = torch.cat([zeros, x], dim=1)
            elif x.size(1) == self.d + 1:
                pass    
            
        # Interpolate feature grid
        if self.grid_type == 'regular':
            feats = grid_interp_regular(self.features, x, self.ignore_level_)
        return feats
    
    def forward(self, x: torch.Tensor, noise_std=0):
        """Predict value for coordinates x.

        Args:
            x (torch.tensor): Query coordinates with shape 
            (N, d) or (H, W, d), i.e., N coordinates or HxW coordinates.
            Each coordinate must fall within the bound.

        Returns:
            _type_: _description_
        """
        # Interpolate feature grid
        feats = self.query_feature(x)

        # Pass thru decoder
        pred = grid_decode(feats, x, self.decoder, self.pos_invariant)
        if noise_std > 0:
            noise = torch.randn(pred.shape, device=x.device) * noise_std
            pred = pred + noise
        return pred
    
    def params_for_features(self, stop_level=None):
        if stop_level is None:
            stop_level = self.num_levels
        assert stop_level <= self.num_levels
        params = []
        for level in range(stop_level):
            params += list(self.features[level].parameters())
        return params
    
    def params_at_level(self, level):
        params = []
        target_levels = [level] if level < self.num_levels else range(self.num_levels)
        for l in target_levels:
            params += list(self.features[l].parameters())
            params += list(self.feature_stability[l].parameters())
        # Always append decoder, if not fixed
        if not self.decoder_fixed:
            params += list(self.decoder.parameters())
        return params