import numpy as np
import torch
import torch.nn.functional as F
import logging
from .utils import all_grid_positions, normalize_coordinates, denormalize_coordinates
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureGridBase(torch.nn.Module):
    def __init__(self, d, fdim, bound, cell_size, name="grid", dtype=torch.float32):
        super().__init__()
        assert d == 2 or d == 3
        self.d = d
        self.fdim = fdim
        self.bound = bound
        self.cell_size = cell_size
        self.dtype = dtype
        self.name = name
        assert self.bound.shape == (d, 2)
    def interpolate(self, x):
        raise NotImplementedError
    def norm(self):
        raise NotImplementedError
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def lock(self):
        #logger.debug(f"Locked grid: {self.name}.")
        for param in self.parameters():
            param.requires_grad = False
    def unlock(self):
        #logger.debug(f"Unlocked grid: {self.name}.")
        for param in self.parameters():
            param.requires_grad = True
    def zero_features(self):
        raise NotImplementedError


class FeatureGrid(FeatureGridBase):
    """Regular (dense) 2D/3D feature grid, where each voxel stores a latent feature of
    dimension fdim.
    """
    def __init__(self, d, fdim, bound, cell_size, name="grid", dtype=torch.float32, initial_feature=None, init_stddev=0.0, second_order_grid_sample=False):
        super().__init__(d=d, fdim=fdim, bound=bound, cell_size=cell_size, name=name, dtype=dtype)
        grid_len = (self.bound[:,1] - self.bound[:,0]).cpu().numpy()
        grid_size = np.ceil(grid_len / cell_size).astype(int)
        # Initialize 2D or 3D feature grid
        # Notice that we change the order of grid_size due to the behavior of grid_sample
        # See discussion here: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997
        # In short, x-axis is along the width of an image and the y-axis is along its height.
        # This is the opposite of the coordinate system used in our implementation.
        if self.d == 2:
            feature_shape = (1, self.fdim, grid_size[1], grid_size[0])
        else:
            feature_shape = (1, self.fdim, grid_size[2], grid_size[1], grid_size[0])
        init_avail = initial_feature is not None
        if initial_feature is None:
            initial_feature = torch.randn(feature_shape, dtype=self.dtype) * init_stddev
        assert initial_feature.shape == feature_shape
        self.feature = torch.nn.Parameter(initial_feature)
        if second_order_grid_sample:
            # Use grid_sample that supports 2nd order derivatives
            import cuda_gridsample as cu
            self.grid_sample_func = cu.grid_sample_3d if self.d == 3 else cu.grid_sample_2d
        else:
            # Use standard pytorch implementation
            self.grid_sample_func = F.grid_sample
        logger.info(f"Regular {self.name}: cell_size={self.cell_size:.2f}, feature shape={feature_shape}, init_avail={init_avail}, 2nd_order_sample={second_order_grid_sample}, init_norm={initial_feature.norm():.1e}.")

    def interpolate(self, x):
        # Normalize query coordinates (required by F.grid_sample)
        x = normalize_coordinates(x, self.bound)
        if self.d == 2:
            N = x.shape[0]
            sample_coords = x.reshape(1, N, 1, 2)
            feats = self.grid_sample_func(
                self.feature,
                sample_coords, 
                align_corners=False,
                padding_mode='zeros'
            )[0, :, :, 0].transpose(0, 1)
        else:
            N = x.shape[0]
            sample_coords = x.reshape(1, N, 1, 1, 3)
            feats = self.grid_sample_func(
                self.feature,
                sample_coords,
                align_corners=False,
                padding_mode='zeros'
            )[0, :, :, 0, 0].transpose(0, 1)
        return feats

    def norm(self):
        return self.feature.norm()
    
    def zero_features(self):
        logger.info(f"Set grid features to zero: {self.name}.")
        with torch.no_grad():
            self.feature.copy_(torch.zeros_like(self.feature))
    
    def randn_features(self, std):
        logger.info(f"Set grid features to randn: {self.name}, std={std:.2e}.")
        with torch.no_grad():
            new_feat = torch.randn(self.feature.shape, dtype=self.dtype) * std
            self.feature.copy_(new_feat.to(self.feature))

    def vertex_positions(self, denormalize=True) -> torch.Tensor:
        """
        Args:
            denormalize (bool, optional): Whether to denormalize positions. Defaults to True.
        Returns:
            torch.Tensor: The 3D positions of grid vertices, unnormalized by default. 
        """
        pos_nrm = all_grid_positions(self.feature)
        pos_nrm = torch.flatten(pos_nrm.squeeze(), start_dim=0, end_dim=-2)  # N,3
        if denormalize:
            return denormalize_coordinates(pos_nrm, self.bound.to(pos_nrm))
        else:
            return pos_nrm
