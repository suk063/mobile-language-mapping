import torch
import numpy as np

def grid_interp_regular(reg_grids, x, ignore_level=None):
    """Interpolate from a list of regular feature grids.

    Args:
        reg_grids: Each element is an instance of FeatureGrid.
        x: Unnormalized query coordinates
        ignore_level: list of bools indicating which level to ignore. Defaults to None.

    Returns:
        Interpolated features, concatenated across the levels.
    """
    num_levels = len(reg_grids)
    if ignore_level is None:
        ignore_level = np.zeros(num_levels).astype(bool)
    level_feats = []
    for level in range(num_levels):
        feats = reg_grids[level].interpolate(x)
        if not ignore_level[level]:
            level_feats.append(feats)
        else:
            level_feats.append(torch.zeros_like(feats))
    return torch.cat(level_feats, dim=1)

def grid_decode(feats, x, decoder=None, pos_invariant=True):
    assert feats.ndim == 2
    
    # Pass through decoder
    if decoder is not None:
        if pos_invariant:
            inputs = feats
        else:
            inputs = torch.cat((feats, x), dim=1)
        preds = decoder(inputs)
    else:
        # In this case, the grid directly gives prediction
        preds = feats

    return preds

def all_grid_positions(features):
    """Return the 3D coordinates of the centers of a 3D regular grid.
    Return shape is (1, Z, Y, X, 3)
    """
    B, C, D, H, W = features.shape
    # D, H, W correspond to the z, y and x dimensions respectively
    half_dx = 0.5 / W
    half_dy = 0.5 / H
    half_dz = 0.5 / D
    xs = 2 * torch.linspace(half_dx, 1 - half_dx, W) - 1.
    ys = 2 * torch.linspace(half_dy, 1 - half_dy, H) - 1.
    zs = 2 * torch.linspace(half_dz, 1 - half_dz, D) - 1.
    xv, yv, zv = torch.meshgrid([xs, ys, zs])
    grid = torch.stack((zv, yv, xv), axis=-1)  
    return grid.unsqueeze(0)

def normalize_coordinates(queries: torch.tensor, bounds: torch.tensor):
    """
    Normalize coordinates to be between [-1, 1] based on given bounds.

    Args:
        queries (torch.Tensor): Tensor of shape (N, d) or (H, W, d) representing coordinates in d dimensions.
        bounds (torch.Tensor): Tensor of shape (d, 2) where each row represents the [min, max] bounds for each dimension.

    Returns:
        torch.Tensor: Normalized coordinates with values between [-1, 1]. The shape of the output will match the shape of the input queries.
    """
    d = bounds.shape[0]
    assert queries.shape[-1] == d

    # Determine the number of dimensions in the queries
    if queries.dim() == 2:
        # 2D case: shape (N, d)
        bounds_min = bounds[:, 0].view(1, -1)  # Shape (1, d)
        bounds_max = bounds[:, 1].view(1, -1)  # Shape (1, d)
    elif queries.dim() == 3:
        # 3D case: shape (H, W, d)
        bounds_min = bounds[:, 0].view(1, 1, -1)  # Shape (1, 1, d)
        bounds_max = bounds[:, 1].view(1, 1, -1)  # Shape (1, 1, d)
    else:
        raise ValueError("queries tensor must be either 2D or 3D")

    # Normalize the queries
    normalized_queries = 2 * (queries - bounds_min) / (bounds_max - bounds_min) - 1
    
    return normalized_queries

def denormalize_coordinates(normalized_queries: torch.Tensor, bounds: torch.Tensor):
    """
    Denormalize coordinates from the range [-1, 1] to the original coordinate range based on given bounds.

    Args:
        normalized_queries (torch.Tensor): Tensor of shape (N, d) or (H, W, d) with normalized coordinates in the range [-1, 1].
        bounds (torch.Tensor): Tensor of shape (d, 2) where each row represents the [min, max] bounds for each dimension.

    Returns:
        torch.Tensor: Denormalized coordinates in the original coordinate range. The shape of the output matches the input normalized_queries.
    """
    d = bounds.shape[0]
    assert normalized_queries.shape[-1] == d

    # Determine the number of dimensions in the normalized_queries
    if normalized_queries.dim() == 2:
        # 2D case: shape (N, d)
        bounds_min = bounds[:, 0].view(1, -1)  # Shape (1, d)
        bounds_max = bounds[:, 1].view(1, -1)  # Shape (1, d)
    elif normalized_queries.dim() == 3:
        # 3D case: shape (H, W, d)
        bounds_min = bounds[:, 0].view(1, 1, -1)  # Shape (1, 1, d)
        bounds_max = bounds[:, 1].view(1, 1, -1)  # Shape (1, 1, d)
    else:
        raise ValueError("normalized_queries tensor must be either 2D or 3D")

    # Denormalize the queries
    original_queries = (normalized_queries + 1) / 2 * (bounds_max - bounds_min) + bounds_min

    return original_queries