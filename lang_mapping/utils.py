import torch
import torch.nn.functional as F
from torchvision import transforms

def get_visual_features(model, x):
    """
    Extracts normalized visual features from a CLIP-based visual encoder.
    Returns features reshaped to (B, C, grid_size, grid_size).
    """
    vision_model = model.visual.trunk
    x = vision_model.forward_features(x)
    x = vision_model.norm(x)
    x = vision_model.fc_norm(x)
    x = vision_model.head_drop(x)
    x = vision_model.head(x)

    # Remove CLS token, normalize, and reshape
    dense_features = x[:, 1:, :]
    dense_features = F.normalize(dense_features, dim=-1)
    num_patches = dense_features.shape[1]
    grid_size = int(num_patches ** 0.5)
    dense_features = dense_features.permute(0, 2, 1)
    dense_features = dense_features.reshape(x.shape[0], -1, grid_size, grid_size)
    return dense_features


def positional_encoding(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    """
    Positional encoding for input x using frequencies 2^i for i in [0..L-1].
    Args:
        x: [N, 3] or [B*N, 3].
        L: Number of frequency bands.
    Returns:
        Encoded tensor of shape [N, 2*L*3].
    """
    pe = []
    for i in range(L):
        freq = 2 ** i
        pe.append(torch.sin(x * freq * torch.pi))
        pe.append(torch.cos(x * freq * torch.pi))
    return torch.cat(pe, dim=-1)

def get_3d_coordinates(
    feature_maps: torch.Tensor,
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
        feature_maps: [B, C, H_feat, W_feat].
        depth: [B, H_feat, W_feat] or [B, 1, H_feat, W_feat].
        camera_extrinsic: [B, 1, 3, 4], world->cam transform (None if absent).
        fx, fy, cx, cy: Camera intrinsics for original_size x original_size.
        original_size: Original image size (default=224).
    Returns:
        coords_world or coords_cam: [B, 3, H_feat, W_feat].
    """
    device = feature_maps.device
    B, C, H_feat, W_feat = feature_maps.shape

    # Adjust depth shape if needed
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    # Scale intrinsics
    scale_x = W_feat / float(original_size)
    scale_y = H_feat / float(original_size)
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    # Create pixel coordinate grid
    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat)
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat)
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
        ones_row = torch.tensor([0, 0, 0, 1], device=device, 
                                dtype=camera_extrinsic.dtype).view(1, 1, 4)
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

def rotary_pe_3d(
    x: torch.Tensor,      # [B, S, D]
    coords: torch.Tensor, # [B, S, 3]
    base: float = 10000.0
) -> torch.Tensor:
    """
    3D rotary positional embedding. Splits D into blocks of 6, each rotated by angles
    derived from coords along x, y, and z.
    Args:
        x: [B, S, D], with D % 6 == 0.
        coords: [B, S, 3].
        base: Base for frequency calculation.
    Returns:
        [B, S, D], same shape as x.
    """
    B, S, D = x.shape
    assert D % 6 == 0, "D must be a multiple of 6"
    num_block = D // 6

    # Frequency factors
    k_idx = torch.arange(num_block, device=x.device, dtype=x.dtype)
    theta_k = 1.0 / (base ** (k_idx / (D / 6)))

    # Reshape into blocks of size 6
    x_splitted = x.view(B, S, num_block, 6)
    x_p, y_p, z_p = coords[..., 0], coords[..., 1], coords[..., 2]

    out_blocks = []
    for k in range(num_block):
        block = x_splitted[:, :, k, :]   # [B, S, 6]
        x_angle = x_p * theta_k[k]
        y_angle = y_p * theta_k[k]
        z_angle = z_p * theta_k[k]

        b0, b1, b2, b3, b4, b5 = (block[..., i] for i in range(6))
        cos_x, sin_x = torch.cos(x_angle), torch.sin(x_angle)
        cos_y, sin_y = torch.cos(y_angle), torch.sin(y_angle)
        cos_z, sin_z = torch.cos(z_angle), torch.sin(z_angle)

        # Rotate each pair around X, Y, Z
        b0_ = b0 * cos_x - b1 * sin_x
        b1_ = b0 * sin_x + b1 * cos_x
        b2_ = b2 * cos_y - b3 * sin_y
        b3_ = b2 * sin_y + b3 * cos_y
        b4_ = b4 * cos_z - b5 * sin_z
        b5_ = b4 * sin_z + b5 * cos_z

        out_blocks.append(torch.stack([b0_, b1_, b2_, b3_, b4_, b5_], dim=-1))

    return torch.stack(out_blocks, dim=2).view(B, S, D)

def chamfer_3d(pred_points, gt_points, threshold=1):
    dist = torch.cdist(pred_points, gt_points, p=2)

    row2col_vals, _ = dist.min(dim=2)  
    col2row_vals, _ = dist.min(dim=1)


    row_mask = (row2col_vals <= threshold)
    col_mask = (col2row_vals <= threshold)
    
    valid_row2col = row2col_vals[row_mask]
    valid_col2row = col2row_vals[col_mask]

    if valid_row2col.numel() > 0:
        row_loss = valid_row2col.mean()
    else:
        row_loss = torch.tensor(0.0, device=dist.device)

    if valid_col2row.numel() > 0:
        col_loss = valid_col2row.mean()
    else:
        col_loss = torch.tensor(0.0, device=dist.device)

    chamfer_loss_val = row_loss + col_loss
    return chamfer_loss_val

def chamfer_3d_weighted(pred_points, gt_points, pred_weights, threshold=1.0):
    """
    A weighted Chamfer 3D function.
    Args:
        pred_points:    (B, N_pred, 3), predicted points.
        gt_points:      (B, N_gt, 3), ground-truth points.
        pred_weights:   (B, N_pred), scalar weights per predicted point.
        threshold:      distance threshold to mask out large distances (optional).
    Returns:
        Weighted Chamfer distance (scalar).
    """
    dist = torch.cdist(pred_points, gt_points, p=2)  # (B, N_pred, N_gt)

    # row2col: distance from each predicted point to its nearest GT
    row2col_vals, _ = dist.min(dim=2)  # (B, N_pred)

    # col2row: distance from each GT to its nearest predicted point
    col2row_vals, _ = dist.min(dim=1)  # (B, N_gt)

    # Apply threshold mask (optional)
    row_mask = (row2col_vals <= threshold)
    col_mask = (col2row_vals <= threshold)

    # Weighted row2col (pred->GT)
    # Only consider distances within threshold
    valid_row2col = row2col_vals[row_mask]
    # Multiply the corresponding weights
    valid_weights = pred_weights[row_mask]
    row_loss = (valid_row2col * valid_weights).mean() if valid_row2col.numel() > 0 else torch.tensor(0.0, device=dist.device)

    return row_loss

# Basic image transform
transform = transforms.Compose([
    transforms.Resize(
        size=224,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    ),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])
