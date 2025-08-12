import torch
import torch.nn.functional as F
from torchvision import transforms

def get_visual_features_clip(model, x):
    """
    Extracts normalized visual features from a CLIP-based visual encoder.
    Returns features reshaped to (B, C, grid_size, grid_size).
    """
    vision_model = model.visual.trunk
    x = vision_model.forward_features(x)
    # x = vision_model.norm(x)
    x = vision_model.fc_norm(x)
    x = vision_model.head_drop(x)
    x = vision_model.head(x)

    # Remove CLS token, normalize, and reshape
    dense_features = x[:, 1:, :]
    dense_features = F.normalize(dense_features, dim=-1)
    num_patches = dense_features.shape[1]
    grid_size = int(num_patches**0.5)
    dense_features = dense_features.permute(0, 2, 1)
    dense_features = dense_features.reshape(x.shape[0], -1, grid_size, grid_size)
    return dense_features

def get_visual_features_dino(model, x):
    """
    x: (B, C, H, W)
    return: (B, 1024, H//14, W//14)
    """
    B, C, H, W = x.size()

    x = model.prepare_tokens_with_masks(x)

    for blk in model.blocks:                      # transformer encoder
        x = blk(x)
    x = model.norm(x)
    x = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H // 14, W // 14).contiguous()
    
    return x

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
        freq = 2**i
        pe.append(torch.sin(x * freq * torch.pi))
        pe.append(torch.cos(x * freq * torch.pi))
    return torch.cat(pe, dim=-1)


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
        feature_maps: [B, C, H_feat, W_feat].
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
    
    B, H_feat, W_feat = depth.shape  

    # Scale intrinsics
    scale_x = W_feat / float(original_size)
    scale_y = H_feat / float(original_size)
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    # Create pixel coordinate grid
    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat) + 0.5
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat) + 0.5
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
    
def rotary_pe_3d(x, coords, base: float = 10000.0, pos_scale: float = 2.0):
    four_d = (x.dim() == 4)
    if four_d:
        B0, H0, S, D = x.shape
        x = x.reshape(B0 * H0, S, D)
        coords = coords.unsqueeze(1).expand(B0, H0, S, 3).reshape(B0 * H0, S, 3)
        B = B0 * H0 
    else:
        B, S, D = x.shape
        B0, H0 = B, 1

    assert D % 6 == 0
    nb = D // 6
    
    coords_n = coords / pos_scale

    x_blocks = x.reshape(B, S, nb, 6)

    k = torch.arange(nb, device=x.device, dtype=torch.float32)
    theta = base ** (-k / float(nb))
    x_p, y_p, z_p = coords_n.unbind(dim=-1)

    ang_x = x_p.float().unsqueeze(-1) * theta
    ang_y = y_p.float().unsqueeze(-1) * theta
    ang_z = z_p.float().unsqueeze(-1) * theta

    cos_x, sin_x = torch.cos(ang_x), torch.sin(ang_x)
    cos_y, sin_y = torch.cos(ang_y), torch.sin(ang_y)
    cos_z, sin_z = torch.cos(ang_z), torch.sin(ang_z)

    cos_x = cos_x.to(x.dtype); sin_x = sin_x.to(x.dtype)
    cos_y = cos_y.to(x.dtype); sin_y = sin_y.to(x.dtype)
    cos_z = cos_z.to(x.dtype); sin_z = sin_z.to(x.dtype)

    b0, b1, b2, b3, b4, b5 = x_blocks.unbind(dim=-1)
    b0p = b0 * cos_x - b1 * sin_x; b1p = b0 * sin_x + b1 * cos_x
    b2p = b2 * cos_y - b3 * sin_y; b3p = b2 * sin_y + b3 * cos_y
    b4p = b4 * cos_z - b5 * sin_z; b5p = b4 * sin_z + b5 * cos_z

    out = torch.stack([b0p, b1p, b2p, b3p, b4p, b5p], dim=-1).reshape(B, S, D)
    if four_d:
        out = out.reshape(B0, H0, S, D)
    return out


def exp_decay_weights(dists: torch.Tensor, alpha: float) -> torch.Tensor:
    w = torch.exp(-alpha * dists.float())
    return w

# Basic image transform
transform = transforms.Compose(
    [
        transforms.Resize(
            size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        ),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
