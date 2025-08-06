from typing import Optional

import sapien
import torch
import transforms3d as t3d

oRc = torch.tensor(  # rotation matrix from camera to optical frame
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def extrinsic_to_cam_pose(extrinsic: torch.Tensor) -> torch.Tensor:
    global oRc
    oRc = oRc.to(extrinsic.device)
    cam_pose = torch.zeros_like(extrinsic)
    cam_pose[:3, :3] = extrinsic[:3, :3].T @ oRc
    cam_pose[:3, 3] = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    if cam_pose.shape[0] == 4:
        cam_pose[3, 3] = 1.0  # homogeneous coordinate
    return cam_pose


def cam_pose_to_extrinsic(cam_pose: torch.Tensor) -> torch.Tensor:
    global oRc
    oRc = oRc.to(cam_pose.device)
    extrinsic = torch.zeros_like(cam_pose)
    extrinsic[:3, :3] = oRc @ cam_pose[:3, :3].T
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ cam_pose[:3, 3]
    if extrinsic.shape[0] == 4:
        extrinsic[3, 3] = 1.0
    return extrinsic


def transform_matrix_to_sapien_pose(matrix: torch.Tensor) -> sapien.Pose:
    matrix_np = matrix.detach().cpu().numpy()
    q = t3d.quaternions.mat2quat(matrix_np[:3, :3])
    p = matrix_np[:3, 3]
    return sapien.Pose(p=p, q=q)


def depth_to_positions(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: Optional[torch.Tensor] = None,
    depth_scale: float = 1.0,
) -> torch.Tensor:
    """
    Convert depth image to 3D point cloud.

    Args:
        depth (torch.Tensor): Depth image of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        extrinsic (torch.Tensor, optional): Camera extrinsic matrix of shape (4, 4) or (3, 4).
            If provided, the points will be transformed to world coordinates.
        depth_scale (float): Scale factor for depth values. Default is 1.0.

    Returns:
        torch.Tensor: 3D points in world coordinates of shape (H, W, 3).
    """
    h, w = depth.shape[0], depth.shape[1]
    device = depth.device
    dtype = intrinsic.dtype
    depth = depth.to(dtype).view(h, w)
    if depth_scale != 1.0:
        depth = depth * depth_scale
    # Create a grid of pixel coordinates
    pix = torch.stack(
        [
            *torch.meshgrid(  # +0.5 to get pixel centers instead of corners
                torch.arange(h, dtype=dtype) + 0.5,
                torch.arange(w, dtype=dtype) + 0.5,
                indexing="xy",
            ),
            torch.ones(h, w, dtype=dtype),  # homogeneous coordinate
        ],
        dim=-1,  # shape (H, W, 3)
    ).to(device)
    # pixel coordinates to normalized device coordinates
    # Note: This assumes the camera's principal point is at the center of the image
    pix = pix @ torch.linalg.inv(intrinsic).T  # shape (H, W, 3)
    # Scale by depth to get 3D points in camera coordinates
    pix = pix * depth[..., torch.newaxis]  # shape (H, W, 3)
    if extrinsic is not None:
        # Now we have 3D points in camera coordinates, we need to transform them to world coordinates
        rot = extrinsic[:3, :3]
        t = -rot.T @ extrinsic[:3, 3]
        pix = pix @ rot + t
    return pix  # shape (H, W, 3) now in world coordinates


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
