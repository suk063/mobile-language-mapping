import torch
from typing import Optional
import transforms3d as t3d
import sapien

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
    """
    h, w = depth.shape[0], depth.shape[1]
    device = depth.device
    dtype = intrinsic.dtype
    depth = depth.to(dtype)
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
