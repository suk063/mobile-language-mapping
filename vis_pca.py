import torch
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import argparse
import matplotlib.cm as cm
import torch.nn.functional as F
import open_clip
from pathlib import Path

CAMERA_POSE = np.array(
[[ 0.504749,  0.84679 ,  0.167855,  0.709454],
 [ 0.813701, -0.40175 , -0.420105, -0.93102 ],
 [-0.288305,  0.348631, -0.891816, 15.656313],
 [ 0.      ,  0.      ,  0.      ,  1.      ]], dtype=float)

def _visualize_with_pose(pcd, window_name, camera_pose=None, point_size: float = 2.0):
    """Helper to visualize a point cloud with a specific camera pose."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name, width=1920, height=1080)
    vis.add_geometry(pcd)

    out_dir = Path("o3d_frames")
    out_dir.mkdir(exist_ok=True)

    # Initialize point size
    try:
        ro = vis.get_render_option()
        ro.point_size = float(point_size)
    except Exception:
        pass

    if camera_pose is not None:
        vc = vis.get_view_control()
        cam_params = vc.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = camera_pose
        vc.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    def print_pose_callback(v):
        extrinsic = v.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
        print("\n# ───── copy below ─────\nCAMERA_POSE = np.array(\n"
              + np.array2string(extrinsic, separator=", ", precision=6)
              + ", dtype=float)\n# ───── copy above ─────\n")
        return False

    def save_image_callback(v):
        filename = window_name.replace(" ", "_").replace("'", "").replace('"', '')
        save_path = out_dir / f"{filename}.png"
        v.capture_screen_image(str(save_path), do_render=True)
        print(f"\n[Open3D] saved screenshot to ⇒ {save_path}")
        return False

    def increase_point_size(v):
        try:
            ro_local = v.get_render_option()
            ro_local.point_size = min(ro_local.point_size + 0.5, 20.0)
            print(f"[Open3D] point size: {ro_local.point_size:.2f}")
        except Exception:
            pass
        return False

    def decrease_point_size(v):
        try:
            ro_local = v.get_render_option()
            ro_local.point_size = max(ro_local.point_size - 0.5, 0.1)
            print(f"[Open3D] point size: {ro_local.point_size:.2f}")
        except Exception:
            pass
        return False

    vis.register_key_callback(ord("P"), print_pose_callback)
    vis.register_key_callback(ord("p"), print_pose_callback)
    vis.register_key_callback(ord("S"), save_image_callback)
    vis.register_key_callback(ord("s"), save_image_callback)
    # Point size controls
    vis.register_key_callback(ord("["), decrease_point_size)
    vis.register_key_callback(ord("]"), increase_point_size)
    vis.register_key_callback(ord("-"), decrease_point_size)
    vis.register_key_callback(ord("="), increase_point_size)
    vis.register_key_callback(ord("+"), increase_point_size)

    print("\n[Open3D] Controls: P to print camera pose, S to save PNG, [-]/[=] or [ ] to adjust point size.")

    vis.run()
    vis.destroy_window()

def _get_module_device(module: torch.nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")

def _minmax(x: np.ndarray,
            tau: float = 0.2,
            eps: float = 1e-8) -> np.ndarray:
    x_shift = (x - x.max()) / tau
    y = np.exp(x_shift)
    return (y - y.min()) / (y.max() - y.min() + eps)

def calculate_features(static_map, implicit_decoder, points_path):
    pcd = o3d.io.read_point_cloud(points_path)
    points_np = np.asarray(pcd.points).astype(np.float32)
    coords_np = points_np

    colors_np = None
    if pcd.has_colors():
        colors_np = np.asarray(pcd.colors)

    batch_size = 500000
    num_points = len(points_np)
    map_device = _get_module_device(static_map)

    all_feats = []
    print(f"Processing {num_points:,} points for voxel features in batches of {batch_size:,}...")
    for i in range(0, num_points, batch_size):
        batch_coords_np = points_np[i:i+batch_size]
        coords_torch = torch.from_numpy(batch_coords_np).to(map_device)
        with torch.no_grad():
            feats = static_map.query_voxel_feature(coords_torch)
            all_feats.append(feats.cpu())
        print(f"  Processed batch {i//batch_size + 1}/{ -(-num_points // batch_size) }")
    feats_torch = torch.cat(all_feats, dim=0)

    implicit_decoder = implicit_decoder.to(map_device)
    implicit_decoder.eval()
    all_decoded_feats = []
    print(f"\nProcessing {num_points:,} points for decoder features in batches of {batch_size:,}...")
    for i in range(0, num_points, batch_size):
        batch_coords_np = points_np[i:i+batch_size]
        coords_torch = torch.from_numpy(batch_coords_np).to(map_device)
        batch_feats_torch = feats_torch[i:i+batch_size].to(map_device)
        with torch.no_grad():
            decoded_feats = implicit_decoder(batch_feats_torch, coords_torch)
            all_decoded_feats.append(decoded_feats.cpu())
        print(f"  Processed batch {i//batch_size + 1}/{ -(-num_points // batch_size) }")
    
    decoded_feats_np = torch.cat(all_decoded_feats, dim=0).numpy()
    
    return coords_np, decoded_feats_np, colors_np

def visualize_raw_rgb(coords_np, colors_np, point_size: float):
    if colors_np is None:
        print("\nPoint cloud has no colors, skipping raw RGB visualization.")
        return

    # Filter points for visualization
    mask = coords_np[:, 2] < 2.5
    viz_coords_np = coords_np[mask]
    viz_colors_np = colors_np[mask]
    print(f"\nVisualizing {len(viz_coords_np)} points for Raw RGB (z < 2.5).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(viz_coords_np)
    pcd.colors = o3d.utility.Vector3dVector(viz_colors_np)
    _visualize_with_pose(pcd, "Raw RGB Visualization", CAMERA_POSE, point_size)

def visualize_pca(coords_np, decoded_feats_np, point_size: float):
    pca_decoder = PCA(n_components=3)
    scaler = MinMaxScaler()
    decoded_feats_pca = pca_decoder.fit_transform(decoded_feats_np)
    print("\nDecoder Features PCA explained variance ratio:", pca_decoder.explained_variance_ratio_)
    print("Sum of explained variance ratio:", pca_decoder.explained_variance_ratio_.sum())

    decoded_feats_pca_norm = scaler.fit_transform(decoded_feats_pca)

    # Filter points for visualization
    mask = coords_np[:, 2] < 2.5
    viz_coords_np = coords_np[mask]
    viz_colors_np = decoded_feats_pca_norm[mask]
    print(f"\nVisualizing {len(viz_coords_np)} points for PCA (z < 2.5).")

    pcd_decoder = o3d.geometry.PointCloud()
    pcd_decoder.points = o3d.utility.Vector3dVector(viz_coords_np)
    pcd_decoder.colors = o3d.utility.Vector3dVector(viz_colors_np)
    _visualize_with_pose(pcd_decoder, "Decoder Features PCA Visualization", CAMERA_POSE, point_size)

def visualize_text_similarity(coords_np, decoded_feats_np, text_embedding, text_label="text", point_size: float = 2.0):
    if text_embedding is None:
        print("No text embedding provided, skipping text similarity visualization.")
        return

    print(f"\nVisualizing cosine similarity to '{text_label}'")
    
    text_vec = text_embedding / np.linalg.norm(text_embedding)
    feats_norm = decoded_feats_np / np.linalg.norm(decoded_feats_np, axis=1, keepdims=True)
    s_raw = feats_norm @ text_vec
    s_final = _minmax(s_raw)
    colors_sim = cm.get_cmap("plasma")(s_final)[:, :3]

    # Filter points for visualization
    mask = coords_np[:, 2] < 2.5
    viz_coords_np = coords_np[mask]
    viz_colors_np = colors_sim[mask]
    print(f"Visualizing {len(viz_coords_np)} points for Text Similarity (z < 2.5).")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(viz_coords_np)
    pcd.colors = o3d.utility.Vector3dVector(viz_colors_np)
    _visualize_with_pose(pcd, f"Cosine similarity to '{text_label}'", CAMERA_POSE, point_size)

def main():
    parser = argparse.ArgumentParser(description='Visualize voxel features using PCA and text similarity')
    parser.add_argument('--model_type', type=str, choices=['dense', 'sparse'], default='sparse',
                      help='Type of model to load (dense or sparse)')
    parser.add_argument('--points', type=str, default='playground/color_0.ply',
                      help='Path to points .ply file')
    parser.add_argument('--label', type=str, default="bowl",
                      help="Text label for text-similarity ('' to disable)")
    parser.add_argument('--point_size', type=float, default=6.0,
                        help='Initial point size (pixels). Adjust at runtime with [-]/[=] or [ ] keys.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from lang_mapping.mapper.mapper import VoxelHashTable
    static_map = VoxelHashTable(device=str(device), mode='train').to(device)
    voxel_checkpoint = torch.load('demo/20250805-161838/best/hash_voxel_dense.pt', map_location=device)
    static_map.load_state_dict(voxel_checkpoint['state_dict'], strict=True)
    
    from lang_mapping.module import ImplicitDecoder
    voxel_feature_dim = static_map.d * len(static_map.levels)
    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=voxel_feature_dim,
        hidden_dim=240,
        output_dim=768
    ).to(device)

    decoder_checkpoint = torch.load('demo/20250805-161838/best/implicit_decoder.pt', map_location=device)
    implicit_decoder.load_state_dict(decoder_checkpoint['model'])

    coords_np, decoded_feats_np, colors_np = calculate_features(static_map, implicit_decoder, points_path=args.points)

    visualize_raw_rgb(coords_np, colors_np, args.point_size)
    visualize_pca(coords_np, decoded_feats_np, args.point_size)

    text_emb = None
    if args.label:
        print(f"\nLoading CLIP model to get embedding for '{args.label}'...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14", pretrained="merged2b_s4b_b131k", device=device
        )
        tokenizer = open_clip.get_tokenizer("EVA02-L-14")

        with torch.no_grad():
            tok = tokenizer([args.label, ""]).to(device)
            all_emb = F.normalize(clip_model.encode_text(tok), dim=-1, p=2)
            text_emb = (all_emb[0] - all_emb[1])
        text_emb = F.normalize(text_emb, dim=-1, p=2).cpu().numpy()

    visualize_text_similarity(coords_np, decoded_feats_np, text_emb, args.label, args.point_size)

if __name__ == "__main__":
    main()
