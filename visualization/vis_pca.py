import torch
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import argparse

def visualize_valid_coords_pca_open3d(static_map, implicit_decoder=None):

    # Load vertices and colors
    vertices_np = np.load("vertices.npy").astype(np.float32)   # (N,3)
    colors_np = np.load("colors.npy").astype(np.float32)       # (N,3)

    # Downsample points
    ds_size = 0.05  # m
    voxel_idx = np.floor(vertices_np / ds_size).astype(np.int32)
    _, uniq = np.unique(voxel_idx, axis=0, return_index=True)

    vertices_ds = vertices_np[uniq]
    colors_ds = colors_np[uniq] / 255.0  # → [0,1]
    print(f"[DownSample] {len(vertices_np):,} → {len(vertices_ds):,} points")

    # Convert to torch tensor
    coords_torch = torch.from_numpy(vertices_ds).to('cuda')

    # Query features
    feats = static_map.query_voxel_feature(coords_torch)  # [N, feature_dim]
    
    # Convert to numpy
    coords_np = coords_torch.detach().cpu().numpy()  
    feats_np = feats.detach().cpu().numpy()          

    # PCA (3D)
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(feats_np)  # [N, 3]
    print("Voxel Features PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Sum of explained variance ratio:", pca.explained_variance_ratio_.sum())

    # Normalize for color
    scaler = MinMaxScaler()
    feats_pca_norm = scaler.fit_transform(feats_pca)  # range [0,1]

    # Open3D visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_np)
    pcd.colors = o3d.utility.Vector3dVector(feats_pca_norm)
    o3d.visualization.draw_geometries([pcd], window_name="Voxel Features PCA Visualization")

    # If decoder is provided, also visualize decoder features
    if implicit_decoder is not None:
        with torch.no_grad():
            decoded_feats = implicit_decoder(feats, coords_torch)  # [N, output_dim]
            decoded_feats_np = decoded_feats.detach().cpu().numpy()

            # PCA on decoder features
            pca_decoder = PCA(n_components=3)
            decoded_feats_pca = pca_decoder.fit_transform(decoded_feats_np)
            print("\nDecoder Features PCA explained variance ratio:", pca_decoder.explained_variance_ratio_)
            print("Sum of explained variance ratio:", pca_decoder.explained_variance_ratio_.sum())

            # Normalize for color
            decoded_feats_pca_norm = scaler.fit_transform(decoded_feats_pca)

            # Visualize decoder features
            pcd_decoder = o3d.geometry.PointCloud()
            pcd_decoder.points = o3d.utility.Vector3dVector(coords_np)
            pcd_decoder.colors = o3d.utility.Vector3dVector(decoded_feats_pca_norm)
            o3d.visualization.draw_geometries([pcd_decoder], window_name="Decoder Features PCA Visualization")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize voxel features using PCA')
    parser.add_argument('--model_type', type=str, choices=['dense', 'sparse'], default='sparse',
                      help='Type of model to load (dense or sparse)')
    args = parser.parse_args()

    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoints based on model type
    if args.model_type == 'dense':
        # Initialize VoxelHashTable for dense model
        from lang_mapping.mapper.mapper import VoxelHashTable
        static_map = VoxelHashTable(
            resolution=0.12,
            hash_table_size=2097152,
            feature_dim=64,
            scene_bound_min=(-2.6, -8.1, 0.0),
            scene_bound_max=(4.6, 4.7, 3.1),
            device=device
        ).to(device)
        
        voxel_checkpoint = torch.load('hash_voxel_dense.pt', map_location=device)
        static_map.load_state_dict(voxel_checkpoint['state_dict'], strict=True)
    else:  # sparse
        # Load sparse model directly
        from lang_mapping.mapper.mapper import VoxelHashTable
        static_map = VoxelHashTable.load_sparse('hash_voxel_sparse.pt', device=device)

    # Load decoder
    from lang_mapping.module import ImplicitDecoder
    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=128,  # feature_dim from VoxelHashTable
        hidden_dim=768,        # default value
        output_dim=768         # CLIP feature dimension
    ).to(device)
    
    decoder_checkpoint = torch.load('implicit_decoder.pt', map_location=device)
    implicit_decoder.load_state_dict(decoder_checkpoint['model'])

    # Visualize with Open3D
    visualize_valid_coords_pca_open3d(static_map, implicit_decoder)

if __name__ == "__main__":
    main()