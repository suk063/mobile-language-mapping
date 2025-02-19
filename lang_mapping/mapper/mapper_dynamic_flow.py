import torch
import torch.nn as nn
from ..module import LocalSelfAttentionFusion, ImplicitDecoder

class VoxelHashTableDynamicFlow(nn.Module):
    """
    A voxel hash table with separate static/dynamic embeddings (same dimension)
    and a time-embedding table for temporal conditioning.
    Extended to predict scene flow with a separate time embedding (time_embeddings_flow).
    """
    def __init__(
        self,
        resolution: float = 0.1,
        hash_table_size: int = 2**20,
        feature_dim: int = 120,
        scene_feature_dim: int =64,
        scene_bound_min: tuple = (-2.6, -8.1, 0),
        scene_bound_max: tuple = (4.6, 4.7, 3.1),
        mod_time: int = 201,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.scene_feature_dim = scene_feature_dim
        self.mod_time = mod_time
        self.device = device

        # Hash primes for x,y,z
        self.primes_xyz = torch.tensor([73856093, 19349669, 83492791],
                                       device=device, dtype=torch.long)

        # Create 3D voxel coordinates
        xs = torch.arange(scene_bound_min[0], scene_bound_max[0], resolution)
        ys = torch.arange(scene_bound_min[1], scene_bound_max[1], resolution)
        zs = torch.arange(scene_bound_min[2], scene_bound_max[2], resolution)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.voxel_coords = torch.stack([gx, gy, gz], dim=-1).view(-1, 3).to(device)
        self.total_voxels = self.voxel_coords.shape[0]

        # Static & dynamic embeddings (for reconstruction tasks, etc.)
        self.static_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )
        self.dynamic_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # Time embeddings (for reconstruction)
        self.time_embeddings = nn.Parameter(
            torch.randn(self.mod_time, feature_dim, device=device) * 0.01
        )

        self.static_flow_features = nn.Parameter(
            torch.randn(self.total_voxels, scene_feature_dim, device=device) * 0.01
        )
        self.dynamic_flow_features = nn.Parameter(
            torch.randn(self.total_voxels, scene_feature_dim, device=device) * 0.01
        )
        # Separate time embedding for scene flow
        self.time_embeddings_flow = nn.Parameter(
            torch.randn(self.mod_time, scene_feature_dim, device=device) * 0.01
        )
        
        # Scene flow MLP
        self.flow_mlp = nn.Sequential(
            nn.Linear(scene_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # outputs a 3D flow vector
        )

        # self.flow_mlp = ImplicitDecoder(voxel_feature_dim=scene_feature_dim, hidden_dim=256, output_dim=3, L=10)

        # Hash table index buffer
        self.buffer_voxel_index = torch.full((hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()

        # Two attention modules for main feats
        self.fusion_time_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)
        self.fusion_static_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)

        # Two attention modules for flow feats
        self.fusion_time_dynamic_flow = LocalSelfAttentionFusion(feat_dim=scene_feature_dim, num_heads=4)
        self.fusion_static_dynamic_flow = LocalSelfAttentionFusion(feat_dim=scene_feature_dim, num_heads=4)

        # Debug storage
        self.voxel_points = {}

    def build_hash_grid(self):
        """
        Build a hash grid of voxel indices. Collisions are skipped.
        """
        grid_coords = torch.floor(self.voxel_coords / self.resolution).to(torch.int64)
        hash_vals = torch.remainder((grid_coords * self.primes_xyz).sum(dim=-1),
                                    self.hash_table_size)

        collisions = 0
        for i in range(self.total_voxels):
            h = hash_vals[i].item()
            if self.buffer_voxel_index[h] == -1:
                self.buffer_voxel_index[h] = i
            else:
                collisions += 1

        if collisions > 0:
            print(f"[WARNING] {collisions} collisions among {self.total_voxels} voxels.")

    def query_voxel_feature(self, query_pts, query_times, return_indices=False):
        """
        Returns fused features for (x,y,z,t) from the static/dynamic embeddings
        with time embedding for reconstruction tasks.
        """
        device = query_pts.device
        M = query_pts.shape[0]

        # Hash lookup
        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_xyz = torch.remainder(
            (grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
        ).long()

        voxel_indices = self.buffer_voxel_index[hash_xyz]
        valid_mask = voxel_indices >= 0

        # Mod time
        t_mod = torch.remainder(query_times, self.mod_time).long().to(device)

        feats = torch.zeros(M, self.feature_dim, device=device)

        if valid_mask.any():
            v_idx = voxel_indices[valid_mask]
            t_idx = t_mod[valid_mask]

            static_feats = self.static_features[v_idx]
            dynamic_feats = self.dynamic_features[v_idx]
            time_emb = self.time_embeddings[t_idx]

            # Fuse dynamic + time
            cond_dynamic = self.fusion_time_dynamic(
                dynamic_feats.unsqueeze(1), time_emb.unsqueeze(1)
            ).squeeze(1)

            # Fuse static + cond_dynamic
            fused = self.fusion_static_dynamic(
                static_feats.unsqueeze(1), cond_dynamic.unsqueeze(1)
            ).squeeze(1)

            feats[valid_mask] = fused

        if return_indices:
            return feats, voxel_indices
        return feats, None

    def query_voxel_flow_feature(self, query_pts, query_times):
        """
        Returns fused flow features using static_flow_features, dynamic_flow_features,
        and time_embeddings_flow.
        """
        device = query_pts.device
        M = query_pts.shape[0]

        # Hash lookup
        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_xyz = torch.remainder(
            (grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
        ).long()

        voxel_indices = self.buffer_voxel_index[hash_xyz]
        valid_mask = voxel_indices >= 0

        # Mod time
        t_mod = torch.remainder(query_times, self.mod_time).long().to(device)

        flow_feats = torch.zeros(M, self.scene_feature_dim, device=device)

        if valid_mask.any():
            v_idx = voxel_indices[valid_mask]
            t_idx = t_mod[valid_mask]

            # flow embeddings
            static_flow = self.static_flow_features[v_idx]
            dynamic_flow = self.dynamic_flow_features[v_idx]
            time_flow = self.time_embeddings_flow[t_idx]

            # time + dynamic flow
            cond_flow_dyn = self.fusion_time_dynamic_flow(
                dynamic_flow.unsqueeze(1), time_flow.unsqueeze(1)
            ).squeeze(1)

            # fuse static_flow + cond_flow_dyn
            fused_flow = self.fusion_static_dynamic_flow(
                static_flow.unsqueeze(1), cond_flow_dyn.unsqueeze(1)
            ).squeeze(1)

            flow_feats[valid_mask] = fused_flow

        return flow_feats

    def query_scene_flow(self, query_pts, query_times):
        """
        Predict scene flow v in R^3 at each spatial-temporal point (x, t).
        """
        flow_feats = self.query_voxel_flow_feature(query_pts, query_times)
        v = self.flow_mlp(flow_feats)  # shape: [N, 3]
        return v

    def add_points(self, voxel_indices, points_3d, times):
        """
        Store sample points/times for debugging.
        """
        v_idx_cpu = voxel_indices.detach().cpu().numpy()
        points_cpu = points_3d.detach().cpu()
        times_cpu = times.detach().cpu().numpy()

        for i in range(len(v_idx_cpu)):
            vid = int(v_idx_cpu[i])
            if vid < 0:
                continue
            if vid not in self.voxel_points:
                self.voxel_points[vid] = []
            if len(self.voxel_points[vid]) < 10:
                self.voxel_points[vid].append((points_cpu[i], times_cpu[i]))