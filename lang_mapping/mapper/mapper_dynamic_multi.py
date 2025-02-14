import torch
import torch.nn as nn
from ..module import LocalSelfAttentionFusion, LocalSelfAttentionFusionMulti

class VoxelHashTableDynamicMulti(nn.Module):
    """
    Single dynamic_features_pose + single pose_mlp + single fusion_pose.
    """
    def __init__(
        self,
        resolution=0.1,
        hash_table_size=2**20,
        feature_dim=120,
        scene_bound_min=(-2.6, -8.1, 0),
        scene_bound_max=(4.6, 4.7, 3.1),
        mod_time=201,
        device="cuda:0",
        pose_dim=12,   # example
        state_dim=42  # example
    ):
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.mod_time = mod_time
        self.device = device

        # Primes for hashing
        self.primes_xyz = torch.tensor([73856093, 19349669, 83492791], device=device, dtype=torch.long)

        # Build voxel coords
        xs = torch.arange(scene_bound_min[0], scene_bound_max[0], resolution)
        ys = torch.arange(scene_bound_min[1], scene_bound_max[1], resolution)
        zs = torch.arange(scene_bound_min[2], scene_bound_max[2], resolution)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.voxel_coords = torch.stack([gx, gy, gz], dim=-1).view(-1, 3).to(device)
        self.total_voxels = self.voxel_coords.shape[0]

        # Static features
        self.static_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # Dynamic features (time)
        self.dynamic_features_time = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(self.mod_time, feature_dim, device=device) * 0.01
        )

        # Single dynamic features for pose (shared by head/hand)
        self.dynamic_features_pose = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )
        # Single MLP for pose (head or hand)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        ).to(device)

        # Dynamic features for state
        self.dynamic_features_state = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        ).to(device)

        # Hash index buffer
        self.buffer_voxel_index = torch.full((hash_table_size,), -1, dtype=torch.long, device=device)
        self.build_hash_grid()

        # Pairwise fusions
        self.fusion_time_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)
        self.fusion_pose = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)
        self.fusion_state = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)

        # Final multi-fusion
        self.fusion_all_multi = LocalSelfAttentionFusionMulti(feat_dim=feature_dim, num_heads=8)

        # Debug storage
        self.voxel_points = {}

    def build_hash_grid(self):
        grid_coords = torch.floor(self.voxel_coords / self.resolution).to(torch.int64)
        hash_vals = torch.remainder((grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size)

        collisions = 0
        for i in range(self.total_voxels):
            h = hash_vals[i].item()
            if self.buffer_voxel_index[h] == -1:
                self.buffer_voxel_index[h] = i
            else:
                collisions += 1

        if collisions > 0:
            print(f"[WARNING] {collisions} collisions among {self.total_voxels} voxels.")

    def query_voxel_feature(
        self,
        query_pts,
        query_times,
        query_pose=None,
        query_state=None,
        return_indices=False
    ):
        device = query_pts.device
        M = query_pts.shape[0]

        # Hash lookup
        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_xyz = torch.remainder((grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size)
        voxel_indices = self.buffer_voxel_index[hash_xyz]
        valid_mask = voxel_indices >= 0

        # Time mod
        t_mod = torch.remainder(query_times, self.mod_time).long().to(device)

        feats = torch.zeros(M, self.feature_dim, device=device)

        if valid_mask.any():
            v_idx = voxel_indices[valid_mask]
            t_idx = t_mod[valid_mask]

            # Static
            static_feats = self.static_features[v_idx]

            # Time
            dynamic_feats_time = self.dynamic_features_time[v_idx]
            time_emb = self.time_embeddings[t_idx]
            cond_time = self.fusion_time_dynamic(
                dynamic_feats_time.unsqueeze(1),
                time_emb.unsqueeze(1)
            ).squeeze(1)

            # Pose
            dynamic_feats_pose = self.dynamic_features_pose[v_idx]
            pose_emb = self.pose_mlp(query_pose[valid_mask])
            cond_pose = self.fusion_pose(
                dynamic_feats_pose.unsqueeze(1),
                pose_emb.unsqueeze(1)
            ).squeeze(1)

            # State
            dynamic_feats_state = self.dynamic_features_state[v_idx]
            cond_state = torch.zeros_like(cond_time)
            state_emb = self.state_mlp(query_state[valid_mask])
            cond_state = self.fusion_state(
                dynamic_feats_state.unsqueeze(1),
                state_emb.unsqueeze(1)
            ).squeeze(1)

            # Fuse 4 tokens
            # (B,N,D) -> here B=valid_count, N=1
            static_1 = static_feats.unsqueeze(1)
            time_1 = cond_time.unsqueeze(1)
            pose_1 = cond_pose.unsqueeze(1)
            state_1 = cond_state.unsqueeze(1)
            fused_4 = self.fusion_all_multi([static_1, time_1, pose_1, state_1]).squeeze(1)

            feats[valid_mask] = fused_4

        if return_indices:
            return feats, voxel_indices
        return feats, None

    def add_points(self, voxel_indices, points_3d, times):
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