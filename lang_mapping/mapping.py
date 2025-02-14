import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import positional_encoding
from .policy import LocalSelfAttentionFusion, LocalSelfAttentionFusionMulti

class VoxelHashTable(nn.Module):
    """
    Builds a 3D voxel grid for a specified scene bound and stores per-voxel features.
    Features are accessed via a hash table for efficient lookup.
    """
    def __init__(
        self,
        resolution: float = 0.1,
        hash_table_size: int = 2**20,
        feature_dim: int = 768,
        scene_bound_min: tuple = (-2.6, -8.1, 0),
        scene_bound_max: tuple = (4.6, 4.7, 3.1),
        device: str = "cuda:0",
    ):
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.device = device

        # Large prime numbers for hashing
        self.primes = torch.tensor([73856093, 19349669, 83492791],
                                   device=device, dtype=torch.long)

        # 1) Create voxel coordinate grid
        xs = torch.arange(scene_bound_min[0], scene_bound_max[0], resolution)
        ys = torch.arange(scene_bound_min[1], scene_bound_max[1], resolution)
        zs = torch.arange(scene_bound_min[2], scene_bound_max[2], resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)
        self.total_voxels = self.voxel_coords.shape[0]

        # 2) Voxel feature parameters
        self.voxel_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # 3) Hash table (stores indices of voxel_features)
        self.buffer_voxel_index = torch.full((self.hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()
        self.voxel_points = {}

    def build_hash_grid(self):
        """
        Maps each voxel in the grid to an entry in the hash table using a simple
        modulo-based hash function. Collisions are possible, and those voxels
        remain unstored in the hash table (index remains -1).
        """
        grid_coords = torch.floor(self.voxel_coords / self.resolution).to(torch.int64)
        hash_vals = torch.fmod((grid_coords * self.primes).sum(dim=-1), self.hash_table_size)

        collisions = 0
        for i in range(self.total_voxels):
            h = hash_vals[i].item()
            if self.buffer_voxel_index[h] == -1:
                self.buffer_voxel_index[h] = i
            else:
                collisions += 1

        if collisions > 0:
            print(f"[WARNING] {collisions} collisions out of {self.total_voxels} voxels. "
                  "Some voxels are not stored in the hash table.")

    def query_voxel_feature(self, query_pts, return_indices=False):
        """
        Returns voxel features for the given 3D query points by looking them up in the hash table.
        
        Args:
            query_pts (Tensor): [M, 3] - 3D coordinates to query.
            return_indices (bool): If True, also returns the voxel indices.

        Returns:
            feats (Tensor): [M, feature_dim] - Retrieved voxel features (zeros if not found).
            voxel_indices (Tensor or None): [M] - Indices of the voxels in the feature buffer (if return_indices=True).
        """
        device = query_pts.device
        M = query_pts.shape[0]

        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_vals = torch.fmod((grid_coords * self.primes).sum(dim=-1), self.hash_table_size).long()

        voxel_indices = self.buffer_voxel_index[hash_vals]  # [M]
        valid_mask = (voxel_indices >= 0)

        feats = torch.zeros(M, self.feature_dim, device=device)
        feats[valid_mask] = self.voxel_features[voxel_indices[valid_mask]]

        if return_indices:
            return feats, voxel_indices
        else:
            return feats, None

    def add_points(self, voxel_indices: torch.Tensor, points_3d: torch.Tensor):
        """
        Stores up to 10 points (3D coordinates) per voxel index for potential debug or analysis.

        Args:
            voxel_indices (Tensor): [M] - Valid voxel indices from the hash table.
            points_3d (Tensor): [M, 3] - 3D coordinates to store.
        """
        voxel_indices_cpu = voxel_indices.detach().cpu().numpy()
        points_cpu = points_3d.detach().cpu()

        for i in range(len(voxel_indices_cpu)):
            v_idx = int(voxel_indices_cpu[i])
            if v_idx < 0:
                continue
            if v_idx not in self.voxel_points:
                self.voxel_points[v_idx] = []
            if len(self.voxel_points[v_idx]) < 10:
                self.voxel_points[v_idx].append(points_cpu[i])

class ImplicitDecoder(nn.Module):
    """
    A simple MLP to decode a 3D coordinate (with positional encoding) and its corresponding
    voxel feature into another feature vector (default 768-D).
    """
    def __init__(self, voxel_feature_dim=120, hidden_dim=256, output_dim=768, L=10):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.pe_dim = 2 * self.L * 3  # 2*L for sine/cosine, times 3 for x, y, z

        # First linear layer input dimension: voxel_feature_dim + pe_dim
        self.input_dim = self.voxel_feature_dim + self.pe_dim
        self.output_dim = output_dim

        # Layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # Third layer also takes the positional encoding as an additional input
        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)

        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, voxel_features, coords_3d):
        """
        Args:
            voxel_features (Tensor): [N, voxel_feature_dim] - Features retrieved from the VoxelHashTable.
            coords_3d (Tensor): [N, 3] - 3D coordinates for positional encoding.

        Returns:
            Tensor: [N, 768] - Decoded feature representation.
        """
        pe = positional_encoding(coords_3d, L=self.L)  # [N, pe_dim]

        # 1) First linear
        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)

        # 2) Second linear
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)

        # 3) Concat positional encoding again
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)

        # 4) Fourth linear
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)

        # 5) Fifth linear
        x = self.fc5(x)
        return x

class VoxelHashTableDynamic(nn.Module):
    """
    A voxel hash table with separate static/dynamic embeddings (same dimension)
    and a time-embedding table for temporal conditioning.
    """
    def __init__(
        self,
        resolution: float = 0.1,
        hash_table_size: int = 2**20,
        feature_dim: int = 120,
        scene_bound_min: tuple = (-2.6, -8.1, 0),
        scene_bound_max: tuple = (4.6, 4.7, 3.1),
        mod_time: int = 201,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
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

        # Static and dynamic embeddings
        self.static_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )
        self.dynamic_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # Time embeddings for conditioning
        self.time_embeddings = nn.Parameter(
            torch.randn(self.mod_time, feature_dim, device=device) * 0.01
        )

        # Hash table index buffer
        self.buffer_voxel_index = torch.full((hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()

        # Two attention modules for fusion
        self.fusion_time_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)
        self.fusion_static_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)

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
        Returns fused features for (x,y,z,t).
        1) Hash to find voxel index
        2) Gather static and dynamic feats
        3) Add time embedding to dynamic via self-attention
        4) Fuse static and conditioned dynamic
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

        # Prepare output
        feats = torch.zeros(M, self.feature_dim, device=device)

        if valid_mask.any():
            v_idx = voxel_indices[valid_mask]
            t_idx = t_mod[valid_mask]

            static_feats = self.static_features[v_idx]
            dynamic_feats = self.dynamic_features[v_idx]
            time_emb = self.time_embeddings[t_idx]

            # Fuse dynamic + time
            cond_dynamic = self.fusion_time_dynamic(
                dynamic_feats.unsqueeze(1),
                time_emb.unsqueeze(1)
            ).squeeze(1)

            # Fuse static + cond_dynamic
            fused = self.fusion_static_dynamic(
                static_feats.unsqueeze(1),
                cond_dynamic.unsqueeze(1)
            ).squeeze(1)

            feats[valid_mask] = fused

        if return_indices:
            return feats, voxel_indices
        return feats, None

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