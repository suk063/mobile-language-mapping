import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import positional_encoding

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
    Dynamic voxel hash table with separate embeddings for static and dynamic components.
    Also stores (or predicts) a flow vector per voxel-time pair.
    """

    def __init__(
        self,
        resolution: float = 0.1,
        hash_table_size: int = 2**20,
        feature_dim_static: int = 384,
        feature_dim_dynamic: int = 384,
        scene_bound_min: tuple = (-2.6, -8.1, 0),
        scene_bound_max: tuple = (4.6, 4.7, 3.1),
        mod_time: int = 201,
        device: str = "cuda:0",
    ):
        """
        Args:
            resolution: Size of each voxel in meters.
            hash_table_size: Size of the underlying hash buffer.
            feature_dim_static: Feature dim for static portion of each voxel.
            feature_dim_dynamic: Feature dim for dynamic portion of each voxel.
            scene_bound_min: Minimum (x, y, z) for the entire scene.
            scene_bound_max: Maximum (x, y, z) for the entire scene.
            mod_time: Time modulus.
            device: Torch device.
        """
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim_static = feature_dim_static
        self.feature_dim_dynamic = feature_dim_dynamic
        self.total_feature_dim = feature_dim_static + feature_dim_dynamic
        self.device = device
        self.mod_time = mod_time

        # Large primes for hashing
        self.primes_xyz = torch.tensor([73856093, 19349669, 83492791],
                                       device=device, dtype=torch.long)
        # Another prime for time
        self.prime_t = torch.tensor(1645333507, device=device, dtype=torch.long)

        # 1) Build the base 3D voxel coordinates
        xs = torch.arange(scene_bound_min[0], scene_bound_max[0], resolution)
        ys = torch.arange(scene_bound_min[1], scene_bound_max[1], resolution)
        zs = torch.arange(scene_bound_min[2], scene_bound_max[2], resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)
        self.total_voxels = self.voxel_coords.shape[0]

        # 2) Define learnable static features [total_voxels, feature_dim_static]
        self.static_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim_static, device=device) * 0.01
        )

        # 3) Define learnable dynamic features [total_voxels, mod_time, feature_dim_dynamic]
        #    We store a separate dynamic embedding for each time index
        self.dynamic_features = nn.Parameter(
            torch.randn(self.total_voxels, self.mod_time, feature_dim_dynamic, device=device) * 0.01
        )

        # 4) Define learnable scene flow [total_voxels, mod_time, 3]
        #    Each voxel-time pair has a flow vector to warp points from t -> t+1
        self.scene_flow = nn.Parameter(
            torch.zeros(self.total_voxels, self.mod_time, 3, device=device)
        )

        # 5) Build the hash table (stores indices of each voxel)
        self.buffer_voxel_index = torch.full((self.hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()
        
        # For debugging / point storage
        self.voxel_points = {}

    def build_hash_grid(self):
        """
        Build the static hash grid for the 3D portion only.
        Time dimension is handled separately in the query.
        Collisions: if a hash bucket is already occupied, we do not store that voxel.
        """
        grid_coords = torch.floor(self.voxel_coords / self.resolution).to(torch.int64)
        hash_vals = torch.fmod((grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size)

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

    def query_voxel_feature(self, query_pts, query_times, return_indices=False):
        """
        Return static + dynamic features for (x, y, z, t).
        
        Args:
            query_pts: [M, 3] float - 3D coordinates to query
            query_times: [M] long - time steps to query
            return_indices: bool - whether to return the raw voxel indices (-1 if not found)

        Returns:
            feats: [M, feature_dim_static + feature_dim_dynamic]
            voxel_indices: [M] if return_indices=True, else None
            flow: [M, 3] scene flow for each voxel-time
        """
        device = query_pts.device
        M = query_pts.shape[0]

        # 1) Compute the 3D voxel index via hash
        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_xyz = torch.fmod((grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size).long()

        voxel_indices = self.buffer_voxel_index[hash_xyz]  # [M]
        valid_mask = (voxel_indices >= 0)

        # 2) Map time to mod_time
        modded_t = torch.fmod(query_times, self.mod_time).long().cuda()  # [M]

        # 3) Gather features
        feats = torch.zeros(M, self.total_feature_dim, device=device)

        if valid_mask.any():
            v_idx = voxel_indices[valid_mask]                # [M_valid]
            t_idx = modded_t[valid_mask]                     # [M_valid]
            
            # Gather static + dynamic
            static_feats = self.static_features[v_idx]                               # [M_valid, feature_dim_static]
            dynamic_feats = self.dynamic_features[v_idx, t_idx]                      # [M_valid, feature_dim_dynamic]
            combined = torch.cat([static_feats, dynamic_feats], dim=-1)             # [M_valid, total_feature_dim]
            
            feats[valid_mask] = combined

        if return_indices:
            return feats, voxel_indices
        else:
            return feats, None

    def add_points(self, voxel_indices: torch.Tensor, points_3d: torch.Tensor, times: torch.Tensor):
        """
        Store some 3D points (and times) for debugging/analysis.
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