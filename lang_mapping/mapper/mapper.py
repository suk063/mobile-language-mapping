import torch
import torch.nn as nn

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
            torch.zeros(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # 3) Hash table (stores indices of voxel_features)
        self.buffer_voxel_index = torch.full((self.hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()
        self.voxel_points = {}
        
        self.register_buffer(
            "used_mask",
            torch.zeros(self.total_voxels, dtype=torch.bool, device=device)
        )
        # self.register_buffer(
        #     "valid_grid_coords",
        #     torch.empty((0, 3), device=device)
        # )

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

        # mark valid voxel features
        # valid_voxel_indices = voxel_indices[valid_mask]
        # self.used_mask[valid_voxel_indices] = True

        # self.valid_grid_coords = self.voxel_coords[self.used_mask]

        if return_indices:
            return feats, voxel_indices
        else:
            return feats, None

    def query_voxel_feature_from_subset(self, subset_coords, return_indices=False):
        return self.query_voxel_feature(subset_coords, return_indices=return_indices)

    def get_all_valid_voxel_data(self):
        """
        Returns:
            valid_coords (Tensor): [N, 3] 
            valid_feats (Tensor):  [N, feature_dim]
        """
        valid_indices = torch.where(self.used_mask)[0]
        valid_coords = self.voxel_coords[valid_indices]
        valid_feats = self.voxel_features[valid_indices]
        return valid_coords, valid_feats

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
