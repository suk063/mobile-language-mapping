import torch
import torch.nn as nn
from ..module import LocalSelfAttentionFusion, ImplicitDecoder

class VoxelHashTableFlowTraverse(nn.Module):
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
        trilinear_feat: bool = False,
        trilinear_flow: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.resolution = resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.scene_feature_dim = scene_feature_dim
        self.mod_time = mod_time
        self.trilinear_feat = trilinear_feat
        self.trilinear_flow = trilinear_flow
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

        # dynamic embeddings (for reconstruction tasks, etc.)
        self.dynamic_features = nn.Parameter(
            torch.randn(self.total_voxels, feature_dim, device=device) * 0.01
        )

        # Time embeddings (for reconstruction)
        self.time_embeddings = nn.Parameter(
            torch.randn(self.mod_time, feature_dim, device=device) * 0.01
        )

        self.dynamic_flow_features = nn.Parameter(
            torch.randn(self.total_voxels, scene_feature_dim, device=device) * 0.01
        )
        # Separate time embedding for scene flow
        self.time_embeddings_flow = nn.Parameter(
            torch.randn(self.mod_time, scene_feature_dim, device=device) * 0.01
        )
        
        # Scene flow MLP
        self.flow_mlp = ImplicitDecoder(voxel_feature_dim=scene_feature_dim, hidden_dim=256, output_dim=3, L=10)

        # Hash table index buffer
        self.buffer_voxel_index = torch.full((hash_table_size,), -1,
                                             dtype=torch.long, device=device)
        self.build_hash_grid()

        # Two attention modules for main feats
        self.fusion_time_dynamic = LocalSelfAttentionFusion(feat_dim=feature_dim, num_heads=8)

        # Two attention modules for flow feats
        self.fusion_time_dynamic_flow = LocalSelfAttentionFusion(feat_dim=scene_feature_dim, num_heads=8)

        # Debug storage
        self.voxel_points = {}

        self.var_norm = None

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
        Returns fused features for (x,y,z,t) from the dynamic embeddings
        plus a time embedding for reconstruction tasks.
        
        If self.trilinear_feat is True, perform trilinear interpolation
        on the 8 corner voxels surrounding each query point.
        Otherwise, return a direct lookup using floor() + hashing.
        """
        device = query_pts.device
        M = query_pts.shape[0]

        # Time index (modulus)
        t_mod = torch.remainder(query_times, self.mod_time).long().to(device)

        # Initialize output
        feats = torch.zeros(M, self.feature_dim, device=device)

        if not self.trilinear_feat:
            # ============ Direct voxel lookup (existing behavior) ============
            grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
            hash_xyz = torch.remainder(
                (grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
            ).long()

            voxel_indices = self.buffer_voxel_index[hash_xyz]
            valid_mask = voxel_indices >= 0

            if valid_mask.any():
                v_idx = voxel_indices[valid_mask]
                t_idx = t_mod[valid_mask]

                dyn_feats = self.dynamic_features[v_idx]
                time_emb = self.time_embeddings[t_idx]

                # Fuse dynamic + time
                cond_dynamic = self.fusion_time_dynamic(
                    dyn_feats.unsqueeze(1), time_emb.unsqueeze(1)
                ).squeeze(1)

                feats[valid_mask] = cond_dynamic

            if return_indices:
                return feats, voxel_indices
            return feats, None

        else:
            # ============ Trilinear interpolation ============
            # Convert query_pts to "grid space" = (x / resolution)
            grid_coords_float = query_pts / self.resolution
            grid_coords_floor = torch.floor(grid_coords_float).to(torch.int64)
            frac = grid_coords_float - grid_coords_floor.float()  # fractional part in [0,1)

            # For time embedding, we still do discrete lookup
            # We'll accumulate the dynamic feats first, then fuse with time.
            dyn_feats_accum = torch.zeros_like(feats)

            # 8 corners in 3D: offsets in {0,1}^3
            corner_offsets = torch.tensor([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ], device=device, dtype=torch.int64)

            for corner in corner_offsets:
                # corner_coords = floor_coords + corner_offset
                corner_coords = grid_coords_floor + corner

                # Compute the interpolation weight for this corner
                # corner is e.g. (0,0,1) => weight_x = (1-frac_x) if corner.x==0, else frac_x
                weight_x = torch.where(corner[0] == 0, 1.0 - frac[:, 0], frac[:, 0])
                weight_y = torch.where(corner[1] == 0, 1.0 - frac[:, 1], frac[:, 1])
                weight_z = torch.where(corner[2] == 0, 1.0 - frac[:, 2], frac[:, 2])
                corner_weight = weight_x * weight_y * weight_z  # shape (M,)

                # Hash lookup for corner
                hash_xyz_corner = torch.remainder(
                    (corner_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
                ).long()

                corner_voxel_idx = self.buffer_voxel_index[hash_xyz_corner]
                valid_mask_c = corner_voxel_idx >= 0

                # Add weighted feats to accum
                if valid_mask_c.any():
                    v_idx = corner_voxel_idx[valid_mask_c]
                    dyn_feats_c = self.dynamic_features[v_idx]

                    # Weighted contribution
                    w_c = corner_weight[valid_mask_c].unsqueeze(-1)  # shape (valid_count,1)
                    dyn_feats_accum[valid_mask_c] += dyn_feats_c * w_c

            # Now fuse with time embedding
            time_emb = self.time_embeddings[t_mod]
            # Note: we fuse each point's dynamic embedding with time embedding
            # using the local self-attention module.
            cond_dynamic = self.fusion_time_dynamic(
                dyn_feats_accum.unsqueeze(1), time_emb.unsqueeze(1)
            ).squeeze(1)

            feats = cond_dynamic

            if return_indices:
                # There's no single voxel index for trilinear, so we can return None or the floor index
                # Returning the floor index or a special value
                voxel_indices = torch.full((M,), -1, device=device, dtype=torch.long)
                return feats, voxel_indices

            return feats, None

    def query_voxel_flow_feature(self, query_pts, query_times):
        """
        Returns fused flow features using dynamic_flow_features
        and time_embeddings_flow. Optionally uses trilinear interpolation
        if self.trilinear_flow is True.
        """
        device = query_pts.device
        M = query_pts.shape[0]

        # Time index (modulus)
        t_mod = torch.remainder(query_times, self.mod_time).long().to(device)

        flow_feats = torch.zeros(M, self.scene_feature_dim, device=device)

        if not self.trilinear_flow:
            # ============ Direct voxel lookup (existing behavior) ============
            grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
            hash_xyz = torch.remainder(
                (grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
            ).long()

            voxel_indices = self.buffer_voxel_index[hash_xyz]
            valid_mask = voxel_indices >= 0

            if valid_mask.any():
                v_idx = voxel_indices[valid_mask]
                t_idx = t_mod[valid_mask]

                dynamic_flow = self.dynamic_flow_features[v_idx]
                time_flow = self.time_embeddings_flow[t_idx]

                # Fuse dynamic flow + time flow
                cond_flow_dyn = self.fusion_time_dynamic_flow(
                    dynamic_flow.unsqueeze(1), time_flow.unsqueeze(1)
                ).squeeze(1)

                flow_feats[valid_mask] = cond_flow_dyn

            return flow_feats

        else:
            # ============ Trilinear interpolation for flow features ============
            grid_coords_float = query_pts / self.resolution
            grid_coords_floor = torch.floor(grid_coords_float).to(torch.int64)
            frac = grid_coords_float - grid_coords_floor.float()

            # Accumulator
            flow_accum = torch.zeros_like(flow_feats)

            corner_offsets = torch.tensor([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ], device=device, dtype=torch.int64)

            for corner in corner_offsets:
                corner_coords = grid_coords_floor + corner

                weight_x = torch.where(corner[0] == 0, 1.0 - frac[:, 0], frac[:, 0])
                weight_y = torch.where(corner[1] == 0, 1.0 - frac[:, 1], frac[:, 1])
                weight_z = torch.where(corner[2] == 0, 1.0 - frac[:, 2], frac[:, 2])
                corner_weight = weight_x * weight_y * weight_z

                # Hash
                hash_xyz_corner = torch.remainder(
                    (corner_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size
                ).long()

                corner_voxel_idx = self.buffer_voxel_index[hash_xyz_corner]
                valid_mask_c = corner_voxel_idx >= 0

                if valid_mask_c.any():
                    v_idx = corner_voxel_idx[valid_mask_c]
                    dyn_flow_feats_c = self.dynamic_flow_features[v_idx]

                    w_c = corner_weight[valid_mask_c].unsqueeze(-1)
                    flow_accum[valid_mask_c] += dyn_flow_feats_c * w_c

            # Fuse with time flow embedding
            time_flow = self.time_embeddings_flow[t_mod]
            cond_flow_dyn = self.fusion_time_dynamic_flow(
                flow_accum.unsqueeze(1), time_flow.unsqueeze(1)
            ).squeeze(1)

            return cond_flow_dyn

    def query_scene_flow(self, query_pts, query_times):
        """
        Predict scene flow v in R^3 at each spatial-temporal point (x, t).
        """
        flow_feats = self.query_voxel_flow_feature(query_pts, query_times)
        v = self.flow_mlp(flow_feats, query_pts)  # shape: [N, 3]
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
    
    @torch.no_grad()
    def compute_time_variance(self, chunk_size: int = 50000):
        """
        Compute time variance (L2 norm of variance vector) over the range of times [0..(self.mod_time-1)].
        We use query_voxel_feature() to observe how each voxel's fused feature changes over time.
        
        Args:
            chunk_size (int): number of voxels to process in a single iteration to reduce memory usage.
        """
        device = self.device
        coords_all = self.voxel_coords  # shape = (N,3)
        N = coords_all.shape[0]
        times_ = torch.arange(0, self.mod_time, device=device, dtype=torch.long)
        T = times_.numel()

        var_norm_list = []

        start_idx = 0
        while start_idx < N:
            end_idx = min(start_idx + chunk_size, N)
            coords_chunk = coords_all[start_idx:end_idx]  # shape (M,3)
            M = coords_chunk.shape[0]

            # Expand coordinates for times
            coords_expanded = coords_chunk.unsqueeze(1).expand(M, T, 3).reshape(-1, 3)  # (M*T,3)
            times_expanded = times_.unsqueeze(0).expand(M, T).reshape(-1)               # (M*T,)

            # Query voxel features (which includes static+dynamic+time attention)
            feats_flat, _ = self.query_voxel_feature(coords_expanded, times_expanded, return_indices=False)
            # feats_flat shape = (M*T, feature_dim)

            feats_3d = feats_flat.view(M, T, self.feature_dim)  # (M, T, feature_dim)
            
            # Compute variance across time dimension
            # var_3d shape = (M, feature_dim)
            var_3d = feats_3d.var(dim=1, unbiased=True)
            
            # L2 norm of the variance vector
            # var_norm shape = (M,)
            var_norm = var_3d.norm(dim=1)

            var_norm_list.append(var_norm)
            start_idx = end_idx

        # Concatenate results for all voxels
        # shape = (N,)
        self.var_norm = torch.cat(var_norm_list, dim=0)
        print(f"[INFO] compute_time_variance: computed time variance for {N} voxels. var_norm shape={self.var_norm.shape}")

    def get_variance_for_points(self, query_pts: torch.Tensor) -> torch.Tensor:
        """
        Given an array of 3D query points, returns the time variance (self.var_norm)
        for the corresponding voxel. If the point is invalid (hash collision or out of range),
        the variance is 0.
        
        Args:
            query_pts (torch.Tensor): shape (N,3), each row is a 3D coordinate in world space.
        
        Returns:
            torch.Tensor: shape (N,), the time variance scalar per point.
        """
        assert self.var_norm is not None, "Must call compute_time_variance() first to populate self.var_norm."

        device = query_pts.device
        N = query_pts.shape[0]

        # Hash lookup
        grid_coords = torch.floor(query_pts / self.resolution).to(torch.int64)
        hash_xyz = torch.remainder((grid_coords * self.primes_xyz).sum(dim=-1), self.hash_table_size).long()

        voxel_indices = self.buffer_voxel_index[hash_xyz]
        valid_mask = (voxel_indices >= 0)

        # Initialize output with zeros
        out_var = torch.zeros(N, device=device, dtype=self.var_norm.dtype)
        # For valid ones, fill in
        out_var[valid_mask] = self.var_norm[voxel_indices[valid_mask]]
        return out_var