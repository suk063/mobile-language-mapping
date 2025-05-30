import torch
import torch.nn as nn


class VoxelHashTable(nn.Module):
    """
    Multi-resolution hashed voxel pyramid.  Each level is an independent
    hashed grid that supports trilinear lookup.  The forward pass returns
    the per-level features concatenated along the last dimension.
    """
    def __init__(
        self,
        resolution: float = 0.12,        # cell size of the finest level
        num_levels: int = 2,             # number of pyramid levels
        level_scale: float = 2.0,        # spacing ratio between levels
        feature_dim: int = 32,           # feature width per level
        hash_table_size: int = 2 ** 20,  # buckets per level
        scene_bound_min: tuple = (-2.6, -8.1, 0.0),
        scene_bound_max: tuple = (4.6, 4.7, 3.1),
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_levels   = num_levels
        self.feature_dim  = feature_dim
        self.device       = device

        # Shared large primes for the spatial hash function
        primes = torch.tensor([73856093, 19349669, 83492791],
                              device=device, dtype=torch.long)

        # Build the pyramid
        self.levels = nn.ModuleList()
        for lv in range(num_levels):
            res = resolution * (level_scale ** lv)
            self.levels.append(
                _SingleResVoxelHashTable(
                    resolution      = res,
                    feature_dim     = feature_dim,
                    hash_table_size = hash_table_size,
                    scene_bound_min = scene_bound_min,
                    scene_bound_max = scene_bound_max,
                    primes          = primes,
                    device          = device,
                )
            )

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collision_stats(self) -> dict:
        """
        Returns a summary of hash-table collisions for every level.
        """
        out = {}
        for i, lv in enumerate(self.levels):
            out[f"level_{i}"] = lv.collision_stats()
        return out

    @torch.no_grad()
    def get_accessed_indices(self) -> list:
        """
        Returns a list of LongTensor(s); one tensor per level containing the
        unique voxel indices that have been accessed since the last reset.
        """
        return [lv.get_accessed_indices() for lv in self.levels]

    @torch.no_grad()
    def reset_access_log(self) -> None:
        """
        Zeros the per-level access masks.
        """
        for lv in self.levels:
            lv.reset_access_log()

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def query_voxel_feature(self, query_pts: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        query_pts : (M, 3) float32 world-space coordinates.

        Returns
        -------
        feats : (M, feature_dim * num_levels)
            Concatenation of trilinear features from all pyramid levels.
        """
        per_level = [lv.query_trilinear(query_pts) for lv in self.levels]
        return torch.cat(per_level, dim=-1)


# ====================================================================== #
#  Internal helper class (single resolution level)                       #
# ====================================================================== #
class _SingleResVoxelHashTable(nn.Module):
    """
    One hashed voxel grid with trilinear interpolation support.
    Also keeps track of collisions (construction-time) and access statistics
    (run-time).
    """
    def __init__(
        self,
        resolution: float,
        feature_dim: int,
        hash_table_size: int,
        scene_bound_min: tuple,
        scene_bound_max: tuple,
        primes: torch.Tensor,
        device: str,
    ):
        super().__init__()
        self.resolution      = resolution
        self.feature_dim     = feature_dim
        self.hash_table_size = hash_table_size
        self.primes          = primes
        self.device          = device

        # Build dense voxel coordinate list --------------------------------------------------
        xs = torch.arange(scene_bound_min[0], scene_bound_max[0], resolution, device=device)
        ys = torch.arange(scene_bound_min[1], scene_bound_max[1], resolution, device=device)
        zs = torch.arange(scene_bound_min[2], scene_bound_max[2], resolution, device=device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        self.voxel_coords  = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)  # (N, 3)
        self.total_voxels  = self.voxel_coords.shape[0]

        # Learnable feature table ------------------------------------------------------------
        self.voxel_features = nn.Parameter(
            torch.zeros(self.total_voxels, feature_dim, device=device).normal_(0, 0.01)
        )

        # Hash buckets (-1 means empty) -------------------------------------------------------
        self.register_buffer(
            "hash2voxel",
            torch.full((hash_table_size,), -1, dtype=torch.long, device=device),
        )

        # Build the hash table and compute collision stats
        self._fill_hashtable()

        # Runtime access log (Boolean mask, not a Parameter)
        self.register_buffer(
            "access_mask",
            torch.zeros(self.total_voxels, dtype=torch.bool, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    #  Construction helpers                                              #
    # ------------------------------------------------------------------ #
    def _fill_hashtable(self) -> None:
        """
        Inserts every voxel coordinate into the hash table.
        When collisions happen, the earliest voxel wins; later voxels remain
        'unreferenced' but are still kept in voxel_features.
        """
        idx_grid = torch.floor(self.voxel_coords / self.resolution).to(torch.int64)
        hv       = (idx_grid * self.primes).sum(dim=-1) % self.hash_table_size

        # Detect duplicates *before* insertion for statistics
        uniq, counts = hv.unique(return_counts=True)
        self.register_buffer(
            "collision_count",
            torch.tensor(int((counts > 1).sum()), device=self.device),
            persistent=False,
        )

        # First-come-first-serve insertion
        empty = self.hash2voxel[hv] == -1
        self.hash2voxel[hv[empty]] = torch.arange(self.total_voxels,
                                                  device=self.device)[empty]

    # ------------------------------------------------------------------ #
    #  Public utilities                                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collision_stats(self) -> dict:
        """
        Returns {total_voxels, hash_table_size, collisions, collision_ratio}.
        """
        collisions = int(self.collision_count.item())
        return {
            "total_voxels"     : self.total_voxels,
            "hash_table_size"  : self.hash_table_size,
            "collisions"       : collisions,
            "collision_ratio"  : collisions / self.total_voxels,
        }

    @torch.no_grad()
    def get_accessed_indices(self) -> torch.Tensor:
        """Returns a 1-D tensor of unique voxel indices that were touched."""
        return torch.nonzero(self.access_mask, as_tuple=False).flatten()

    @torch.no_grad()
    def reset_access_log(self) -> None:
        """Clears the per-voxel access mask."""
        self.access_mask.zero_()

    # ------------------------------------------------------------------ #
    #  Core lookup routines                                              #
    # ------------------------------------------------------------------ #
    def _lookup(self, idx_grid: torch.Tensor) -> torch.Tensor:
        """
        Returns the feature vectors for an int64 grid-coordinate tensor.
        Non-existent voxels yield zero vectors.
        """
        hv    = (idx_grid * self.primes).sum(dim=-1) % self.hash_table_size
        v_idx = self.hash2voxel[hv]            # (..., )
        valid = v_idx >= 0

        out = torch.zeros(*idx_grid.shape[:-1], self.feature_dim,
                          device=self.device, dtype=self.voxel_features.dtype)
        if valid.any():
            # Log the accessed voxel indices (for training diagnostics)
            self.access_mask[v_idx[valid]] = True
            out[valid] = self.voxel_features[v_idx[valid]]
        return out

    # ------------------------------------------------------------------ #
    #  Public query helper                                               #
    # ------------------------------------------------------------------ #
    def query_trilinear(self, query_pts: torch.Tensor) -> torch.Tensor:
        """
        Trilinear-interpolated feature for each 3-D world-space query point.
        """
        q_scaled = query_pts / self.resolution
        base_idx = torch.floor(q_scaled).to(torch.int64)            # (M, 3)
        frac     = q_scaled - base_idx.float()                      # (M, 3)

        # 8 binary corner offsets (0/1, 0/1, 0/1)
        offsets = torch.stack(
            torch.meshgrid(
                *(torch.tensor([0, 1], device=self.device, dtype=torch.int64)
                  for _ in range(3)),
                indexing="ij"
            ),
            dim=-1
        ).reshape(-1, 3)                                            # (8, 3)

        # Corner grid indices (M, 8, 3)
        corner_idx = base_idx[:, None, :] + offsets[None, :, :]

        # Corner features (M, 8, D)
        corner_feat = self._lookup(corner_idx)

        # Trilinear weights (M, 8)
        frac_exp    = frac.unsqueeze(1)             # (M, 1, 3)
        offsets_exp = offsets.unsqueeze(0).float()  # (1, 8, 3)
        w = torch.where(offsets_exp.bool(), frac_exp, 1.0 - frac_exp).prod(dim=2)

        return (corner_feat * w.unsqueeze(-1)).sum(dim=1)           # (M, D)
