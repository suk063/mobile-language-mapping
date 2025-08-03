import torch, torch.nn as nn
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #
def _primes(dev):  # 3-tuple of large primes
    return torch.tensor([73856093, 19349669, 83492791], device=dev, dtype=torch.long)


def _corner_offsets(dev):  # (8,3) corner offsets
    return torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=dev,
        dtype=torch.long,
    )


# --------------------------------------------------------------------------- #
#  dense level (train)                                                        #
# --------------------------------------------------------------------------- #
class _TrainLevel(nn.Module):
    def __init__(self, res, d, buckets, smin, smax, primes, dev):
        super().__init__()
        self.res, self.d, self.buckets = res, d, buckets
        self.smin = torch.tensor(smin, device=dev, dtype=torch.float)
        self.smax = torch.tensor(smax, device=dev, dtype=torch.float)

        self.register_buffer("primes", primes, persistent=False)
        self.primes: torch.Tensor

        xs = torch.arange(smin[0], smax[0], res, device=dev)
        ys = torch.arange(smin[1], smax[1], res, device=dev)
        zs = torch.arange(smin[2], smax[2], res, device=dev)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")

        self.register_buffer("coords", torch.stack([gx, gy, gz], -1).view(-1, 3), persistent=False)
        self.coords: torch.Tensor
        self.N = self.coords.size(0)
        
        self.voxel_features = nn.Parameter(torch.zeros(self.N, d, device=dev).normal_(0, 0.01))

        self.register_buffer("hash2vox", torch.full((buckets,), -1, dtype=torch.long, device=dev))
        self._fill()
        self.register_buffer("access", torch.zeros(self.N, dtype=torch.bool, device=dev), persistent=False)

    def _fill(self):
        idx = torch.floor((self.coords - self.smin) / self.res).long()
        hv = (idx * self.primes).sum(-1) % self.buckets
        empty = self.hash2vox[hv] == -1
        self.hash2vox[hv[empty]] = torch.arange(self.N, device=self.voxel_features.device)[empty]
        dup = hv.unique(return_counts=True)[1] > 1
        self.register_buffer("col", torch.tensor(int(dup.sum()), device=self.voxel_features.device), persistent=False)
        logging.info(f"Level filled: {self.N} voxels, {self.col} collisions")

    # ---------- public utils
    @torch.no_grad()  # short stats
    def collision_stats(self):
        return dict(total=self.N, col=int(self.col))

    @torch.no_grad()
    def get_accessed_indices(self):
        return torch.nonzero(self.access).flatten()

    @torch.no_grad()  # clear log
    def reset_access_log(self):
        self.access.zero_()

    @torch.no_grad()  # sparse dump
    def export_sparse(self):
        used = self.get_accessed_indices()
        return dict(
            resolution=self.res,
            coords=self.coords[used].cpu(),
            features=self.voxel_features[used].cpu(),
            smin=self.smin.cpu(),
            smax=self.smax.cpu(),
        )

    # ---------- internals
    def _lookup(self, idxg):
        hv = (idxg * self.primes).sum(-1) % self.buckets
        vid = self.hash2vox[hv]
        valid = vid >= 0
        out = torch.zeros(*idxg.shape[:-1], self.d, device=self.voxel_features.device, dtype=self.voxel_features.dtype)
        if valid.any():
            self.access[vid[valid]] = True
            out[valid] = self.voxel_features[vid[valid]]
        return out

    def query(self, pts):
        q, offs = (pts - self.smin) / self.res, _corner_offsets(pts.device)
        base = torch.floor(q).long()
        frac = q - base.float()
        idx = base[:, None, :] + offs[None, :, :]
        feat = self._lookup(idx)

        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], 1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], 1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], 1)
        w = wx[:, [0, 1, 0, 1, 0, 1, 0, 1]] * wy[:, [0, 0, 1, 1, 0, 0, 1, 1]] * wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
        return (feat * w.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------- #
#  sparse level (infer)                                                       #
# --------------------------------------------------------------------------- #
class _InferLevel(nn.Module):
    def __init__(self, pay, d, buckets, primes, dev):
        super().__init__()
        self.res, self.d, self.buckets, self.primes = float(pay["resolution"]), d, buckets, primes
        coords, feats = pay["coords"].to(dev), pay["features"].to(dev)
        self.register_buffer("coords", coords, persistent=False)
        self.voxel_features = nn.Parameter(feats, requires_grad=False)
        # Use provided scene bounds if available, else fall back to coords min/max
        self.smin = torch.tensor(pay['smin'], device=dev).float()
        self.smax = torch.tensor(pay['smax'], device=dev).float()

        self.register_buffer("hash2vox", torch.full((buckets,), -1, dtype=torch.long, device=dev), persistent=False)
        idx = torch.floor((coords - self.smin) / self.res).long()
        hv = (idx * self.primes).sum(-1) % buckets

        # detect collisions by counting duplicate hash values
        dup = hv.unique(return_counts=True)[1] > 1
        self.register_buffer("col", torch.tensor(int(dup.sum()), device=dev), persistent=False)

        # Sunghwan:log collisions and total voxels for debugging
        logging.info(f"[InferLevel] Initialized with {coords.size(0)} voxels, {int(dup.sum())} collisions")

        self.hash2vox[hv] = torch.arange(coords.size(0), device=dev)

    # short stats
    def collision_stats(self):
        return dict(total=self.coords.size(0), col=int(self.col))

    def get_accessed_indices(self):
        return torch.empty(0, dtype=torch.long, device=self.coords.device)

    def reset_access_log(self):
        pass

    def _lookup(self, idxg):
        hv = (idxg * self.primes).sum(-1) % self.buckets
        vid = self.hash2vox[hv]
        valid = vid >= 0
        out = torch.zeros(*idxg.shape[:-1], self.d, device=self.coords.device, dtype=self.voxel_features.dtype)
        if valid.any():
            out[valid] = self.voxel_features[vid[valid]]
        return out

    def query(self, pts):
        q, offs = (pts - self.smin) / self.res, _corner_offsets(pts.device)
        base = torch.floor(q).long()
        frac = q - base.float()
        idx = base[:, None, :] + offs[None, :, :]
        feat = self._lookup(idx)

        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], 1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], 1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], 1)
        w = wx[:, [0, 1, 0, 1, 0, 1, 0, 1]] * wy[:, [0, 0, 1, 1, 0, 0, 1, 1]] * wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
        return (feat * w.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------- #
#  public pyramid                                                             #
# --------------------------------------------------------------------------- #
class VoxelHashTable(nn.Module):
    """
    mode='train' → dense levels, mode='infer' → sparse levels
    """

    def __init__(
        self,
        resolution: float = 0.12,
        num_levels: int = 2,
        level_scale: float = 2.0,
        feature_dim: int = 32,
        hash_table_size: int = 2**21,
        scene_bound_min: Tuple[float, float, float] = (-2.6, -8.1, 0),
        scene_bound_max: Tuple[float, float, float] = (4.6, 4.7, 3.1),
        device: str = "cuda:0",
        mode: str = "train",
        sparse_data: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode, self.d = mode, feature_dim
        dev = torch.device(device)
        primes = _primes(dev)
        self.levels = nn.ModuleList()

        if mode == "train":
            # Iterate coarse → fine by reversing the exponent.
            for lv in range(num_levels):
                res = resolution * (level_scale ** (num_levels - 1 - lv))
                self.levels.append(
                    _TrainLevel(res, feature_dim, hash_table_size, scene_bound_min, scene_bound_max, primes, dev)
                )
        elif mode == "infer":
            if sparse_data is None:
                raise ValueError("sparse_data is required in infer mode")
            # Sort payloads from coarse (larger res) → fine (smaller res)
            sorted_levels = sorted(sparse_data["levels"], key=lambda p: p["resolution"], reverse=True)
            for pay in sorted_levels:
                self.levels.append(_InferLevel(pay, feature_dim, hash_table_size, primes, dev))
        else:
            raise ValueError("mode must be 'train' or 'infer'")

    # forward -----------------------------------------------------------------
    def query_voxel_feature(self, pts):  # (M,3) → (M, d*L)
        per = [lv.query(pts) for lv in self.levels]
        return torch.cat(per, -1)

    # utils -------------------------------------------------------------------
    @torch.no_grad()
    def collision_stats(self):
        return {f"level_{i}": lv.collision_stats() for i, lv in enumerate(self.levels)}

    @torch.no_grad()
    def get_accessed_indices(self):
        return [lv.get_accessed_indices() for lv in self.levels]

    @torch.no_grad()
    def reset_access_log(self):
        for lv in self.levels:
            lv.reset_access_log()

    # save / load -------------------------------------------------------------
    @torch.no_grad()
    def export_sparse(self):
        if self.mode != "train":
            raise RuntimeError("export_sparse only in train mode")
        return dict(num_levels=len(self.levels), feature_dim=self.d, levels=[lv.export_sparse() for lv in self.levels])

    # dense weight file
    def save_dense(self, path):
        torch.save({"state_dict": self.state_dict()}, path)

    # sparse file
    def save_sparse(self, path):
        torch.save(self.export_sparse(), path)

    @staticmethod
    def load_dense(path, device="cuda:0"):
        chk = torch.load(path, map_location="cpu")
        vt = VoxelHashTable(device=device)  # default ctor, train mode
        vt.load_state_dict(chk["state_dict"])
        return vt.to(device)

    @staticmethod
    def load_sparse(path, device="cuda:0"):
        sparse = torch.load(path, map_location="cpu")
        return VoxelHashTable(mode="infer", sparse_data=sparse, device=device)


class MultiVoxelHashTable(nn.Module):
    def __init__(
        self,
        n_scenes: int,
        resolution: float = 0.12,
        num_levels: int = 2,
        level_scale: float = 2.0,
        feature_dim: int = 64,
        hash_table_size: int = 2**21,
        scene_bound_min: list[float] = [-2.6, -8.1, 0],
        scene_bound_max: list[float] = [4.6, 4.7, 3.1],
        mode: str = "train",
        sparse_data: Optional[List[Dict]] = None,
    ):
        super(MultiVoxelHashTable, self).__init__()
        self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.n_scenes = n_scenes
        self.resolution = resolution
        self.num_levels = num_levels
        self.level_scale = level_scale
        self.feature_dim = feature_dim
        self.hash_table_size = hash_table_size
        self.scene_bound_min = scene_bound_min
        self.scene_bound_max = scene_bound_max
        self.mode = mode

        self.voxel_hash_tables = nn.ModuleList()
        for i in range(n_scenes):
            self.voxel_hash_tables.append(
                VoxelHashTable(
                    resolution=resolution,
                    num_levels=num_levels,
                    level_scale=level_scale,
                    feature_dim=feature_dim,
                    hash_table_size=hash_table_size,
                    scene_bound_min=scene_bound_min,
                    scene_bound_max=scene_bound_max,
                    device="cpu",
                    mode=mode,
                    sparse_data=sparse_data[i] if sparse_data is not None else None,
                )
            )

    def distribute_to_devices(self):
        for scene_id in range(self.n_scenes):
            device = self.devices[scene_id % len(self.devices)]
            self.voxel_hash_tables[scene_id].to(device)

    def query_feature(self, x: torch.Tensor, scene_id: torch.Tensor) -> torch.Tensor:
        """
        x: (N,3)
        scene_id: (N,) long tensor indicating which scene each point belongs to
        returns: (N, d*L) features
        """
        scene_id = scene_id.squeeze()  # Ensure shape is (M,)
        N = x.shape[0]
        output_dim = self.num_levels * self.feature_dim
        all_feats = torch.zeros(N, output_dim, device=x.device, dtype=x.dtype)

        unique_scenes = torch.unique(scene_id)

        for s_id_tensor in unique_scenes:
            s_id = s_id_tensor.item()
            if not (0 <= s_id < self.n_scenes):
                logger.error(f"Invalid scene_id {s_id} encountered in query batch.")
                continue
            mask = scene_id == s_id_tensor
            device = self.devices[s_id % len(self.devices)]
            x_scene = x[mask]
            if x_scene.shape[0] == 0:
                continue
            # Move x_scene to the appropriate device
            x_scene = x_scene.to(device)
            # Query the voxel feature
            voxel_hash_table: VoxelHashTable = self.voxel_hash_tables[s_id]
            feats = voxel_hash_table.query_voxel_feature(x_scene)
            # Move feats back to the output tensor on the original device
            all_feats[mask] = feats.to(all_feats.device)
        return all_feats

    def save_sparse(self, path: str):
        sparse_data = {f"{i}": self.voxel_hash_tables[i].export_sparse() for i in range(self.n_scenes)}
        torch.save(
            {
                "state_dict": sparse_data,
                "n_scenes": self.n_scenes,
                "resolution": self.resolution,
                "num_levels": self.num_levels,
                "level_scale": self.level_scale,
                "feature_dim": self.feature_dim,
                "hash_table_size": self.hash_table_size,
                "scene_bound_min": self.scene_bound_min,
                "scene_bound_max": self.scene_bound_max,
            },
            path,
        )

    def save_dense(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_scenes": self.n_scenes,
                "resolution": self.resolution,
                "num_levels": self.num_levels,
                "level_scale": self.level_scale,
                "feature_dim": self.feature_dim,
                "hash_table_size": self.hash_table_size,
                "scene_bound_min": self.scene_bound_min,
                "scene_bound_max": self.scene_bound_max,
                "mode": self.mode,
            },
            path,
        )

    @staticmethod
    def load_sparse(path: str):
        sparse_data = torch.load(path, map_location="cpu")
        n_scenes = len(sparse_data)
        return MultiVoxelHashTable(
            n_scenes=n_scenes,
            resolution=sparse_data["resolution"],
            num_levels=sparse_data["num_levels"],
            level_scale=sparse_data["level_scale"],
            feature_dim=sparse_data["feature_dim"],
            hash_table_size=sparse_data["hash_table_size"],
            scene_bound_min=sparse_data["scene_bound_min"],
            scene_bound_max=sparse_data["scene_bound_max"],
            mode="infer",
            sparse_data=[sparse_data[str(i)] for i in range(n_scenes)],
        )

    @staticmethod
    def load_dense(path: str):
        dense_data = torch.load(path, map_location="cpu")
        model = MultiVoxelHashTable(
            n_scenes=dense_data["n_scenes"],
            resolution=dense_data["resolution"],
            num_levels=dense_data["num_levels"],
            level_scale=dense_data["level_scale"],
            feature_dim=dense_data["feature_dim"],
            hash_table_size=dense_data["hash_table_size"],
            scene_bound_min=dense_data["scene_bound_min"],
            scene_bound_max=dense_data["scene_bound_max"],
            mode="train",
        )
        model.load_state_dict(dense_data["state_dict"])
        return model
