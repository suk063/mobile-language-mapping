from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridDefinition:
    type: str = "regular"
    spatial_dim: int = 3
    feature_dim: int = 60
    init_stddev: float = 0.2
    bound: list[list[float]] = field(default_factory=lambda: [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]])
    base_cell_size: float = 0.4
    per_level_scale: float = 2.0
    n_levels: int = 2


@dataclass
class VoxelHashTableConfig:
    one_to_one: bool  # whether to use one-to-one mapping for voxel features
    resolution: float  # finest cell size (e.g. 0.12)
    num_levels: int  # pyramid depth (e.g. 2)
    level_scale: float  # ratio between levels (e.g. 2.0)
    voxel_feature_dim: int  # per-level feature width (e.g. 32)
    hash_table_size: int  # buckets per level (power of two)
    scene_bound_min: list[float]  # xyz lower corner
    scene_bound_max: list[float]  # xyz upper corner


@dataclass
class GridCfg:
    name: str = "grid_net"  # grid_net or voxel_hash_table
    spatial_dim: int = 3
    n_scenes: int = 161
    grid_net: Optional[GridDefinition] = None
    voxel_hash_table: Optional[VoxelHashTableConfig] = None

    def as_dict(self):
        out = vars(self)
        if self.grid_net is not None:
            out["grid_net"] = vars(self.grid_net)
        if self.voxel_hash_table is not None:
            out["voxel_hash_table"] = vars(self.voxel_hash_table)
        return out
