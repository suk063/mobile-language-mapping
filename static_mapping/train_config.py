from dataclasses import dataclass
from typing import Optional

from dataset import DataConfig
from grid_cfg import GridCfg


@dataclass
class ClipModelConfig:
    model_name: str = "EVA02-L-14"
    model_pretrained: str = "merged2b_s4b_b131k"


@dataclass
class TrainConfig:
    seed: int
    torch_deterministic: bool
    device_clip: str
    device_decoder: str
    epochs: int
    optimizer: str
    optimizer_kwargs: dict
    data: DataConfig
    clip_model: ClipModelConfig
    grid_cfg: GridCfg
    depth_downsample_method: str  # "nearest-exact", "nearest", "avg2d", "avg3d"
    decoder_hidden_dim: int
    decoder_output_dim: int
    output_dir: str
    valid_interval: int
    ckpt_interval: int
    test_model_dir: Optional[str]

    def as_dict(self):
        out = vars(self)
        out["data"] = vars(self.data)
        out["clip_model"] = vars(self.clip_model)
        out["grid_cfg"] = self.grid_cfg.as_dict()
        return out
