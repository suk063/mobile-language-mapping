import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm
import random

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lang_mapping.grid_net import GridNet
from lang_mapping.agent.agent_global_multistep_gridnet_rel import Agent_global_multistep_gridnet_rel
from lang_mapping.module import ImplicitDecoder
from lang_mapping.dataset import TempTranslateToPointDataset

from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler

@dataclass
class GridDefinition:
    type: str = "regular"
    feature_dim: int = 120
    init_stddev: float = 0.2
    bound: List[List[float]] = field(
        default_factory=lambda: [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]]
    )
    base_cell_size: float = 0.5
    per_level_scale: float = 5.0
    n_levels: int = 2
    second_order_grid_sample: bool = False


@dataclass
class GridCfg:
    name: str = "grid_net"
    spatial_dim: int = 3
    grid: GridDefinition = field(default_factory=GridDefinition)


@dataclass
class BCConfig:
    name: str = "bc"

    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 1

    eval_freq: int = 1
    log_freq: int = 1
    save_freq: int = 1
    save_backup_ckpts: bool = False

    data_dir_fp: str = None
    max_cache_size: int = 0
    trajs_per_obj: Union[str, int] = "all"
    torch_deterministic: bool = True

    # CLIP / Agent Settings
    clip_input_dim: int = 768
    open_clip_model_name: str = "EVA02-L-14"
    open_clip_model_pretrained: str = "merged2b_s4b_b131k"
    text_input: List[str] = field(default_factory=lambda: ["bowl", "apple"])
    camera_intrinsics: List[float] = field(default_factory=lambda: [71.9144, 71.9144, 112, 112])
    hidden_dim: int = 240
    cos_loss_weight: float = 0.1
    pe_type: str = "none"
    pe_level: int = 10

    num_eval_envs: int = field(init=False)
    grid_cfg: GridCfg = field(default_factory=GridCfg)

    def _additional_processing(self):
        assert self.name == "bc"
        try:
            self.trajs_per_obj = int(self.trajs_per_obj)
        except:
            pass
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: BCConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "If resuming, logger paths must match"
            else:
                assert old_config_path.exists(), f"No old config at {old_config_path}"
                old_config = get_mshab_train_cfg(parse_cfg(default_cfg_path=old_config_path))
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))

def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Make eval env
    print("Making eval env...")
    eval_envs = make_env(cfg.eval_env, video_path=cfg.logger.eval_video_path)
    print("Eval env made.")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)

    dummy_grid = GridNet(cfg=asdict(cfg.algo.grid_cfg), device=device)
    
    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=cfg.algo.grid_cfg.grid.feature_dim * cfg.algo.grid_cfg.grid.n_levels,
        hidden_dim=cfg.algo.hidden_dim,
        output_dim=cfg.algo.clip_input_dim,
        L=cfg.algo.pe_level,
        pe_type=cfg.algo.pe_type
    ).to(device)
    
    # Agent
    agent = Agent_global_multistep_gridnet_rel(
        device=device,
        open_clip_model=(cfg.algo.open_clip_model_name, cfg.algo.open_clip_model_pretrained),
        camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
        static_map=dummy_grid,
        implicit_decoder=implicit_decoder,
    ).to(device)
    
    logger = Logger(logger_cfg=cfg.logger, save_fn=None)
    writer = SummaryWriter(log_dir=cfg.logger.log_path)
    
    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"

    global_step = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0
    print("Start training...")

    timer = NonOverlappingTimeProfiler()

    EPOCHS_PER_TRAJ = 30

    def check_freq(freq, epoch):
        return (epoch % freq) == 0

    # ------------------------------------------------
    # Mapping
    # ------------------------------------------------
    for t_idx in range(1000):
        dataset_single = TempTranslateToPointDataset(
            data_path=cfg.algo.data_dir_fp,
            obs_horizon=2,
            pred_horizon=2,
            control_mode=eval_envs.unwrapped.control_mode,
            trajs_per_obj=cfg.algo.trajs_per_obj,
            max_image_cache_size=cfg.algo.max_cache_size,
            truncate_trajectories_at_success=True,
            cat_state=cfg.eval_env.cat_state,
            cat_pixels=cfg.eval_env.cat_pixels,
            single_traj_idx=t_idx, 
        )
        
        dloader_single = ClosableDataLoader(
            dataset_single,
            batch_size=cfg.algo.batch_size,
            shuffle=True,
            num_workers=0
        )

        new_grid_net = GridNet(cfg=asdict(cfg.algo.grid_cfg), device=device)
        agent.static_map = new_grid_net

        for name, param in agent.clip_model.named_parameters():
            param.requires_grad = False 
        
        params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
        optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.algo.lr)
        
        agent.to(device)

        print(f"\n=== Train traj_idx = {t_idx} ===")

        for epoch in range(EPOCHS_PER_TRAJ):
            agent.train()
            tot_loss, n_samples = 0.0, 0

            for obs, act, subtask_uids, traj_idx_batch, is_grasped in tqdm(
                dloader_single,
                desc=f"[traj={t_idx}] epoch={epoch}",
                unit="batch"
            ):
                obs = to_tensor(obs, device=device, dtype="float")
                act = to_tensor(act, device=device, dtype="float")
                is_grasped = to_tensor(is_grasped, device=device, dtype="float")

                total_cos_loss = agent.forward_mapping(obs, is_grasped)
                if total_cos_loss == 0:
                    continue

                cos_loss = total_cos_loss * cfg.algo.cos_loss_weight
                loss = cos_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                n_samples += act.size(0)
                global_step += 1

                writer.add_scalar(f"Loss/traj_{t_idx}", loss.item(), global_step)

            avg_loss = tot_loss / n_samples

            print(f"[traj={t_idx}] epoch={epoch}, avg_loss={avg_loss:.6f}")
            timer.end(key="train")
            
        dloader_single.close()
        dataset_single.close()

    print("Done training for all trajectories.")

    final_save_path = logger.model_path / "implicit_decoder_final.pt"
    torch.save(agent.implicit_decoder.state_dict(), final_save_path)
    print(f"Final implicit_decoder saved at {final_save_path}")

    eval_envs.close()
    logger.close()
    writer.close()

if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)