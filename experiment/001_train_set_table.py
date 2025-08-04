import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dacite import from_dict
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mani_skill.envs
from lang_mapping.agent.agent_map_bc import Agent_map_bc
from lang_mapping.utils.dataset import (
    DPDataset,
    build_object_map,
    get_object_labels_batch,
    build_episode_subtask_maps,
)
from lang_mapping.mapper.mapper import MultiVoxelHashTable
from lang_mapping.module import ImplicitDecoder
from lang_mapping.utils.utils import exp_decay_weights
from lang_mapping.utils.eval import run_eval_episode, _pretty_print_stats
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler

@dataclass
class BCConfig:
    name: str
    lr: float
    batch_size: int
    epochs: int

    eval_freq: int
    log_freq: int
    save_freq: int
    save_backup_ckpts: bool

    data_dir_fp: Optional[str]
    max_cache_size: int
    trajs_per_obj: Union[str, int]
    torch_deterministic: bool

    # Pretrained model paths
    static_map_path: str
    implicit_decoder_path: str

    # CLIP / Agent Settings
    clip_input_dim: int
    open_clip_model_name: str
    open_clip_model_pretrained: str
    text_input: List[str]
    camera_intrinsics: List[float]
    cos_loss_weight: float
    bc_loss_weight: float
    num_heads: int
    hidden_dim: int
    num_layers_transformer: int
    num_action_layer: int
    action_pred_horizon: int
    neighbor_k: int
    action_temp_weights: float
    transf_input_dim: int

    num_eval_envs: int = field(init=False)

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
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
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

def save_checkpoint(agent, logger, name="latest"):
    """
    Save the agent, voxel table, decoder, and optimizer states.
    """
    agent_path = logger.model_path / f"{name}_agent.pt"
    torch.save(agent.state_dict(), agent_path)

def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid_to_label_map = build_object_map(
        cfg.eval_env.task_plan_fp, cfg.algo.text_input
    )
    _episode2subtasks, _episode2id, uid2episode_id = build_episode_subtask_maps(
        cfg.eval_env.task_plan_fp
    )

    # Make eval env
    print("Making eval env...")
    eval_envs = make_env(cfg.eval_env, video_path=cfg.logger.eval_video_path)
    print("Eval env made.")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    # MARK: Here we try to save the plan indexes
    fixed_plan_idxs = eval_envs.unwrapped.task_plan_idxs.clone()

    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)


    # We use fixed hyperparams for the static map and implicit decoder
    static_maps = MultiVoxelHashTable.load_sparse(cfg.algo.static_map_path).to(device)

    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=128,
        hidden_dim=240,
        output_dim=cfg.algo.clip_input_dim,
    ).to(device)

    # Agent
    agent = Agent_map_bc(
        sample_obs=eval_obs,
        single_act_shape=eval_envs.unwrapped.single_action_space.shape,
        device=device,
        transf_input_dim=cfg.algo.transf_input_dim,
        open_clip_model=(
            cfg.algo.open_clip_model_name,
            cfg.algo.open_clip_model_pretrained,
        ),
        text_input=cfg.algo.text_input,
        clip_input_dim=cfg.algo.clip_input_dim,
        camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
        static_map=None,
        implicit_decoder=None,
        num_heads=cfg.algo.num_heads,
        num_layers_transformer=cfg.algo.num_layers_transformer,
        num_action_layer=cfg.algo.num_action_layer,
        action_pred_horizon=cfg.algo.action_pred_horizon,
    ).to(device)

    agent.implicit_decoder = implicit_decoder

    implicit_decoder.load_state_dict(
        torch.load(cfg.algo.implicit_decoder_path, map_location=device)["model"],
        strict=True,
    )

    logger = Logger(logger_cfg=cfg.logger, save_fn=None)
    writer = SummaryWriter(log_dir=cfg.logger.log_path)

    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"
    action_pred_horizon = cfg.algo.action_pred_horizon

    # Create BC dataset and dataloader
    bc_dataset = DPDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=1,
        pred_horizon=action_pred_horizon,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_cache_size,
        truncate_trajectories_at_success=True,
    )

    bc_dataloader = ClosableDataLoader(
        bc_dataset, batch_size=cfg.algo.batch_size, shuffle=True, num_workers=0
    )

    global_step = 0
    logger_start_log_step = (
        logger.last_log_step + 1 if logger.last_log_step > 0 else 0
    )
    print("Start training...")

    timer = NonOverlappingTimeProfiler()

    for param in static_maps.parameters():
        param.requires_grad = False

    for param in implicit_decoder.parameters():
        param.requires_grad = False

    time_weights = exp_decay_weights(
        torch.arange(cfg.algo.action_pred_horizon, device=device),
        cfg.algo.action_temp_weights,
    ).view(1, -1, 1)

    params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.algo.lr)

    # Training loop
    for epoch in range(cfg.algo.epochs):
        global_epoch = logger_start_log_step + epoch

        logger.print(f"[Stage 1] Epoch: {global_epoch}")
        tot_loss, n_samples = 0.0, 0
        agent.train()

        for batch in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            obs = batch["observations"]  # dict
            act = batch["actions"]  # (pred_horizon, A)
            subtask_uids = batch["subtask_uid"]  # str

            subtask_labels = get_object_labels_batch(
                uid_to_label_map, subtask_uids
            ).to(device)
            epi_ids = torch.tensor(
                [uid2episode_id[uid] for uid in subtask_uids],
                device=device,
                dtype=torch.long,
            )

            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
                act, device=device, dtype="float"
            )

            pi = agent.forward(obs, subtask_labels, epi_ids)

            total_bc_loss = F.smooth_l1_loss(pi, act, reduction="none")
            total_bc_loss = total_bc_loss * time_weights
            bc_loss = total_bc_loss.mean()

            bc_loss = bc_loss * cfg.algo.bc_loss_weight
            loss = bc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = act.size(0)
            tot_loss += loss.item() * batch_size
            n_samples += batch_size
            global_step += 1

            writer.add_scalar("BC Loss/Iteration", bc_loss.item(), global_step)

        avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
        loss_logs = dict(loss=avg_loss)
        timer.end(key="train")

        # Logging
        if (epoch % cfg.algo.log_freq) == 0:
            logger.store(tag="losses", **loss_logs)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(global_epoch)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq and (epoch % cfg.algo.eval_freq) == 0:
            agent.eval()
            eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})

            print("Run eval episode (single horizon)")
            stats_single = run_eval_episode(
                eval_envs, eval_obs, agent, uid_to_label_map, uid2episode_id
            )
            _pretty_print_stats("[Eval-Single]", stats_single, logger, color="yellow")

            if len(eval_envs.return_queue) > 0:
                store_eval_stats(logger, eval_envs, device)
            logger.log(global_epoch)
            timer.end(key="eval")

        # Saving
        if (epoch % cfg.algo.save_freq) == 0:
            save_checkpoint(agent, logger, name=f"latest_{epoch}")
            timer.end(key="checkpoint")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()
    writer.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
