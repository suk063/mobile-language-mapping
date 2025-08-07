import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dacite import from_dict
from omegaconf import OmegaConf
from torch.optim import Optimizer
from tqdm import tqdm

from lang_mapping.agent.agent_map_bc import Agent_map_bc, Agent_image_bc
from lang_mapping.utils.dataset import (
    DPDataset,
    build_object_map,
    get_object_labels_batch,
    build_uid_episode_scene_maps,
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
    max_image_cache_size: int
    num_dataload_workers: int
    trajs_per_obj: Union[str, int]
    torch_deterministic: bool

    vis_representation: str

    # Pretrained model paths
    static_map_path: str
    implicit_decoder_path: str

    # CLIP / Agent Settings
    clip_input_dim: int
    text_input: List[str]
    camera_intrinsics: List[float]
    bc_loss_weight: float
    num_heads: int
    num_layers_transformer: int
    num_action_layer: int
    action_pred_horizon: int
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
        # -------------------------------------------------------------
        # Handle resuming logic
        # -------------------------------------------------------------
        if self.resume_logdir is not None:
            # Explicit resume directory provided by the user
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"

            # Load the old config so that we can reuse logging paths, wandb id, etc.
            old_config = get_mshab_train_cfg(
                parse_cfg(default_cfg_path=old_config_path)
            )
            # Reuse the exact same experiment directories so that checkpoints/logs
            # are preserved. Most importantly, disable `clear_out` so that the
            # logger does NOT delete the existing directory when it is
            # re-initialised.
            self.logger.workspace = old_config.logger.workspace
            self.logger.exp_path = old_config.logger.exp_path
            self.logger.log_path = old_config.logger.log_path
            self.logger.model_path = old_config.logger.model_path
            self.logger.train_video_path = old_config.logger.train_video_path
            self.logger.eval_video_path = old_config.logger.eval_video_path
            # Make sure we never wipe previous outputs when resuming.
            self.logger.clear_out = False

            # Reuse W&B information so that the run is properly resumed.
            if self.wandb_id is None and old_config.wandb_id is not None:
                self.wandb_id = old_config.wandb_id
            # If the previous run was using W&B, ensure it is enabled now as well.
            if old_config.logger.wandb:
                self.logger.wandb = True

            # By default, resume from the latest checkpoint if the caller did not
            # specify an explicit checkpoint path.
            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "ckpt_latest.pt"

        # ------------------------------------------------------------------
        # Automatic resume even when `resume_logdir` is not provided.
        # ------------------------------------------------------------------
        if self.model_ckpt is None:
            # ------------------------------------------------------------------
            # Scan the model directory for available checkpoints and pick the one
            # with the largest epoch number. Preference order:
            #   1. Highest `ckpt_epoch_*.pt`
            #   2. Fallback to `ckpt_latest.pt`
            # ------------------------------------------------------------------
            backup_ckpts = list(self.logger.model_path.glob("ckpt_epoch_*.pt"))
            latest_backup_ckpt = None
            if backup_ckpts:
                # Extract epoch numbers using regex and find the max
                def _epoch_num(path):
                    m = re.search(r"ckpt_epoch_(\d+)\.", path.name)
                    return int(m.group(1)) if m else -1
                latest_backup_ckpt = max(backup_ckpts, key=_epoch_num)

            latest_ckpt = self.logger.model_path / "ckpt_latest.pt"

            # Decide which checkpoint to use
            candidate_ckpts = []
            if latest_backup_ckpt is not None:
                candidate_ckpts.append(latest_backup_ckpt)
            if latest_ckpt.exists():
                candidate_ckpts.append(latest_ckpt)

            if candidate_ckpts:
                # Choose the checkpoint whose stored epoch is the largest.
                best_ckpt = None
                best_epoch = -1
                for ckpt_path in candidate_ckpts:
                    try:
                        epoch_val = torch.load(ckpt_path, map_location="cpu")["epoch"]
                        if epoch_val > best_epoch:
                            best_epoch = epoch_val
                            best_ckpt = ckpt_path
                    except Exception as e:
                        print(f"[Warn] Failed to read epoch from {ckpt_path}: {e}")
                # Fallback if epoch field missing or unreadable
                if best_ckpt is None:
                    best_ckpt = max(candidate_ckpts, key=lambda p: p.stat().st_mtime)

                self.model_ckpt = best_ckpt
                # Make sure we do not delete previous results.
                self.logger.clear_out = False

        # Validate the checkpoint path if we have decided to resume.
        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: dict) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))





def setup_models_and_optimizer(
    cfg: TrainConfig, device: torch.device, sample_obs, single_act_shape, vis_representation
) -> Tuple[Agent_map_bc, Optimizer]:

    if vis_representation == "map":
        # We use fixed hyperparams for the static map and implicit decoder
        static_maps = MultiVoxelHashTable.load_sparse(cfg.algo.static_map_path).to(device)

        # (NOTE) Hardcoded hyperparams for the implicit decoder
        voxel_feature_dim = 128
        hidden_dim = 240

        implicit_decoder = ImplicitDecoder(
            voxel_feature_dim=voxel_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=cfg.algo.clip_input_dim,
        ).to(device)

        implicit_decoder.load_state_dict(
            torch.load(cfg.algo.implicit_decoder_path, map_location=device)["model"],
            strict=True,
        )

        for param in static_maps.parameters():
            param.requires_grad = False
        for param in implicit_decoder.parameters():
            param.requires_grad = False

        agent = Agent_map_bc(
            sample_obs=sample_obs,
            single_act_shape=single_act_shape,
            transf_input_dim=cfg.algo.transf_input_dim,
            clip_input_dim=cfg.algo.clip_input_dim,
            text_input=cfg.algo.text_input,
            camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
            static_maps=static_maps,
            implicit_decoder=implicit_decoder,
            num_heads=cfg.algo.num_heads,
            num_layers_transformer=cfg.algo.num_layers_transformer,
            num_action_layer=cfg.algo.num_action_layer,
            action_pred_horizon=cfg.algo.action_pred_horizon,
        ).to(device)

    else:
        raise ValueError(f"Invalid vis_representation: {vis_representation}")

    params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.algo.lr)

    return agent, optimizer


def train_one_epoch(
    agent: Agent_map_bc,
    optimizer: Optimizer,
    dataloader: ClosableDataLoader,
    device: torch.device,
    cfg: BCConfig,
    uid_to_label_map: Dict,
    uid2scene_id: Dict,
    time_weights: torch.Tensor,
    global_step: int,
    logger: Logger,
) -> Tuple[float, int]:
    tot_loss, n_samples = 0.0, 0
    agent.train()

    for batch in tqdm(dataloader, desc="Batch", unit="batch"):
        obs = batch["observations"]  # dict
        act = batch["actions"]  # (pred_horizon, A)
        subtask_uids = batch["subtask_uid"]  # str

        subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(
            device
        )
        scene_ids = torch.tensor(
            [uid2scene_id[uid] for uid in subtask_uids],
            device=device,
            dtype=torch.long,
        )

        obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
            act, device=device, dtype="float"
        )

        pi = agent(obs, subtask_labels, scene_ids)

        total_bc_loss = F.smooth_l1_loss(pi, act, reduction="none")
        total_bc_loss = total_bc_loss * time_weights
        bc_loss = total_bc_loss.mean()

        bc_loss = bc_loss * cfg.bc_loss_weight
        loss = bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = act.size(0)
        tot_loss += loss.item() * batch_size
        n_samples += batch_size
        global_step += 1
        
        logger.store(tag="train_iter", bc_loss=bc_loss.item())
        logger.log(global_step)

    return tot_loss / n_samples if n_samples > 0 else 0.0, global_step


def evaluate_agent(
    agent: Agent_map_bc,
    eval_envs,
    fixed_plan_idxs,
    uid_to_label_map,
    uid2scene_id,
    logger,
    device,
    global_epoch,
):
    agent.eval()
    eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})

    print("Run eval episode (single horizon)")
    stats_single = run_eval_episode(
        eval_envs, eval_obs, agent, uid_to_label_map, uid2scene_id, device
    )
    _pretty_print_stats("[Eval-Single]", stats_single, logger, color="yellow")

    logger.store(tag="eval", success_once=stats_single["success_once"])
    logger.store(tag="eval", return_per_step=stats_single["return_per_step"])
    logger.log(global_epoch)


def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid_to_label_map = build_object_map(cfg.eval_env.task_plan_fp, cfg.algo.text_input)
    _, uid2scene_id = build_uid_episode_scene_maps(cfg.eval_env.task_plan_fp, 'pretrained/scene_ids.yaml')

    # Make eval env
    print("Making eval env...")
    eval_envs = make_env(cfg.eval_env, video_path=cfg.logger.eval_video_path)
    print("Eval env made.")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    fixed_plan_idxs = eval_envs.unwrapped.task_plan_idxs.clone()
    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)

    agent, optimizer = setup_models_and_optimizer(
        cfg, device, eval_obs, eval_envs.unwrapped.single_action_space.shape, cfg.algo.vis_representation
    )

    def save(save_path):
        torch.save(
            dict(
                agent=agent.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                global_step=global_step,
            ),
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(str(load_path), map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    start_epoch = 0
    global_step = 0
    if cfg.model_ckpt and cfg.model_ckpt.exists():
        print(f"Resuming from checkpoint: {cfg.model_ckpt}")
        resumed_epoch, resumed_global_step = load(cfg.model_ckpt)
        start_epoch = resumed_epoch + 1
        global_step = resumed_global_step
        print(f"Resumed from epoch {resumed_epoch}. Starting at epoch {start_epoch}.")

    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"
    bc_dataset = DPDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=1,
        pred_horizon=cfg.algo.action_pred_horizon,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset, 
        batch_size=cfg.algo.batch_size, 
        shuffle=True, 
        num_workers=cfg.algo.num_dataload_workers, 
        pin_memory=True,
        persistent_workers=(cfg.algo.num_dataload_workers > 0),
        drop_last=True,
    )

    # Note: global_step will be used directly for consistent logging

    def check_freq(freq):
        return epoch % freq == 0

    def store_env_stats(key):
        assert key == "eval", "Only eval env for BC"
        log_env = eval_envs
        logger.store(
            key,
            return_per_step=torch.tensor(log_env.return_queue, device=device).float().mean() / log_env.max_episode_steps,
            success_once=torch.tensor(log_env.success_once_queue, device=device).float().mean(),
            success_at_end=torch.tensor(log_env.success_at_end_queue, device=device).float().mean(),
            len=torch.tensor(log_env.length_queue, device=device).float().mean(),
        )
        log_env.reset_queues()

    print("Start training...")
    timer = NonOverlappingTimeProfiler()

    time_weights = exp_decay_weights(
        torch.arange(cfg.algo.action_pred_horizon, device=device),
        cfg.algo.action_temp_weights,
    ).view(1, -1, 1)

    for epoch in range(start_epoch, cfg.algo.epochs):

        if epoch >= cfg.algo.epochs:
            break
        logger.print(
            f"Epoch: {epoch}; Global step: {global_step}"
        )

        avg_loss, global_step = train_one_epoch(
            agent,
            optimizer,
            bc_dataloader,
            device,
            cfg.algo,
            uid_to_label_map,
            uid2scene_id,
            time_weights,
            global_step,
            logger,
        )
        timer.end(key="train")

        # Log
        if check_freq(cfg.algo.log_freq):
            logger.store(tag="losses", loss=avg_loss)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(global_step)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq:
            if check_freq(cfg.algo.eval_freq):
                evaluate_agent(
                    agent,
                    eval_envs,
                    fixed_plan_idxs,
                    uid_to_label_map,
                    uid2scene_id,
                    logger,
                    device,
                    global_step,
                )
                timer.end(key="eval")

        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{epoch}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
