import os
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Union

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ManiSkill imports
from mani_skill.utils import common

from lang_mapping.agent.agent_3dencoder import Agent_3dencoder
from lang_mapping.dataset import TempTranslateToPointDataset, build_object_map, get_object_labels_batch, merge_t_m1
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler

def run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map):
    """
    Run a single evaluation episode.
    Uses 'prev_obs' and 'current_obs' for merging observations.
    """
    device = eval_obs["state"].device
    max_steps = eval_envs.max_episode_steps
    prev_obs = eval_obs  # For the first step, prev_obs = current_obs = reset result

    # Get subtask info (labels and indices) for the episode
    plan0 = eval_envs.unwrapped.task_plan[0]
    subtask_labels = get_object_labels_batch(uid_to_label_map, plan0.composite_subtask_uids).to(device)

    # Batch size (number of parallel evaluation environments)
    B = subtask_labels.size(0)

    for t in range(max_steps):
        # Merge observations from time t and t-1
        agent_obs = merge_t_m1(prev_obs, eval_obs)

        with torch.no_grad():
            action = agent(agent_obs, subtask_labels)

        # Environment step
        next_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])
        prev_obs = eval_obs
        eval_obs = next_obs

    return eval_obs
          
@dataclass
class BCConfig:
    name: str = "bc"
    lr: float = 3e-4               # learning rate
    batch_size: int = 256          # batch size
    epochs: int = 2         # epochs

    eval_freq: int = 1
    log_freq: int = 1
    save_freq: int = 1
    save_backup_ckpts: bool = False

    data_dir_fp: str = None        # path to data .h5 files
    max_cache_size: int = 0        # max data points to cache
    trajs_per_obj: Union[str, int] = "all"
    torch_deterministic: bool = True

    # Voxel/Scene Settings
    voxel_feature_dim: int = 240
    resolution: float = 0.12
    hash_table_size: int = 2**21
    scene_bound_min: List[float] = field(default_factory=lambda: [-2.6, -8.1, 0.0])
    scene_bound_max: List[float] = field(default_factory=lambda: [4.6, 4.7, 3.1])

    # CLIP / Agent Settings
    clip_input_dim: int = 768
    open_clip_model_name: str = "EVA02-L-14"
    open_clip_model_pretrained: str = "merged2b_s4b_b131k"
    text_input: List[str] = field(default_factory=lambda: ["bowl", "apple"])
    camera_intrinsics: List[float] = field(default_factory=lambda: [71.9144, 71.9144, 112, 112])
    hidden_dim: int = 240
    num_heads: int = 8
    num_layers_transformer: int = 4
    action_horizon: int = 16
    scaling_factor: float = 0.3
    bc_loss_weights: float = 10.0
    voxel_feature_dim: int = 240

    num_eval_envs: int = field(init=False)

    # Pre-trained weights
    pretrained_agent_path: str = None
    pretrained_optimizer_path: str = None

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
    uid_to_label_map = build_object_map(cfg.eval_env.task_plan_fp, cfg.algo.text_input)

    # Make eval env
    print("Making eval env...")
    eval_envs = make_env(cfg.eval_env, video_path=cfg.logger.eval_video_path)
    # eval_uids = eval_envs.unwrapped.task_plan[0].composite_subtask_uids
    print("Eval env made.")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    # MARK: Here we try to save the plan indexes
    fixed_plan_idxs = eval_envs.unwrapped.task_plan_idxs.clone()
    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)

    # Agent
    agent = Agent_3dencoder(
        sample_obs=eval_obs,
        single_act_shape=eval_envs.unwrapped.single_action_space.shape,
        device=device,
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
        open_clip_model=(cfg.algo.open_clip_model_name, cfg.algo.open_clip_model_pretrained),
        text_input=cfg.algo.text_input,
        clip_input_dim=cfg.algo.clip_input_dim,
        camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
        num_heads = cfg.algo.num_heads,
        num_layers_transformer = cfg.algo.num_layers_transformer,
    ).to(device)

    if cfg.algo.pretrained_agent_path is not None and os.path.exists(cfg.algo.pretrained_agent_path):
        print(f"[INFO] Loading pretrained agent from {cfg.algo.pretrained_agent_path}")
        agent.load_state_dict(torch.load(cfg.algo.pretrained_agent_path, map_location=device))

    logger = Logger(logger_cfg=cfg.logger, save_fn=None)
    writer = SummaryWriter(log_dir=cfg.logger.log_path)

    def save_checkpoint(name="latest"):
        """
        Save the agent, voxel table, decoder, and optimizer states.
        """
        agent_path = logger.model_path / f"{name}_agent.pt"
        optim_path = logger.model_path / f"{name}_optimizer.pt"

        torch.save(agent.state_dict(), agent_path)
        torch.save(optimizer.state_dict(), optim_path)

    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"
    
    action_horizon = cfg.algo.action_horizon
    
    # Create BC dataset and dataloader
    bc_dataset = TempTranslateToPointDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=2,
        pred_horizon=action_horizon+1,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_cache_size,
        truncate_trajectories_at_success=True,
        cat_state=cfg.eval_env.cat_state,
        cat_pixels=cfg.eval_env.cat_pixels,
    )

    bc_dataloader = ClosableDataLoader(
        bc_dataset, batch_size=cfg.algo.batch_size, shuffle=True, num_workers=0
    )

    global_step = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0
    print("Start training...")

    timer = NonOverlappingTimeProfiler()

    def check_freq(freq, epoch):
        return (epoch % freq) == 0

    def store_env_stats(key):
        """
        Store env stats in logger (evaluation only).
        """

        if key == "eval":
            log_env = eval_envs
        else:
            raise ValueError(f"Unsupported key: {key}")
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device).float().mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device).float().mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device).float().mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        log_env.reset_queues()
        
    # dict to keep track of the totals outside the loop
    agg = dict(ret_sum=0.0, suc1_sum=0.0, sucE_sum=0.0, len_sum=0.0, n=0)
    
    # Helper method to flush the stats
    def flush_env_stats(env):
        r  = torch.tensor(env.return_queue,          dtype=torch.float32)
        s1 = torch.tensor(env.success_once_queue,    dtype=torch.float32)
        sE = torch.tensor(env.success_at_end_queue,  dtype=torch.float32)
        L  = torch.tensor(env.length_queue,          dtype=torch.float32)

        agg["ret_sum"] += r.sum().item()
        agg["suc1_sum"] += s1.sum().item()
        agg["sucE_sum"] += sE.sum().item()
        agg["len_sum"] += L.sum().item()
        agg["n"]       += r.numel()

        # clear for next env or next chunk
        env.reset_queues()    

    # ------------------------------------------------
    # Stage 1:  Policy only (BC loss)
    # ------------------------------------------------
    
    for name, param in agent.named_parameters():
        param.requires_grad = True 
    
    for name, param in agent.clip_model.named_parameters():
        param.requires_grad = False 
    
    params_to_optimize = agent.parameters()
    optimizer = torch.optim.Adam(params_to_optimize, lr=cfg.algo.lr)
    
    agent.to(device)

    alpha = cfg.algo.scaling_factor 
    time_indices = torch.arange(action_horizon).to(device)           # [0, 1, 2, ..., horizon-1]
    time_weights = torch.exp(-alpha * time_indices)                  # exp(-alpha * i)
    time_weights = time_weights / time_weights.sum()
    time_weights = time_weights.view(1, action_horizon, 1)

    for epoch in range(cfg.algo.epochs):
        global_epoch = logger_start_log_step + epoch
        
        logger.print(f"[Stage 1] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.train()

        for obs, act, subtask_uids, _, _ in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi = agent(obs, subtask_labels)
            raw_bc_loss = F.smooth_l1_loss(pi, act, reduction='none')
            
            weighted_bc_loss = raw_bc_loss * time_weights
            bc_loss = cfg.algo.bc_loss_weights * weighted_bc_loss.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()

            tot_loss += bc_loss.item()
            n_samples += act.size(0)
            global_step += 1

            writer.add_scalar("BC Loss/Iteration", bc_loss.item(), global_step)

        avg_loss = tot_loss / n_samples
        loss_logs = dict(loss=avg_loss)
        timer.end(key="train")

        # Logging
        if check_freq(cfg.algo.log_freq, epoch):
            logger.store(tag="losses", **loss_logs)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(global_epoch)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq and check_freq(cfg.algo.eval_freq, epoch):
            agent.eval()

            # If not last epoch
            if epoch < (cfg.algo.epochs - 1):
                eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})
                # DEBUG
                # for i, plan in enumerate(eval_envs.unwrapped.task_plan):
                #     print(f"[Eval Env {i}] subtask UIDs = {plan.composite_subtask_uids}")
                run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map)
                # Final stats
                if len(eval_envs.return_queue) > 0:
                    store_env_stats("eval")
                logger.log(global_epoch)
                timer.end(key="eval")
            # For last epoch, run all task plans in chunks
            else:
                # For now we run subset like the previous epochs and run eval on all tasks seperately
                print("Running normal fixed-plan eval for last epoch...")
                eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})
                # DEBUG
                # for i, plan in enumerate(eval_envs.unwrapped.task_plan):
                #     print(f"[Eval Env {i}] subtask UIDs = {plan.composite_subtask_uids}")
                run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map)
                if len(eval_envs.return_queue) > 0:
                    store_env_stats("eval")
                logger.log(global_epoch)

                batch_size = eval_envs.num_envs
                all_plan_count = cfg.eval_env.all_plan_count
                all_plan_idxs_list = list(range(all_plan_count))

                print("Now running all tasks in chunks...")
                pbar = tqdm(total=all_plan_count, desc="Evaluating all tasks (last epoch)")

                chunk_start = 0
                while chunk_start < all_plan_count:
                    chunk_end = min(chunk_start + batch_size, all_plan_count)
                    chunk_size = chunk_end - chunk_start
                    chunk = all_plan_idxs_list[chunk_start:chunk_end]

                    # MARK: (woojeh) If it's a last chunk (maybe smaller than batch_size) than we flush and rebuild the env
                    if chunk_size < batch_size:
                        flush_env_stats(eval_envs)
                        eval_envs.close()
                        # # DEBUG
                        # print(f"We create last env with {chunk_size} envs.")
                        temp_cfg = replace(cfg.eval_env, num_envs=chunk_size)
                        eval_envs = make_env(temp_cfg, video_path=cfg.logger.eval_video_path)
                        batch_size = chunk_size

                    plan_idxs_tensor = torch.tensor(chunk, dtype=torch.int)

                    eval_obs, info = eval_envs.reset(options={"task_plan_idxs": plan_idxs_tensor})

                    # DEBUG
                    # for i, plan in enumerate(eval_envs.unwrapped.task_plan):
                    #     print(f"[Eval Env {i}] subtask UIDs = {plan.composite_subtask_uids}")

                    run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map)
                    
                    flush_env_stats(eval_envs)
                    
                    chunk_start += batch_size
                    pbar.update(chunk_size)

                pbar.close() 

                # Done with all tasks so we store from the agg
                logger.store(
                    "eval_all",
                    return_per_step = agg["ret_sum"]/agg["len_sum"],
                    success_once    = agg["suc1_sum"]/agg["n"],
                    success_at_end  = agg["sucE_sum"]/agg["n"],
                    len         = agg["len_sum"]/agg["n"],
                    len_sum             = agg["len_sum"],
                    n               = agg["n"],
                )
                logger.log(global_epoch)
                timer.end(key="eval")

        # Saving
        if check_freq(cfg.algo.save_freq, epoch):
            save_checkpoint(name="latest")
            timer.end(key="checkpoint")

    # Final save
    save_checkpoint(name="stage1-final")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()
    writer.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
