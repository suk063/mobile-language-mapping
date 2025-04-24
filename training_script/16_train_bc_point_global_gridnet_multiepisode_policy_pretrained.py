import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from lang_mapping.grid_net import GridNet

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm
import open3d as o3d
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ManiSkill imports
import mani_skill.envs
from mani_skill.utils import common

from lang_mapping.agent.agent_global_gridnet_multiepisode_pretrained import Agent_global_gridnet_multiepisode_pretrained
from lang_mapping.module import ImplicitDecoder
from lang_mapping.dataset import TempTranslateToPointDataset, merge_t_m1, build_uid_mapping, build_object_map, get_object_labels_batch

from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler
import copy


def build_episode_subtask_maps(
    task_plan_fp: str,
    keep_dict_order: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int]]:
    """
    Returns
    -------
    episode2subtasks : {episode_name(str) : [uid, ...]}
    episode2id       : {episode_name(str) : episode_id(int)}
    uid2episode_id   : {subtask_uid(str)  : episode_id(int)}
    """
    with open(task_plan_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    episode2subtasks = defaultdict(list)
    for plan in data["plans"]:
        # `train/set_table/episode_1010.json`  ->  `episode_1010`
        ep_name = os.path.splitext(os.path.basename(plan["init_config_name"]))[0]
        for st in plan["subtasks"]:
            episode2subtasks[ep_name].extend(st["composite_subtask_uids"])

    # episode → int id
    ep_names      = list(episode2subtasks) if keep_dict_order else sorted(episode2subtasks)
    episode2id    = {ep: i for i, ep in enumerate(ep_names)}

    # uid → episode int id
    uid2episode_id = {}
    for ep, uid_list in episode2subtasks.items():
        eid = episode2id[ep]
        for uid in uid_list:
            uid2episode_id[uid] = eid

    return dict(episode2subtasks), episode2id, uid2episode_id

@dataclass
class GridDefinition:
    type: str = "regular"
    feature_dim: int = 60
    init_stddev: float = 0.2
    bound: List[List[float]] = field(
        default_factory=lambda: [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]]
    )
    base_cell_size: float = 0.4
    per_level_scale: float = 2.0
    n_levels: int = 2
    n_scenes: int = 122
    second_order_grid_sample: bool = False

@dataclass
class GridCfg:
    name: str = "grid_net"
    spatial_dim: int = 3
    grid: GridDefinition = field(default_factory=GridDefinition)           
                 
@dataclass
class BCConfig:
    name: str = "bc"
    lr: float = 3e-4               # learning rate
    batch_size: int = 256          # batch size
    epochs: int = 1         # stage 1 epochs

    eval_freq: int = 1
    log_freq: int = 1
    save_freq: int = 1
    save_backup_ckpts: bool = False

    data_dir_fp: str = None        # path to data .h5 files
    max_cache_size: int = 0        # max data points to cache
    trajs_per_obj: Union[str, int] = "all"
    torch_deterministic: bool = True

    # CLIP / Agent Settings
    clip_input_dim: int = 768
    open_clip_model_name: str = "EVA02-L-14"
    open_clip_model_pretrained: str = "merged2b_s4b_b131k"
    text_input: List[str] = field(default_factory=lambda: ["bowl", "apple"])
    camera_intrinsics: List[float] = field(default_factory=lambda: [71.9144, 71.9144, 112, 112])
    state_mlp_dim: int = 1024
    hidden_dim: int = 512
    cos_loss_weight: float = 0.1
    num_heads: int = 8
    num_layers_transformer: int = 4
    num_layers_perceiver: int = 4
    num_learnable_tokens: int = 32
    num_action_layer: int = 6
    pe_level: int = 10
    action_horizon: int = 16
    neighbor_k: int = 512
    action_temp_weights: float = 0.25
    transf_input_dim: int = 768

    num_eval_envs: int = field(init=False)
    grid_cfg: GridCfg = field(default_factory=GridCfg)

    # Pre-trained weights
    pretrained_agent_path: str = None
    pretrained_voxel_path:str = None
    pretrained_implicit_path: str = None
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

def recursive_h5py_to_numpy(h5py_obs, slice=None):
    if isinstance(h5py_obs, h5py.Group) or isinstance(h5py_obs, dict):
        return dict(
            (k, recursive_h5py_to_numpy(h5py_obs[k], slice)) for k in h5py_obs.keys()
        )
    if isinstance(h5py_obs, list):
        return [recursive_h5py_to_numpy(x, slice) for x in h5py_obs]
    if isinstance(h5py_obs, tuple):
        return tuple(recursive_h5py_to_numpy(x, slice) for x in h5py_obs)
    if slice is not None:
        return h5py_obs[slice]
    return h5py_obs[:]

def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid_to_label_map = build_object_map(cfg.eval_env.task_plan_fp, cfg.algo.text_input)
    uid2index_map = build_uid_mapping(cfg.eval_env.task_plan_fp)
    
    episode2subtasks, episode2id, uid2episode_id = build_episode_subtask_maps(cfg.eval_env.task_plan_fp)
    num_episodes = len(episode2id)
    
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

    # VoxelHashTable and ImplicitDecoder
    static_map = GridNet(cfg=asdict(cfg.algo.grid_cfg))
    
    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=cfg.algo.grid_cfg.grid.feature_dim * cfg.algo.grid_cfg.grid.n_levels,
        hidden_dim=cfg.algo.hidden_dim,
        output_dim=cfg.algo.clip_input_dim,
    ).to(device)

    # Agent
    agent = Agent_global_gridnet_multiepisode_pretrained(
        sample_obs=eval_obs,
        single_act_shape=eval_envs.unwrapped.single_action_space.shape,
        device=device,
        transf_input_dim=cfg.algo.transf_input_dim,
        open_clip_model=(cfg.algo.open_clip_model_name, cfg.algo.open_clip_model_pretrained),
        text_input=cfg.algo.text_input,
        clip_input_dim=cfg.algo.clip_input_dim,
        state_mlp_dim=cfg.algo.state_mlp_dim,
        camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
        static_map=static_map,
        implicit_decoder=implicit_decoder,
        num_heads = cfg.algo.num_heads,
        num_layers_transformer=cfg.algo.num_layers_transformer,
        num_layers_perceiver=cfg.algo.num_layers_perceiver,
        num_learnable_tokens=cfg.algo.num_learnable_tokens,
        num_action_layer=cfg.algo.num_action_layer,
        neighbor_k=cfg.algo.neighbor_k,
        action_horizon=cfg.algo.action_horizon
    ).to(device) 
    
    # agent.load_state_dict(torch.load('pre-trained/latest_agent.pt', map_location=device), strict=True)

    agent.static_map.load_state_dict(torch.load('pre-trained/latest_static_map.pt', mmap=True), strict=True)
    agent.implicit_decoder.load_state_dict(torch.load('pre-trained/latest_decoder.pt', map_location=device), strict=True)
    
    
    def load_changed_centers(
        root_dir: str = "pre-trained",
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[int, dict[int, torch.Tensor]]:
        """
        Returns
        -------
        {level → {scene_id → (N,3) torch.Tensor}}
        """
        centers: dict[int, dict[int, torch.Tensor]] = {}

        for fp in sorted(Path(root_dir).glob("level*_centers.npz")):
            lvl = int(fp.stem.split("_")[0].lstrip("level"))
            data = np.load(fp, allow_pickle=True)

            # numpy → torch(Tensor)
            centers[lvl] = {
                int(k): torch.as_tensor(data[k], dtype=dtype, device=device)
                for k in data.files
            }

        if not centers:
            raise FileNotFoundError(f"No center files found in {root_dir}")
        return centers
        
    agent.valid_coords = load_changed_centers(device=device)[1]

    logger = Logger(logger_cfg=cfg.logger, save_fn=None)
    writer = SummaryWriter(log_dir=cfg.logger.log_path)

    def save_checkpoint(name="latest"):
        """
        Save the agent, voxel table, decoder, and optimizer states.
        """
        static_map_path = logger.model_path / f"{name}_static_map.pt"
        decoder_path = logger.model_path / f"{name}_decoder.pt"
        agent_path = logger.model_path / f"{name}_agent.pt"

        # torch.save(static_map.state_dict(), static_map_path)
        # torch.save(implicit_decoder.state_dict(), decoder_path)
        torch.save(agent.state_dict(), agent_path)

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
        elif key == "eval_all":
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


    # ------------------------------------------------
    # Mapping
    # ------------------------------------------------
    for name, param in agent.named_parameters():
        param.requires_grad = True 
    
    for name, param in agent.clip_model.named_parameters():
        param.requires_grad = False 
        
    for name, param in agent.static_map.named_parameters():
        param.requires_grad = False
        
    for name, param in agent.implicit_decoder.named_parameters():
        param.requires_grad = False  
    
    alpha = cfg.algo.action_temp_weights
    time_indices = torch.arange(action_horizon).to(device)           # [0, 1, 2, ..., horizon-1]
    time_weights = torch.exp(-alpha * time_indices)                  # exp(-alpha * i)
    time_weights = time_weights / time_weights.sum()
    time_weights = time_weights.view(1, action_horizon, 1)
    

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
        epi_ids = torch.tensor(
            [uid2episode_id[uid] for uid in plan0.composite_subtask_uids],
            device=device,
            dtype=torch.long,
        )    
        
        # Batch size (number of parallel evaluation environments)
        B = subtask_labels.size(0)

        for t in range(max_steps):
            # Merge observations from time t and t-1
            agent_obs = merge_t_m1(prev_obs, eval_obs)

            with torch.no_grad():
                action = agent.forward_policy(agent_obs, subtask_labels, epi_ids)

            # Environment step
            next_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])
            prev_obs = eval_obs
            eval_obs = next_obs

        return eval_obs
    
    params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=cfg.algo.lr)

    agent.to(device)

    # Training loop
    for epoch in range(cfg.algo.epochs):
        global_epoch = logger_start_log_step + epoch
        
        logger.print(f"[Stage 1] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.train()

        for obs, act, subtask_uids, traj_idx, is_grasped in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            epi_ids = torch.tensor(
                [uid2episode_id[uid] for uid in subtask_uids],
                device=device,
                dtype=torch.long,
            )    
            
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi = agent.forward_policy(obs, subtask_labels, epi_ids)
            
            total_bc_loss = F.smooth_l1_loss(pi, act, reduction='none')
            weighted_bc_loss = total_bc_loss * time_weights
            bc_loss = 10 * weighted_bc_loss.mean()
        
            loss = bc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})
            # DEBUG
            for i, plan in enumerate(eval_envs.unwrapped.task_plan):
                print(f"[Eval Env {i}] subtask UIDs = {plan.composite_subtask_uids}")
            run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map)
            # Final stats
            if len(eval_envs.return_queue) > 0:
                store_env_stats("eval")
            logger.log(global_epoch)
            timer.end(key="eval")

        # Saving
        if check_freq(cfg.algo.save_freq, epoch):
            save_checkpoint(name="latest")
            timer.end(key="checkpoint")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()
    writer.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)