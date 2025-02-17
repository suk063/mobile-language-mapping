import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
import mani_skill.envs
from mani_skill.utils import common

from lang_mapping.agent.agent_dynamic import Agent_point_dynamic
from lang_mapping.module import ImplicitDecoder
from lang_mapping.mapper.mapper_dynamic import VoxelHashTableDynamic
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


def build_object_map(json_file_path: str, object_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Build a label map from subtask_uid to object label. 
    Returns an empty dict if no JSON file is found.
    """
    if not os.path.exists(json_file_path):
        print(f"[Warning] subtask map JSON not found: {json_file_path}")
        return {}

    with open(json_file_path, "r") as f:
        data = json.load(f)

    uid_to_label_map = {}
    for plan in data.get("plans", []):
        for subtask in plan.get("subtasks", []):
            subtask_uid = subtask["uid"]
            obj_id = subtask["obj_id"]

            found_label = None
            for i, obj_name in enumerate(object_names):
                if obj_name in obj_id:
                    found_label = i
                    break

            if found_label is None:
                raise ValueError(f"Unsupported object_id: {obj_id}")

            uid_to_label_map[subtask_uid] = torch.tensor(found_label, dtype=torch.long)

    return uid_to_label_map


def get_object_labels_batch(
    object_map: Dict[str, torch.Tensor], uids: List[str]
) -> torch.Tensor:
    """
    Return a tensor of labels for a batch of uids. 
    If a uid is not found in the map, label = -1.
    """
    labels = []
    for uid in uids:
        if uid not in object_map:
            labels.append(torch.tensor(-1, dtype=torch.long))
        else:
            labels.append(object_map[uid])
    return torch.stack(labels, dim=0)


@dataclass
class BCConfig:
    name: str = "bc"
    lr: float = 3e-4               # learning rate
    batch_size: int = 256          # batch size
    stage1_epochs: int = 2         # stage 1 epochs
    stage2_epochs: int = 6        # stage 2 epochs
    stage3_epochs: int = 2        # stage 3 epochs
    eval_freq: int = 1
    log_freq: int = 1
    save_freq: int = 1
    save_backup_ckpts: bool = False

    data_dir_fp: str = None        # path to data .h5 files
    max_cache_size: int = 0        # max data points to cache
    trajs_per_obj: Union[str, int] = "all"
    torch_deterministic: bool = True

    # Voxel/Scene Settings
    voxel_feature_dim: int = 120
    resolution: float = 0.12
    hash_table_size: int = 2**21
    scene_bound_min: List[float] = field(default_factory=lambda: [-2.6, -8.1, 0.0])
    scene_bound_max: List[float] = field(default_factory=lambda: [4.6, 4.7, 3.1])
    mod_time: int = 10

    # CLIP / Agent Settings
    clip_input_dim: int = 768
    open_clip_model_name: str = "EVA02-L-14"
    open_clip_model_pretrained: str = "merged2b_s4b_b131k"
    text_input: List[str] = field(default_factory=lambda: ["bowl", "apple"])
    camera_intrinsics: List[float] = field(default_factory=lambda: [71.9144, 71.9144, 112, 112])
    state_mlp_dim: int = 1024
    stage1_cos_loss_weight: float = 0.5
    stage2_cos_loss_weight: float = 0.005
    stage2_linear_scheduling: bool = True
    dynamic_only: bool = True

    num_eval_envs: int = field(init=False)

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


def recursive_tensor_size_bytes(obj):
    """
    Calculate total size of nested tensors in bytes.
    """
    extra_obj_size = 0
    if isinstance(obj, dict):
        extra_obj_size = sum([recursive_tensor_size_bytes(v) for v in obj.values()])
    elif isinstance(obj, (list, tuple)):
        extra_obj_size = sum([recursive_tensor_size_bytes(x) for x in obj])
    elif isinstance(obj, torch.Tensor):
        extra_obj_size = obj.nelement() * obj.element_size()
    return sys.getsizeof(obj) + extra_obj_size


class BCDataset(ClosableDataset):
    """
    Dataset for Behavior Cloning. 
    Loads data from specified .h5 files, potentially caching samples.
    """
    def __init__(
        self,
        data_dir_fp: str,
        max_cache_size: int,
        transform_fn=torch.from_numpy,
        trajs_per_obj: Union[str, int] = "all",
        cat_state=True,
        cat_pixels=False,
    ):
        data_dir_fp: Path = Path(data_dir_fp)
        self.data_files: List[h5py.File] = []
        self.json_files: List[Dict] = []
        self.obj_names_in_loaded_order: List[str] = []

        if data_dir_fp.is_file():
            data_file_names = [data_dir_fp.name]
            data_dir_fp = data_dir_fp.parent
        else:
            data_file_names = os.listdir(data_dir_fp)

        for data_fn in data_file_names:
            if data_fn.endswith(".h5"):
                json_fn = data_fn.replace(".h5", ".json")
                self.data_files.append(h5py.File(data_dir_fp / data_fn, "r"))
                with open(data_dir_fp / json_fn, "rb") as f:
                    self.json_files.append(json.load(f))
                self.obj_names_in_loaded_order.append(data_fn.replace(".h5", ""))

        self.dataset_idx_to_data_idx = {}
        dataset_idx = 0

        # Each h5/json pair is assumed to hold multiple trajectories
        for file_idx, json_file in enumerate(self.json_files):
            if trajs_per_obj == "all":
                use_ep_jsons = json_file["episodes"]
            else:
                assert trajs_per_obj <= len(json_file["episodes"]), (
                    f"Requested {trajs_per_obj} but found only "
                    f"{len(json_file['episodes'])} episodes."
                )
                use_ep_jsons = random.sample(json_file["episodes"], k=trajs_per_obj)

            for ep_json in use_ep_jsons:
                ep_id = ep_json["episode_id"]
                subtask_uid = ep_json["subtask_uid"]
                for step in range(ep_json["elapsed_steps"]):
                    self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step, subtask_uid)
                    dataset_idx += 1

        self._data_len = dataset_idx
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.transform_fn = transform_fn
        self.cat_state = cat_state
        self.cat_pixels = cat_pixels

    def transform_idx(self, x, data_index):
        if isinstance(x, (h5py.Group, dict)):
            return {k: self.transform_idx(v, data_index) for k, v in x.items()}
        out = self.transform_fn(np.array(x[data_index]))
        if len(out.shape) == 0:
            out = out.unsqueeze(0)
        return out

    def get_single_item(self, index):
        if index in self.cache:
            return self.cache[index]

        file_num, ep_num, step_num, subtask_uid = self.dataset_idx_to_data_idx[index]
        ep_data = self.data_files[file_num][f"traj_{ep_num}"]
        observation = ep_data["obs"]

        agent_obs = self.transform_idx(observation["agent"], step_num)
        extra_obs = self.transform_idx(observation["extra"], step_num)

        # Depth and RGB frames for each camera, unsqueezed to shape (1, H, W)
        fetch_hand_depth = self.transform_idx(
            observation["sensor_data"]["fetch_hand"]["depth"], step_num
        ).squeeze(-1).unsqueeze(0)
        fetch_hand_rgb = self.transform_idx(
            observation["sensor_data"]["fetch_hand"]["rgb"], step_num
        ).squeeze(-1).unsqueeze(0)

        fetch_head_depth = self.transform_idx(
            observation["sensor_data"]["fetch_head"]["depth"], step_num
        ).squeeze(-1).unsqueeze(0)
        fetch_head_rgb = self.transform_idx(
            observation["sensor_data"]["fetch_head"]["rgb"], step_num
        ).squeeze(-1).unsqueeze(0)

        # Camera poses
        fetch_hand_pose = self.transform_idx(
            observation["sensor_param"]["fetch_hand"]["extrinsic_cv"], step_num
        ).squeeze(-1).unsqueeze(0)

        fetch_head_pose = self.transform_idx(
            observation["sensor_param"]["fetch_head"]["extrinsic_cv"], step_num
        ).squeeze(-1).unsqueeze(0)

        if self.cat_state:
            state_obs = {
                "state": torch.cat(
                    [
                        *agent_obs.values(),
                        *extra_obs.values(),
                    ],
                    dim=0,
                )
            }
        else:
            state_obs = {"agent_obs": agent_obs, "extra_obs": extra_obs}

        if self.cat_pixels:
            pixel_obs = {
                "all_depth": torch.stack([fetch_head_depth, fetch_head_rgb], dim=-3)
            }
        else:
            pixel_obs = {
                "fetch_hand_depth": fetch_hand_depth,
                "fetch_hand_rgb": fetch_hand_rgb,
                "fetch_head_depth": fetch_head_depth,
                "fetch_head_rgb": fetch_head_rgb,
                "fetch_hand_pose": fetch_hand_pose,
                "fetch_head_pose": fetch_head_pose,
            }

        obs = {**state_obs, "pixels": pixel_obs}
        act = self.transform_idx(ep_data["actions"], step_num)

        data_point = (obs, act, subtask_uid, step_num)
        if len(self.cache) < self.max_cache_size:
            self.cache[index] = data_point
        return data_point

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            return self.get_single_item(indexes)
        return [self.get_single_item(i) for i in indexes]

    def __len__(self):
        return self._data_len

    def close(self):
        for f in self.data_files:
            f.close()


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
    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)

    # VoxelHashTable and ImplicitDecoder
    hash_voxel = VoxelHashTableDynamic(
        resolution=cfg.algo.resolution,
        hash_table_size=cfg.algo.hash_table_size,
        feature_dim=cfg.algo.voxel_feature_dim,
        scene_bound_min=tuple(cfg.algo.scene_bound_min),
        scene_bound_max=tuple(cfg.algo.scene_bound_max),
        mod_time= cfg.algo.mod_time,
        dynamaic_only=cfg.algo.dynamic_only,
        device=device
    ).to(device)

    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
        hidden_dim=256,
        output_dim=cfg.algo.clip_input_dim,
        L=10
    ).to(device)

    # Agent
    agent = Agent_point_dynamic(
        sample_obs=eval_obs,
        single_act_shape=eval_envs.unwrapped.single_action_space.shape,
        device=device,
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
        open_clip_model=(cfg.algo.open_clip_model_name, cfg.algo.open_clip_model_pretrained),
        text_input=cfg.algo.text_input,
        clip_input_dim=cfg.algo.clip_input_dim,
        state_mlp_dim=cfg.algo.state_mlp_dim,
        camera_intrinsics=tuple(cfg.algo.camera_intrinsics),
        max_time_steps=eval_envs.max_episode_steps + 1,
        hash_voxel=hash_voxel,
        implicit_decoder=implicit_decoder
    ).to(device)

    # Combine parameters
    params_to_optimize = (
        list(agent.parameters())
        + list(hash_voxel.parameters())
        + list(implicit_decoder.parameters())
    )
    optimizer = torch.optim.Adam(params_to_optimize, lr=cfg.algo.lr)

    if cfg.algo.pretrained_agent_path is not None and os.path.exists(cfg.algo.pretrained_agent_path):
        print(f"[INFO] Loading pretrained agent from {cfg.algo.pretrained_agent_path}")
        agent.load_state_dict(torch.load(cfg.algo.pretrained_agent_path, map_location=device))

    if cfg.algo.pretrained_voxel_path is not None and os.path.exists(cfg.algo.pretrained_voxel_path):
        print(f"[INFO] Loading pretrained voxel from {cfg.algo.pretrained_voxel_path}")
        hash_voxel.load_state_dict(torch.load(cfg.algo.pretrained_voxel_path, map_location=device))

    if cfg.algo.pretrained_implicit_path is not None and os.path.exists(cfg.algo.pretrained_implicit_path):
        print(f"[INFO] Loading pretrained implicit decoder from {cfg.algo.pretrained_implicit_path}")
        implicit_decoder.load_state_dict(torch.load(cfg.algo.pretrained_implicit_path, map_location=device))

    if cfg.algo.pretrained_optimizer_path is not None and os.path.exists(cfg.algo.pretrained_optimizer_path):
        print(f"[INFO] Loading pretrained optimizer state from {cfg.algo.pretrained_optimizer_path}")
        optimizer.load_state_dict(torch.load(cfg.algo.pretrained_optimizer_path, map_location=device))

    logger = Logger(logger_cfg=cfg.logger, save_fn=None)
    writer = SummaryWriter(log_dir=cfg.logger.log_path)

    def save_checkpoint(name="latest"):
        """
        Save the agent, voxel table, decoder, and optimizer states.
        """
        agent_path = logger.model_path / f"{name}_agent.pt"
        voxel_path = logger.model_path / f"{name}_voxel.pt"
        decoder_path = logger.model_path / f"{name}_implicit.pt"
        optim_path = logger.model_path / f"{name}_optimizer.pt"

        torch.save(agent.state_dict(), agent_path)
        torch.save(hash_voxel.state_dict(), voxel_path)
        torch.save(implicit_decoder.state_dict(), decoder_path)
        torch.save(optimizer.state_dict(), optim_path)

    # Create BC dataset and dataloader
    bc_dataset = BCDataset(
        cfg.algo.data_dir_fp,
        cfg.algo.max_cache_size,
        cat_state=cfg.eval_env.cat_state,
        cat_pixels=cfg.eval_env.cat_pixels,
        trajs_per_obj=cfg.algo.trajs_per_obj,
    )
    logger.print(
        f"BC Dataset: {len(bc_dataset)} samples "
        f"({cfg.algo.trajs_per_obj} trajs/obj) for "
        f"{len(bc_dataset.obj_names_in_loaded_order)} objects",
        flush=True,
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
        assert key == "eval"
        log_env = eval_envs
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device).float().mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device).float().mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device).float().mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        log_env.reset_queues()

    # -------------------------
    # Stage 1: Only Mapping
    # -------------------------
    for epoch in range(cfg.algo.stage1_epochs):
        global_epoch = logger_start_log_step + epoch
        logger.print(f"[Stage 1] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.epoch = epoch
        agent.train()
        hash_voxel.train()

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss = agent(obs, subtask_labels, step_nums)
            cos_loss = cfg.algo.stage1_cos_loss_weight * cos_loss
            loss = cos_loss  # Stage 1 uses only cos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            n_samples += act.size(0)
            global_step += 1

            # Write to TensorBoard
            writer.add_scalar("Loss/Iteration", loss.item(), global_step)
            writer.add_scalar("Cosine Loss/Iteration", cos_loss.item(), global_step)

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
            eval_obs, _ = eval_envs.reset()
            eval_subtask_labels = get_object_labels_batch(uid_to_label_map, eval_envs.unwrapped.task_plan[0].composite_subtask_uids).to(device)
            B = eval_subtask_labels.size()

            for t in range(eval_envs.max_episode_steps):
                with torch.no_grad():
                    time_step = torch.tensor([t], dtype=torch.int32).repeat(B)
                    action, _ = agent(eval_obs, eval_subtask_labels, time_step)
                eval_obs, _, _, _, _ = eval_envs.step(action)

            if len(eval_envs.return_queue) > 0:
                store_env_stats("eval")
            logger.log(global_epoch)
            timer.end(key="eval")

        # Saving
        if check_freq(cfg.algo.save_freq, epoch):
            save_checkpoint(name="latest")
            timer.end(key="checkpoint")

    save_checkpoint(name="stage1-final")

    # ----------------------------
    # Stage 2: Mapping + Policy Learning
    # ----------------------------
    for epoch in range(cfg.algo.stage2_epochs):
        global_epoch = logger_start_log_step + cfg.algo.stage1_epochs + epoch
        logger.print(f"[Stage 2] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.epoch = epoch
        agent.train()
        hash_voxel.train()

        if cfg.algo.stage2_linear_scheduling:
            linear_scale = 1.0 - epoch / cfg.algo.stage2_epochs
            cos_loss_weight =  cfg.algo.stage2_cos_loss_weight * linear_scale
        else:
            cos_loss_weight =  cfg.algo.stage2_cos_loss_weight

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage2-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss = agent(obs, subtask_labels, step_nums)
            cos_loss = cos_loss_weight * cos_loss
            bc_loss = F.mse_loss(pi, act)
            loss = cos_loss + bc_loss  # Stage 2 uses both

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            n_samples += act.size(0)
            global_step += 1

            writer.add_scalar("Loss/Iteration", loss.item(), global_step)
            writer.add_scalar("Cosine Loss/Iteration", cos_loss.item(), global_step)
            writer.add_scalar("Behavior Cloning Loss/Iteration", bc_loss.item(), global_step)

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
            eval_obs, _ = eval_envs.reset()
            eval_subtask_labels = get_object_labels_batch(uid_to_label_map, eval_envs.unwrapped.task_plan[0].composite_subtask_uids).to(device)
            B = eval_subtask_labels.size()

            for t in range(eval_envs.max_episode_steps):
                with torch.no_grad():
                    time_step = torch.tensor([t], dtype=torch.int32).repeat(B)
                    action, _ = agent(eval_obs, eval_subtask_labels, time_step)
                eval_obs, _, _, _, _ = eval_envs.step(action)

            if len(eval_envs.return_queue) > 0:
                store_env_stats("eval")
            logger.log(global_epoch)
            timer.end(key="eval")

        # Saving
        if check_freq(cfg.algo.save_freq, epoch):
            save_checkpoint(name="latest")
            timer.end(key="checkpoint")

    # Final save
    save_checkpoint(name="stage2-final")

    # ------------------------------------------------
    # Stage 3: Freeze mapping + Policy only (BC loss)
    # ------------------------------------------------
    # 1) Freeze mapping modules
    for param in hash_voxel.parameters():
        param.requires_grad = False
    for param in implicit_decoder.parameters():
        param.requires_grad = False

    for epoch in range(cfg.algo.stage3_epochs):
        global_epoch = (
            logger_start_log_step
            + cfg.algo.stage1_epochs
            + cfg.algo.stage2_epochs
            + epoch
        )
        logger.print(f"[Stage 3] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.train()
        # hash_voxel & implicit_decoder remain in eval/frozen
        hash_voxel.eval()
        implicit_decoder.eval()

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage3-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss = agent(obs, subtask_labels, step_nums)
            # We ignore cos_loss now (mapping is frozen)
            bc_loss = F.mse_loss(pi, act)
            loss = bc_loss  # Stage 3 uses only BC

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            n_samples += act.size(0)
            global_step += 1

            writer.add_scalar("Loss/Iteration", loss.item(), global_step)
            writer.add_scalar("Behavior Cloning Loss/Iteration", bc_loss.item(), global_step)

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
            # Mapping modules remain frozen
            hash_voxel.eval()
            implicit_decoder.eval()

            eval_obs, _ = eval_envs.reset()
            eval_subtask_labels = get_object_labels_batch(uid_to_label_map, eval_envs.unwrapped.task_plan[0].composite_subtask_uids).to(device)
            B = eval_subtask_labels.size()

            for t in range(eval_envs.max_episode_steps):
                with torch.no_grad():
                    time_step = torch.tensor([t], dtype=torch.int32).repeat(B)
                    action, _ = agent(eval_obs, eval_subtask_labels, time_step)
                # Stub environment step
                eval_obs, _, _, _, _ = eval_envs.step(action)
            if len(eval_envs.return_queue) > 0:
                store_env_stats("eval")
            logger.log(global_epoch)
            timer.end(key="eval")

        # Saving
        if check_freq(cfg.algo.save_freq, epoch):
            save_checkpoint(name="latest")
            timer.end(key="checkpoint")

    # Final save
    save_checkpoint(name="stage3-final")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()
    writer.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
