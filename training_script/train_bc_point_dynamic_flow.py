import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

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

from lang_mapping.agent.agent_dynamic_flow import Agent_point_dynamic_flow
from lang_mapping.module import ImplicitDecoder
from lang_mapping.mapper.mapper_dynamic_flow import VoxelHashTableDynamicFlow
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
    max_cache_size: Union[int, Literal["all"]] = 0        # max data points to cache
    trajs_per_obj: Union[str, int] = "all"
    torch_deterministic: bool = True

    # Voxel/Scene Settings
    voxel_feature_dim: int = 120
    resolution: float = 0.12
    hash_table_size: int = 2**21
    scene_bound_min: List[float] = field(default_factory=lambda: [-2.6, -8.1, 0.0])
    scene_bound_max: List[float] = field(default_factory=lambda: [4.6, 4.7, 3.1])
    mod_time: int = 201
    trilinear_feat: bool = True
    trilinear_flow: bool = True

    # CLIP / Agent Settings
    clip_input_dim: int = 768
    open_clip_model_name: str = "EVA02-L-14"
    open_clip_model_pretrained: str = "merged2b_s4b_b131k"
    text_input: List[str] = field(default_factory=lambda: ["bowl", "apple"])
    camera_intrinsics: List[float] = field(default_factory=lambda: [71.9144, 71.9144, 112, 112])
    state_mlp_dim: int = 1024
    cos_loss_weight: float = 0.01
    flow_cos_loss_weight: float = 0.01
    scene_flow_loss_weight: float = 0.1

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


class DPDataset(ClosableDataset):
    def __init__(
        self,
        data_path,
        obs_horizon,
        pred_horizon,
        control_mode,
        trajs_per_obj="all",
        max_image_cache_size=0,
        truncate_trajectories_at_success=True,
    ):
        data_path = Path(data_path)
        if data_path.is_dir():
            h5_fps = [
                data_path / fp for fp in os.listdir(data_path) if fp.endswith(".h5")
            ]
        else:
            h5_fps = [data_path]

        trajectories = dict(actions=[], observations=[], subtask_uids=[])
        num_cached = 0
        self.h5_files: List[h5py.File] = []
        for fp_num, fp in enumerate(h5_fps):
            json_fp = fp.with_suffix(".json")
            with open(json_fp, "rb") as json_f:
                json_file = json.load(json_f)

            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                ep_num = int(k.replace("traj_", ""))
                subtask_uid = json_file["episodes"][ep_num]["subtask_uid"]

                obs, act = f[k]["obs"], f[k]["actions"][:]

                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    success_cutoff = min(success.index(True) + 1, len(success))
                    del success
                else:
                    success_cutoff = len(act)

                # NOTE (arth): we always cache state obs and actions because they take up very little memory.
                #       mostly constraints are on images, since those take up much more memory
                state_obs_list = [
                    *recursive_h5py_to_numpy(
                        obs["agent"], slice=slice(success_cutoff + 1)
                    ).values(),
                    *recursive_h5py_to_numpy(
                        obs["extra"], slice=slice(success_cutoff + 1)
                    ).values(),
                ]
                state_obs_list = [
                    x[:, None] if len(x.shape) == 1 else x for x in state_obs_list
                ]
                state_obs = torch.from_numpy(np.concatenate(state_obs_list, axis=1))
                # don't cut off actions in case we are able to use in place of padding
                act = torch.from_numpy(act)

                pixel_obs = dict(
                    fetch_head_rgb=obs["sensor_data"]["fetch_head"]["rgb"],
                    fetch_head_depth=obs["sensor_data"]["fetch_head"]["depth"],
                    fetch_hand_rgb=obs["sensor_data"]["fetch_hand"]["rgb"],
                    fetch_hand_depth=obs["sensor_data"]["fetch_hand"]["depth"],
                )
                if (
                    max_image_cache_size == "all"
                    or len(act) <= max_image_cache_size - num_cached
                ):
                    pixel_obs = to_tensor(
                        recursive_h5py_to_numpy(
                            pixel_obs, slice=slice(success_cutoff + 1)
                        )
                    )
                    num_cached += len(act)
                    print(num_cached)
                    import psutil
                    process = psutil.Process(os.getpid())
                    print(
                        f"cpu_mem_use_GB={process.memory_info().rss / (10**9)}"
                    )
                else:
                    num_uncached_this_file += len(act)

                # add cam extrinsics
                cam_pose_obs = to_tensor(
                    recursive_h5py_to_numpy(
                        dict(
                            fetch_head_pose=obs["sensor_param"]["fetch_head"][
                                "extrinsic_cv"
                            ],
                            fetch_hand_pose=obs["sensor_param"]["fetch_hand"][
                                "extrinsic_cv"
                            ],
                        ),
                        slice=slice(success_cutoff + 1),
                    ),
                    dtype=torch.float,
                )
                pixel_obs.update(**cam_pose_obs)

                trajectories["actions"].append(act)
                trajectories["observations"].append(dict(state=state_obs, **pixel_obs))
                trajectories["subtask_uids"].append(subtask_uid)

            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            # NOTE (arth): since we cut off data at first success, we might have extra actions available
            #   after the end of slice which we can use instead of hand-made padded zero actions
            L = trajectories["observations"][traj_idx]["state"].shape[0] - 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            # NOTE (arth): add subtask_uid here for convenience
            self.slices += [
                (
                    trajectories["subtask_uids"][traj_idx],
                    traj_idx,
                    start,
                    start + pred_horizon,
                )
                # for start in range(-pad_before, L - pred_horizon + pad_after)
                # NOTE (arth): start at 0 since we use o_t and o_{t+1} for scene flow est
                for start in range(0, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        subtask_uid, traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if len(obs_seq[k].shape) == 4:
                obs_seq[k] = to_tensor(obs_seq[k])  # FS, D, H, W
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {"observations": obs_seq, "actions": act_seq, "subtask_uid": subtask_uid}

    def __len__(self):
        return len(self.slices)

    def close(self):
        for h5_file in self.h5_files:
            h5_file.close()


class TempTranslateToPointDataset(DPDataset):
    def __init__(self, *args, cat_state=True, cat_pixels=False, **kwargs):
        assert (
            cat_state
        ), "This is a low-effort temp wrapper which requires cat_state=True"
        assert (
            not cat_pixels
        ), "This is a low-effort temp wrapper which requires cat_pixels=False"
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        assert isinstance(index, int)
        item = super().__getitem__(index)

        # NOTE (arth): reformat DPDataset obs to work with current train code

        state_obs = item["observations"]["state"]
        assert state_obs.size(0) == 2
        state_obs = {"state": state_obs[0], "state_p1": state_obs[1]}

        pixel_obs = {
            "fetch_hand_depth": item["observations"]["fetch_hand_depth"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_depth_p1": item["observations"]["fetch_hand_depth"][1].squeeze(-1).unsqueeze(0),
            "fetch_hand_rgb": item["observations"]["fetch_hand_rgb"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_rgb_p1": item["observations"]["fetch_hand_rgb"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_depth": item["observations"]["fetch_head_depth"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_depth_p1": item["observations"]["fetch_head_depth"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_rgb": item["observations"]["fetch_head_rgb"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_rgb_p1": item["observations"]["fetch_head_rgb"][1].squeeze(-1).unsqueeze(0),
            "fetch_hand_pose": item["observations"]["fetch_hand_pose"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_pose_p1": item["observations"]["fetch_hand_pose"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_pose": item["observations"]["fetch_head_pose"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_pose_p1": item["observations"]["fetch_head_pose"][1].squeeze(-1).unsqueeze(0),
        }

        obs = {**state_obs, "pixels": pixel_obs}

        # NOTE (arth): we use start act and step_num since we use o_t and o_{t+1} for scene flow est
        act = item["actions"][0]
        step_num = self.slices[index][2]

        subtask_uid = item["subtask_uid"]

        return (obs, act, subtask_uid, step_num)


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
    hash_voxel = VoxelHashTableDynamicFlow(
        resolution=cfg.algo.resolution,
        hash_table_size=cfg.algo.hash_table_size,
        feature_dim=cfg.algo.voxel_feature_dim,
        scene_bound_min=tuple(cfg.algo.scene_bound_min),
        scene_bound_max=tuple(cfg.algo.scene_bound_max),
        mod_time= cfg.algo.mod_time,
        trilinear_feat = cfg.algo.trilinear_feat,
        trilinear_flow = cfg.algo.trilinear_flow,
        device=device
    ).to(device)

    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
        hidden_dim=256,
        output_dim=cfg.algo.clip_input_dim,
        L=10
    ).to(device)

    # Agent
    agent = Agent_point_dynamic_flow(
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
    # bc_dataset = BCDataset(
    #     cfg.algo.data_dir_fp,
    #     cfg.algo.max_cache_size,
    #     cat_state=cfg.eval_env.cat_state,
    #     cat_pixels=cfg.eval_env.cat_pixels,
    #     trajs_per_obj=cfg.algo.trajs_per_obj,
    # )
    # logger.print(
    #     f"BC Dataset: {len(bc_dataset)} samples "
    #     f"({cfg.algo.trajs_per_obj} trajs/obj) for "
    #     f"{len(bc_dataset.obj_names_in_loaded_order)} objects",
    #     flush=True,
    # )
    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"
    bc_dataset = TempTranslateToPointDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=2,
        pred_horizon=2,
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
        hash_voxel.compute_time_variance(chunk_size=1000)

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss, scene_flow_loss, scene_flow_cos_loss = agent.forward_train(obs, subtask_labels, step_nums)
            cos_loss = cfg.algo.cos_loss_weight * cos_loss
            scene_flow_loss = cfg.algo.scene_flow_loss_weight * scene_flow_loss
            scene_flow_cos_loss = cfg.algo.flow_cos_loss_weight * scene_flow_cos_loss
            loss = cos_loss + scene_flow_loss + scene_flow_cos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            n_samples += act.size(0)
            global_step += 1

            # Write to TensorBoard
            writer.add_scalar("Loss/Iteration", loss.item(), global_step)
            writer.add_scalar("Cosine Loss/Iteration", cos_loss.item(), global_step)
            writer.add_scalar("Scene Flow Loss/Iteration", scene_flow_loss.item(), global_step)
            writer.add_scalar("Scene Flow Cos Loss/Iteration", scene_flow_cos_loss.item(), global_step)

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
                    action = agent.forward_eval(eval_obs, eval_subtask_labels, time_step)
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
        hash_voxel.compute_time_variance(chunk_size=1000)

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage2-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss, scene_flow_loss, scene_flow_cos_loss = agent.forward_train(obs, subtask_labels, step_nums)
            cos_loss = cfg.algo.cos_loss_weight * cos_loss
            scene_flow_loss = cfg.algo.scene_flow_loss_weight * scene_flow_loss
            scene_flow_cos_loss = cfg.algo.flow_cos_loss_weight * scene_flow_cos_loss
            bc_loss = F.mse_loss(pi, act)
            loss = cos_loss  + + scene_flow_loss + scene_flow_cos_loss + bc_loss  # Stage 2 uses both

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            n_samples += act.size(0)
            global_step += 1

            writer.add_scalar("Loss/Iteration", loss.item(), global_step)
            writer.add_scalar("Cosine Loss/Iteration", cos_loss.item(), global_step)
            writer.add_scalar("Scene Flow Loss/Iteration", scene_flow_loss.item(), global_step)
            writer.add_scalar("Scene Flow Cos Loss/Iteration", scene_flow_cos_loss.item(), global_step)
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
                    action = agent.forward_eval(eval_obs, eval_subtask_labels, time_step)
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
        hash_voxel.compute_time_variance(chunk_size=1000)

        for obs, act, subtask_uids, step_nums in tqdm(bc_dataloader, desc="Stage3-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, cos_loss, scene_flow_loss, scene_flow_cos_loss = agent.forward_train(obs, subtask_labels, step_nums)
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
            writer.add_scalar("Cosine Loss/Iteration", cos_loss.item(), global_step)
            writer.add_scalar("Scene Flow Loss/Iteration", scene_flow_loss.item(), global_step)
            writer.add_scalar("Scene Flow Cos Loss/Iteration", scene_flow_cos_loss.item(), global_step)
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
                    action = agent.forward_eval(eval_obs, eval_subtask_labels, time_step)
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
