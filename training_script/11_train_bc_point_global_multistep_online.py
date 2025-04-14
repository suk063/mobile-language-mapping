import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
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

from lang_mapping.agent.agent_global_multistep_gridnet import Agent_global_multistep_gridnet
from lang_mapping.module import ImplicitDecoder
from lang_mapping.mapper.mapper import VoxelHashTable
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler

def merge_t_m1(obs_m1: Dict[str, np.ndarray], obs_t: Dict[str, np.ndarray]):
    agent_obs = {
        "state": obs_t["state"],      # t
        "state_m1": obs_m1["state"],    # t-1
        "pixels": {
            "fetch_hand_rgb": obs_t['pixels']["fetch_hand_rgb"],
            "fetch_hand_rgb_m1": obs_m1['pixels']["fetch_hand_rgb"],
            "fetch_hand_depth": obs_t['pixels']["fetch_hand_depth"],
            "fetch_hand_depth_m1": obs_m1['pixels']["fetch_hand_depth"],
            "fetch_hand_pose": obs_t['pixels']["fetch_hand_pose"],
            "fetch_hand_pose_m1": obs_m1['pixels']["fetch_hand_pose"],

            "fetch_head_rgb": obs_t['pixels']["fetch_head_rgb"],
            "fetch_head_rgb_m1": obs_m1['pixels']["fetch_head_rgb"],
            "fetch_head_depth": obs_t['pixels']["fetch_head_depth"],
            "fetch_head_depth_m1": obs_m1['pixels']["fetch_head_depth"],
            "fetch_head_pose": obs_t['pixels']["fetch_head_pose"],
            "fetch_head_pose_m1": obs_m1['pixels']["fetch_head_pose"],
        }
    }
    return agent_obs

def build_uid_mapping(task_plan_path: str):
    """
    Reads all composite_subtask_uids from the task_plan file located at task_plan_path,
    and returns a dictionary mapping each unique uid to a unique integer from 0 to N-1.
    """
    with open(task_plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_uids = []
    # Iterate over all plans
    for plan in data["plans"]:
        # Iterate over all subtasks within each plan
        for subtask in plan["subtasks"]:
            # Add each composite_subtask_uid from the current subtask to the list
            for c_uid in subtask["composite_subtask_uids"]:
                all_uids.append(c_uid)

    # Extract a sorted list of unique uids
    unique_uids = sorted(list(set(all_uids)))
    print(f"Total unique subtask_uids (N) = {len(unique_uids)}")

    # Create a mapping from uid to an integer index (0 to N-1)
    uid2index = {uid: idx for idx, uid in enumerate(unique_uids)}
    return uid2index

def map_uids_to_indices_tensor(uids, uid2index):
    """
    Given a list of arbitrary uid strings, returns a torch.Tensor containing 
    their corresponding integer indices as defined in uid2index mapping.
    """
    indices = []
    for uid in uids:
        if uid in uid2index:
            indices.append(uid2index[uid])
        else:
            # Assign -1 if the uid is not found in the mapping
            indices.append(-1)
    return torch.tensor(indices, dtype=torch.long)

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
    epochs: int = 1         # stage 1 epochs

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
    state_mlp_dim: int = 1024
    hidden_dim: int = 240
    cos_loss_weight: float = 0.1
    num_heads: int = 8
    num_layers_transformer: int = 4
    num_layers_perceiver: int = 2
    num_learnable_tokens: int = 16
    pe_level: int = 10
    action_horizon: int = 16

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
                    success_cutoff = min(success.index(True), len(success))
                    del success
                else:
                    # success_cutoff = len(act)
                    success_cutoff = 100

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
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "subtask_uid": subtask_uid,
            "traj_idx": traj_idx, 
        }


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
        state_obs = {"state_m1": state_obs[0], "state": state_obs[1]}
        
        pixel_obs = {
            "fetch_hand_depth_m1": item["observations"]["fetch_hand_depth"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_depth": item["observations"]["fetch_hand_depth"][1].squeeze(-1).unsqueeze(0),
            "fetch_hand_rgb_m1": item["observations"]["fetch_hand_rgb"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_rgb": item["observations"]["fetch_hand_rgb"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_depth_m1": item["observations"]["fetch_head_depth"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_depth": item["observations"]["fetch_head_depth"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_rgb_m1": item["observations"]["fetch_head_rgb"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_rgb": item["observations"]["fetch_head_rgb"][1].squeeze(-1).unsqueeze(0),
            "fetch_hand_pose_m1": item["observations"]["fetch_hand_pose"][0].squeeze(-1).unsqueeze(0),
            "fetch_hand_pose": item["observations"]["fetch_hand_pose"][1].squeeze(-1).unsqueeze(0),
            "fetch_head_pose_m1": item["observations"]["fetch_head_pose"][0].squeeze(-1).unsqueeze(0),
            "fetch_head_pose": item["observations"]["fetch_head_pose"][1].squeeze(-1).unsqueeze(0),
        }

        obs = {**state_obs, "pixels": pixel_obs}

        # NOTE (arth): we use start act and step_num since we use o_t and o_{t+1} for scene flow est
        
        act = item["actions"][1:] 

        subtask_uid = item["subtask_uid"]
        traj_idx = item["traj_idx"]

        return (obs, act, subtask_uid, traj_idx)

def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid_to_label_map = build_object_map(cfg.eval_env.task_plan_fp, cfg.algo.text_input)
    
    uid2index_map = build_uid_mapping(cfg.eval_env.task_plan_fp)
    
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

    grid_cfg = {
        "name": "grid_net",
        "spatial_dim": 3,
        "decoder": {
            "type": "mlp",
            "hidden_dim": 240,
            "hidden_layers": 3,
            "out_dim": 768,
            "pos_invariant": True,
            "fix": False,
            "pretrained_model": None
        },
        "grid": {
            "type": "regular",
            "feature_dim": 120,
            "init_stddev": 0.2,
            "bound": [
                [-2.6, 4.6],
                [-8.1, 4.7],
                [0.0, 3.1]
            ],
            "base_cell_size": 0.5,
            "per_level_scale": 5.0,
            "n_levels": 2,
            "second_order_grid_sample": False
        }
    }

    # VoxelHashTable and ImplicitDecoder
    static_map = VoxelHashTable(
        resolution=cfg.algo.resolution,
        hash_table_size=cfg.algo.hash_table_size,
        feature_dim=cfg.algo.voxel_feature_dim,
        scene_bound_min=tuple(cfg.algo.scene_bound_min),
        scene_bound_max=tuple(cfg.algo.scene_bound_max),
        device=device
    ).to(device)
    
    implicit_decoder = ImplicitDecoder(
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
        hidden_dim=cfg.algo.hidden_dim,
        output_dim=cfg.algo.clip_input_dim,
        L=cfg.algo.pe_level
    ).to(device)

    # from lang_mapping.grid_net import GridNet
    # static_map = GridNet(cfg=grid_cfg, device=device)
    
    # Agent
    agent = Agent_global_multistep_gridnet(
        sample_obs=eval_obs,
        single_act_shape=eval_envs.unwrapped.single_action_space.shape,
        device=device,
        voxel_feature_dim=cfg.algo.voxel_feature_dim,
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
        num_learnable_tokens=cfg.algo.num_learnable_tokens
    ).to(device)
    
    if cfg.algo.pretrained_agent_path is not None and os.path.exists(cfg.algo.pretrained_agent_path):
        print(f"[INFO] Loading pretrained agent from {cfg.algo.pretrained_agent_path}")
        agent.load_state_dict(torch.load(cfg.algo.pretrained_agent_path, map_location=device), strict=False)

    if cfg.algo.pretrained_voxel_path is not None and os.path.exists(cfg.algo.pretrained_voxel_path):
        print(f"[INFO] Loading pretrained voxel from {cfg.algo.pretrained_voxel_path}")
        static_map.load_state_dict(torch.load(cfg.algo.pretrained_voxel_path, map_location=device), strict=False)

    if cfg.algo.pretrained_implicit_path is not None and os.path.exists(cfg.algo.pretrained_implicit_path):
        print(f"[INFO] Loading pretrained implicit decoder from {cfg.algo.pretrained_implicit_path}")
        implicit_decoder.load_state_dict(torch.load(cfg.algo.pretrained_implicit_path, map_location=device), strict=True)

    # voxel_checkpoint = torch.load('pre-trained/hash_voxel.pt', map_location=device)
    # decoder_checkpoint = torch.load('pre-trained/implicit_decoder.pt', map_location=device)
    # static_map.load_state_dict(voxel_checkpoint['model'], strict=False)
    # implicit_decoder.load_state_dict(decoder_checkpoint['model'], strict=True)
    # def verify_load(tensor1, tensor2, tensor_name):
    #     if torch.allclose(tensor1, tensor2):
    #         print(f"{tensor_name} successfully loaded")
    #     else:
    #         print(f"{tensor_name} loading failed")
    # for name, param in implicit_decoder.named_parameters():
    #     loaded_param = decoder_checkpoint['model'][name]
    #     verify_load(param.data, loaded_param, f"implicit_decoder.{name}")
    
    buffers = torch.load('voxel_buffers.pt')
    
    static_map.used_mask = buffers['used_mask'].to(static_map.device)
    static_map.valid_grid_coords = buffers['valid_grid_coords'].to(static_map.device)
    
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

    # ------------------------------------------------
    # Mapping
    # ------------------------------------------------
    for name, param in agent.named_parameters():
        param.requires_grad = True 
    
    for name, param in agent.clip_model.named_parameters():
        param.requires_grad = False 
        
    # for name, param in agent.static_map.named_parameters():
    #     param.requires_grad = False 
        
    # for name, param in agent.implicit_decoder.named_parameters():
    #     param.requires_grad = False 
    
    
    alpha = 0.3
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

        # Batch size (number of parallel evaluation environments)
        B = subtask_labels.size(0)

        for t in range(max_steps):
            # Merge observations from time t and t-1
            agent_obs = merge_t_m1(prev_obs, eval_obs)

            with torch.no_grad():
                action, _ = agent(agent_obs, subtask_labels)

            # Environment step
            next_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])
            prev_obs = eval_obs
            eval_obs = next_obs

        return eval_obs

    # def run_eval_episode(eval_envs, eval_obs, agent, uid_map):
    #     max_steps = eval_envs.max_episode_steps

    #     for t in range(max_steps):
    #         plan0 = eval_envs.unwrapped.task_plan[0]
    #         subtask_labels = get_object_labels_batch(uid_map, plan0.composite_subtask_uids).to(device)

    #         with torch.no_grad():
    #             time_step = torch.tensor([t], dtype=torch.int32, device=device).repeat(eval_envs.num_envs)
    #             action, _ = agent(eval_obs, subtask_labels)
    #         eval_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])

    # def run_eval_episode(eval_envs, eval_obs, agent, uid_map):
    #     max_steps = eval_envs.max_episode_steps
    #     plan0 = eval_envs.unwrapped.task_plan[0]
    #     subtask_labels = get_object_labels_batch(uid_map, plan0.composite_subtask_uids).to(device)
    #     agent.eval()
    
    #     while len(eval_envs.return_queue) < cfg.algo.num_eval_envs:
    #         action, _  = agent(eval_obs, subtask_labels)
    #         action = action.detach()
                      
    #         for i in range(action.shape[1]):
    #             eval_obs, rew, terminated, truncated, info = eval_envs.step(
    #                 action[:, i, :]
    #             )
    #             if truncated.any():
    #                 break
    #         if truncated.any():
    #             assert (
    #                 truncated.all() == truncated.any()
    #             ), "all episodes should truncate at the same time for fair evaluation with other algorithms"

    params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.algo.lr)
    
    agent.to(device)

    for epoch in range(cfg.algo.epochs):
        global_epoch = logger_start_log_step + epoch
        
        logger.print(f"[Stage 1] Epoch: {global_epoch}")
        tot_loss, n_samples = 0, 0
        agent.train()

        for obs, act, subtask_uids, traj_idx in tqdm(bc_dataloader, desc="Stage1-Batch", unit="batch"):
            subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(device)
            
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(act, device=device, dtype="float")

            pi, total_cos_loss = agent(obs, subtask_labels)
            cos_loss = total_cos_loss * cfg.algo.cos_loss_weight
            bc_loss = F.smooth_l1_loss(pi, act, reduction='none')
            # bc_loss = F.l1_loss(pi, act)
            
            weighted_bc_loss = bc_loss * time_weights
            bc_loss = 10 * weighted_bc_loss.mean()
        
            loss = bc_loss + cos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += bc_loss.item()
            n_samples += act.size(0)
            global_step += 1

            writer.add_scalar("BC Loss/Iteration", bc_loss.item(), global_step)
            writer.add_scalar("cos Loss/Iteration", cos_loss.item(), global_step)

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
            
    # Now evaluate all task plans in chunks
    batch_size = eval_envs.num_envs
    all_plan_count = cfg.eval_env.all_plan_count
    all_plan_idxs_list = list(range(all_plan_count))

    print("Evaluating all tasks in chunks...")
    agent.eval()
    pbar = tqdm(total=all_plan_count, desc="Evaluating all tasks (last epoch)")

    chunk_start = 0
    while chunk_start < all_plan_count:
        chunk_end = min(chunk_start + batch_size, all_plan_count)
        chunk_size = chunk_end - chunk_start
        chunk = all_plan_idxs_list[chunk_start:chunk_end]

        # If chunk is smaller than batch_size, pad it with the last element
        if chunk_size < batch_size:
            chunk += [chunk[-1]] * (batch_size - chunk_size)

        plan_idxs_tensor = torch.tensor(chunk, dtype=torch.int)

        eval_obs, info = eval_envs.reset(options={"task_plan_idxs": plan_idxs_tensor})
        run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map)

        chunk_start += batch_size
        pbar.update(chunk_size)

    if len(eval_envs.return_queue) > 0:
        store_env_stats("eval_all")

    pbar.close()
    logger.log(global_epoch)
    timer.end(key="eval")

    # Final save
    save_checkpoint(name="stage-final")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()
    writer.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)