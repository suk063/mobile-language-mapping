import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import h5py
from tqdm import tqdm
import random
import numpy as np
import torch
import copy
from collections import defaultdict

from mshab.utils.array import to_tensor
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset


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
        single_traj_idx: Optional[int] = None,
        allowed_uids: Optional[set[str]] = None,
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
            
            # Note (sh): sample based on trajectory idx
            if single_traj_idx is not None:
                keys = []
                for t_idx in single_traj_idx:
                    key_str = f"traj_{t_idx}"
                    if key_str in f:
                        keys.append(key_str)
            else:
                if trajs_per_obj == "all":
                    keys = list(f.keys())
                else:
                    keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                ep_num = int(k.replace("traj_", ""))
                subtask_uid = json_file["episodes"][ep_num]["subtask_uid"]
                
                if allowed_uids is not None and subtask_uid not in allowed_uids:
                    continue
                
                #f[k]['obs']['extra'].keys()
                # <KeysViewHDF5 ['tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped']>

                obs, act = f[k]["obs"], f[k]["actions"][:]

                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    success_cutoff = min(success.index(True)+10, len(success))
                    del success
                else:
                    success_cutoff = len(act)
                    # success_cutoff = 100

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
                
                is_grasped_obs = to_tensor(
                    recursive_h5py_to_numpy(
                        dict(
                            fetch_is_grasped=obs["extra"]["is_grasped"]
                        ),
                        slice=slice(success_cutoff + 1),
                    ),
                    dtype=torch.bool,
                )
                
                pixel_obs.update(**is_grasped_obs)

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

                for start in range(-pad_before, L-pred_horizon+pad_after)
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

        act = item["actions"][1:]

        subtask_uid = item["subtask_uid"]
        traj_idx = item["traj_idx"]
        
        is_grasped = item["observations"]["fetch_is_grasped"][1]

        return (obs, act, subtask_uid, traj_idx, is_grasped)

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