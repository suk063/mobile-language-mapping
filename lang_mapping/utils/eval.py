import torch
import numpy as np
from typing import Dict

from mshab.utils.array import to_tensor
from mani_skill.utils import common

from mshab.utils.logger import Logger
from lang_mapping.utils.dataset import get_object_labels_batch

def _collect_stats(envs, device):
    stats = dict(
        return_per_step=(
            common.to_tensor(envs.return_queue, device=device).float().mean().item()
            / envs.max_episode_steps
        ),
        success_once=common.to_tensor(
            envs.success_once_queue, device=device
        )
        .float()
        .mean()
        .item(),
    )
    envs.reset_queues()
    return stats


def _pretty_print_stats(tag: str, stats: dict, logger: Logger, color: str):
    logger.print(
        f"{tag:<14}│ Return: {stats['return_per_step']:.2f} │ "
        f"Success_once: {stats['success_once']:.2f}",
        color=color,
        bold=True,
    )


def _flatten_obs(
    obs_raw: Dict[str, np.ndarray | torch.Tensor], device
) -> Dict[str, torch.Tensor]:
    flat = {"state": to_tensor(obs_raw["state"], device=device)}

    px = obs_raw["pixels"]
    for k in (
        "fetch_hand_rgb",
        "fetch_head_rgb",
        "fetch_hand_depth",
        "fetch_head_depth",
        "fetch_hand_pose",
        "fetch_head_pose",
    ):
        flat[k] = to_tensor(px[k], device=device)

    return flat


def run_eval_episode(eval_envs, eval_obs, agent, uid_to_label_map, uid2episode_id, device):
    """Runs one episode of evaluation."""
    max_steps = eval_envs.max_episode_steps

    # Get subtask info (labels and indices) for the episode
    plan0 = eval_envs.unwrapped.task_plan[0]
    subtask_labels = get_object_labels_batch(
        uid_to_label_map, plan0.composite_subtask_uids
    ).to(device)
    epi_ids = torch.tensor(
        [uid2episode_id[uid] for uid in plan0.composite_subtask_uids],
        device=device,
        dtype=torch.long,
    )

    for _ in range(max_steps):
        agent_obs = _flatten_obs(eval_obs, device)

        with torch.no_grad():
            action = agent(agent_obs, subtask_labels, epi_ids)

        # Environment step
        eval_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])

    return _collect_stats(eval_envs, device)
