import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file
from mshab.utils.array import to_tensor
from mani_skill.utils import common

# BC agents
from lang_mapping.agent.agent_map_bc import Agent_map_bc
from lang_mapping.agent.agent_uplifted_bc import Agent_uplifted_bc
from lang_mapping.agent.agent_image_bc import Agent_image_bc
from lang_mapping.agent.agent_point_bc import Agent_point_bc

# Utils for labels, scenes, and eval rollouts
from lang_mapping.utils.dataset import (
    build_object_map,
    build_uid_episode_scene_maps,
    get_object_labels_batch,
)

# Text prompts per task (used by CLIP encoders inside agents)
TEXT_PROMPTS: Dict[str, List[str]] = {
    "set_table": ["bowl", "apple"],
    "tidy_house": [
        "potted_meat_can",
        "master_chef_can",
        "bowl",
        "tomato_soup_can",
        "cracker_box",
        "sugar_box",
        "gelatin_box",
        "pudding_box",
        "tuna_fish_can",
    ],
    "prepare_groceries": [
        "tomato_soup_can",
        "cracker_box",
        "tuna_fish_can",
        "bowl",
        "sugar_box",
        "master_chef_can",
        "pudding_box",
        "potted_meat_can",
        "gelatin_box",
    ],
}


def pad_ids(ids: List[int], num_envs: int) -> List[int]:
    if len(ids) == 0:
        return []
    return ids + [ids[-1]] * (num_envs - len(ids))


def collect_env_stats(envs, device):
    """
    Collects environment statistics from finished episodes.

    NOTE: This implementation differs from the one in `lang_mapping.utils.eval`.
    This version handles tensor conversions and is tailored for the eval script's
    specific environment setup, which may not use pre-converted numpy queues.
    """
    returns = common.to_tensor(envs.return_queue, device=device).float()
    successes_once = common.to_tensor(envs.success_once_queue, device=device).float()
    successes_at_end = common.to_tensor(envs.success_at_end_queue, device=device).float()
    lengths = common.to_tensor(envs.length_queue, device=device).float()

    records = []
    num_episodes = returns.numel()
    if num_episodes == 0:
        return []

    for i in range(num_episodes):
        records.append(
            dict(
                rps=returns[i].item() / envs.max_episode_steps,
                succ_once=successes_once[i].item(),
                succ_end=successes_at_end[i].item(),
                length=lengths[i].item(),
            )
        )
    envs.reset_queues()
    return records


def _flatten_obs(
    obs_raw: Dict[str, np.ndarray | torch.Tensor], device
) -> Dict[str, torch.Tensor]:
    """Flattens nested observations from ManiSkill."""
    flat = {"state": to_tensor(obs_raw["state"], device=device)}
    
    # When FrameStack wrapper is used, pixel observations are nested
    px = obs_raw.get("pixels", {})

    for k in (
        "fetch_hand_rgb",
        "fetch_head_rgb",
        "fetch_hand_depth",
        "fetch_head_depth",
        "fetch_hand_pose",
        "fetch_head_pose",
    ):
        if k in px:
            flat[k] = to_tensor(px[k], device=device)
        elif k in obs_raw: # Fallback for non-nested structure
             flat[k] = to_tensor(obs_raw[k], device=device)

    return flat


def build_agent(
    kind: str,
    sample_obs,
    single_act_shape,
    device: torch.device,
    text_input: List[str],
    clip_input_dim: int,
    camera_intrinsics: List[float],
    num_heads: int,
    num_layers_transformer: int,
    num_action_layer: int,
    action_pred_horizon: int,
    transf_input_dim: int,
    static_map_path: Path | None = None,
    implicit_decoder_path: Path | None = None,
):
    if kind == "map":
        assert static_map_path is not None and implicit_decoder_path is not None, (
            "For 'map' agent, provide --static-map-path and --implicit-decoder-path"
        )

        # Lazy imports to avoid overhead if not used
        from lang_mapping.mapper.mapper import MultiVoxelHashTable
        from lang_mapping.module import ImplicitDecoder

        static_maps = MultiVoxelHashTable.load_sparse(static_map_path).to(device)
        for p in static_maps.parameters():
            p.requires_grad = False

        # These hyperparameters mirror training defaults
        voxel_feature_dim = 128
        hidden_dim = 240
        implicit_decoder = ImplicitDecoder(
            voxel_feature_dim=voxel_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=clip_input_dim,
        ).to(device)
        state = torch.load(implicit_decoder_path, map_location=device)
        implicit_decoder.load_state_dict(state["model"], strict=True)
        for p in implicit_decoder.parameters():
            p.requires_grad = False

        agent = Agent_map_bc(
            sample_obs=sample_obs,
            single_act_shape=single_act_shape,
            transf_input_dim=transf_input_dim,
            clip_input_dim=clip_input_dim,
            text_input=text_input,
            camera_intrinsics=tuple(camera_intrinsics),
            static_maps=static_maps,
            implicit_decoder=implicit_decoder,
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            num_action_layer=num_action_layer,
            action_pred_horizon=action_pred_horizon,
        ).to(device)
        return agent

    if kind == "uplifted":
        return Agent_uplifted_bc(
            sample_obs=sample_obs,
            single_act_shape=single_act_shape,
            transf_input_dim=transf_input_dim,
            clip_input_dim=clip_input_dim,
            text_input=text_input,
            camera_intrinsics=tuple(camera_intrinsics),
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            num_action_layer=num_action_layer,
            action_pred_horizon=action_pred_horizon,
        ).to(device)

    if kind == "image":
        return Agent_image_bc(
            sample_obs=sample_obs,
            single_act_shape=single_act_shape,
            transf_input_dim=transf_input_dim,
            clip_input_dim=clip_input_dim,
            text_input=text_input,
            camera_intrinsics=tuple(camera_intrinsics),
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            num_action_layer=num_action_layer,
            action_pred_horizon=action_pred_horizon,
        ).to(device)

    if kind == "point":
        return Agent_point_bc(
            sample_obs=sample_obs,
            single_act_shape=single_act_shape,
            transf_input_dim=transf_input_dim,
            clip_input_dim=clip_input_dim,
            text_input=text_input,
            camera_intrinsics=tuple(camera_intrinsics),
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            num_action_layer=num_action_layer,
            action_pred_horizon=action_pred_horizon,
        ).to(device)

    raise ValueError(f"Unknown agent kind: {kind}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained BC agents over all task plans"
    )
    parser.add_argument("--agent", required=True, choices=["map", "image", "uplifted", "point"])
    parser.add_argument("--task", required=True, choices=list(TEXT_PROMPTS.keys()))
    parser.add_argument("--subtask", required=True)
    parser.add_argument("--plan-file", required=True, help="Plan file name or absolute path (e.g., all_10.json)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])  # dataset split
    parser.add_argument("--seed", type=int, help="seed in training (single)")
    parser.add_argument("--seeds", type=int, nargs="+", help="evaluate multiple seeds (space-separated)")
    parser.add_argument(
        "--root",
        default=str(Path("~/.maniskill/data/scene_datasets/replica_cad_dataset").expanduser()),
        help="Dataset root containing rearrange/ and rearrange/spawn_data",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scene-ids-yaml", default="pretrained/scene_ids.yaml")
    parser.add_argument("--ckpt-name", default="best_eval_success_once_ckpt.pt")
    # Architecture defaults (must match training)
    parser.add_argument("--clip-input-dim", type=int, default=768)
    parser.add_argument("--transf-input-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers-transformer", type=int, default=4)
    parser.add_argument("--num-action-layer", type=int, default=6)
    parser.add_argument("--action-pred-horizon", type=int, default=16)
    parser.add_argument(
        "--camera-intrinsics",
        nargs=4,
        type=float,
        default=[71.9144, 71.9144, 112.0, 112.0],
        metavar=("fx", "fy", "cx", "cy"),
        help="Camera intrinsics (fx, fy, cx, cy)",
    )

    # Map-specific resources
    parser.add_argument("--static-map-path", type=str, default=None)
    parser.add_argument("--implicit-decoder-path", type=str, default=None)

    # Allow overriding model directory
    parser.add_argument("--model-dir", type=str, default=None, help="Path to the model directory, overrides default path construction")

    args = parser.parse_args()

    device = torch.device(args.device)

    root = Path(args.root)

    # Resolve plan file path (supports absolute path or file name)
    plan_file_arg = Path(args.plan_file)
    if plan_file_arg.is_absolute():
        plan_fp = plan_file_arg
    else:
        plan_fp = (
            root
            / "rearrange"
            / "task_plans"
            / args.task
            / args.subtask
            / args.split
            / args.plan_file
        )
    assert plan_fp.exists(), f"Plan file not found: {plan_fp}"

    # Resolve spawn data path based on task/subtask/split
    spawn_fp = root / "rearrange" / "spawn_data" / args.task / args.subtask / args.split / "spawn_data.pt"
    assert spawn_fp.exists(), f"Spawn data not found: {spawn_fp}"

    # Plans metadata
    plan_data = plan_data_from_file(plan_fp)
    all_plan_count = len(plan_data.plans)
    # all_plan_count = 50
    # Create eval envs
    env_cfg = EnvConfig(
        env_id="PickSubtaskTrain-v0",
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        task_plan_fp=str(plan_fp),
        spawn_data_fp=str(spawn_fp),
        record_video=False,
        cat_state=True,
        cat_pixels=False,
        frame_stack=1,
    )
    env = make_env(env_cfg, video_path=None)

    # Build label/scene maps
    text_prompts = TEXT_PROMPTS[args.task]
    uid2lbl = build_object_map(str(plan_fp), text_prompts)
    _, uid2scene_id = build_uid_episode_scene_maps(str(plan_fp), args.scene_ids_yaml)

    # Sample obs for initialising agent modules
    sample_obs, _ = env.reset(seed=0)

    # Build agent (will reuse instance across seeds, only reloading weights)
    agent = build_agent(
        kind=args.agent,
        sample_obs=sample_obs,
        single_act_shape=env.single_action_space.shape,
        device=device,
        text_input=text_prompts,
        clip_input_dim=args.clip_input_dim,
        camera_intrinsics=args.camera_intrinsics,
        num_heads=args.num_heads,
        num_layers_transformer=args.num_layers_transformer,
        num_action_layer=args.num_action_layer,
        action_pred_horizon=args.action_pred_horizon,
        transf_input_dim=args.transf_input_dim,
        static_map_path=Path(args.static_map_path) if args.static_map_path else None,
        implicit_decoder_path=Path(args.implicit_decoder_path) if args.implicit_decoder_path else None,
    )

    # Resolve seeds list
    seeds: list[int]
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        raise AssertionError("Provide --seed or --seeds")

    all_results = []

    for seed in seeds:
        # Load weights for this seed
        if args.model_dir:
            base_dir = Path(args.model_dir)
        else:
            base_dir = Path(
                f"mshab_exps/PickSubtaskTrain-v0/{args.task}-{args.subtask}/{args.task}-{args.subtask}-{args.agent}-{seed}/models"
            )
        ckpt_path = base_dir / args.ckpt_name
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt["agent"] if isinstance(ckpt, dict) and "agent" in ckpt else ckpt
        agent.load_state_dict(state_dict)
        agent.eval()

        # Evaluate all plan ids
        all_ids = list(range(all_plan_count))
        
        # Store results per batch to calculate weighted average later
        batch_results = []

        for start in tqdm(range(0, len(all_ids), args.num_envs), desc=f"Evaluating seed {seed}"):
            chunk = all_ids[start : start + args.num_envs]
            padded_ids = pad_ids(chunk, args.num_envs)
            obs, _ = env.reset(options={
                "task_plan_idxs": torch.tensor(padded_ids, dtype=torch.long, device=device)
            })

            # Run a full episode rollout for the current batch
            # NOTE: This logic is inlined from the original `run_eval_episode` to prevent
            # it from consuming stats queues before the main eval loop can.
            max_steps = env.max_episode_steps

            # This is a likely bug in the original code, but we preserve it to fix the crash.
            plan0 = env.unwrapped.task_plan[0]
            subtask_labels = get_object_labels_batch(
                uid2lbl, plan0.composite_subtask_uids
            ).to(device)
            scene_ids = torch.tensor(
                [uid2scene_id[uid] for uid in plan0.composite_subtask_uids],
                device=device,
                dtype=torch.long,
            )
            
            # Run simulation for one full episode
            for _ in range(max_steps):
                agent_obs = _flatten_obs(obs, device)
                with torch.no_grad():
                    action = agent(agent_obs, subtask_labels, scene_ids)
                obs, _, _, _, _ = env.step(action[:, 0, :])

            # Collect per-episode stats for the unpadded subset
            batch_recs = collect_env_stats(env, device)[: len(chunk)]
            if batch_recs:
                batch_results.append((len(chunk), batch_recs))

        # Aggregate results for this seed
        total_n_episodes = sum(n for n, _ in batch_results)
        if total_n_episodes == 0:
            print(f"No episodes collected for seed {seed}. Skipping.")
            continue
        
        # Calculate weighted average for each metric
        weighted_sums = defaultdict(float)
        for n_chunk, recs_chunk in batch_results:
            # key: "rps", "succ_once", etc.
            for key in recs_chunk[0].keys():
                batch_mean = np.mean([rec[key] for rec in recs_chunk])
                weighted_sums[key] += batch_mean * n_chunk

        summary = {
            "agent": args.agent,
            "task": args.task,
            "subtask": args.subtask,
            "split": args.split,
            "seed": int(seed),
            "ckpt_name": str(ckpt_path.name),
            "n_episodes": int(total_n_episodes),
            "return_per_step": float(weighted_sums["rps"] / total_n_episodes),
            "success_once": float(weighted_sums["succ_once"] / total_n_episodes),
            "success_at_end": float(weighted_sums["succ_end"] / total_n_episodes),
            "plan_file": str(plan_fp),
            "spawn_file": str(spawn_fp),
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

        plan_file_name = Path(args.plan_file).stem
        out_json = ckpt_path.parent / f"{Path(args.ckpt_name).stem}_{plan_file_name}.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

        print("Metrics saved to", out_json)
        print(json.dumps(summary, indent=2))
        all_results.append(summary)

    # Close env after all seeds processed
    env.close()


if __name__ == "__main__":
    main()


