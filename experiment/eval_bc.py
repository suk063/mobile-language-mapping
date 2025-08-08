import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file

# BC agents
from lang_mapping.agent.agent_map_bc import Agent_map_bc
from lang_mapping.agent.agent_uplifted_bc import Agent_uplifted_bc
from lang_mapping.agent.agent_image_bc import Agent_image_bc
from lang_mapping.agent.agent_point_bc import Agent_point_bc

# Utils for labels, scenes, and eval rollouts
from lang_mapping.utils.dataset import (
    build_object_map,
    build_uid_episode_scene_maps,
)
from lang_mapping.utils.eval import run_eval_episode

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


def collect_env_stats(env):
    ret = torch.as_tensor(env.return_queue, dtype=torch.float32)
    suc1 = torch.as_tensor(env.success_once_queue, dtype=torch.float32)
    sucE = torch.as_tensor(env.success_at_end_queue, dtype=torch.float32)
    L = torch.as_tensor(env.length_queue, dtype=torch.float32)

    env.reset_queues()
    max_steps = env.max_episode_steps

    records = []
    for i in range(ret.numel()):
        records.append(
            dict(
                rps=(ret[i] / max_steps).item(),
                succ_once=suc1[i].item(),
                succ_end=sucE[i].item(),
                length=L[i].item(),
            )
        )
    return records


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
    parser.add_argument("--ckpt-name", default="model_best.pt")
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

    # Output
    parser.add_argument("--out-dir", default="eval/results")

    args = parser.parse_args()

    device = torch.device(args.device)

    root = Path(args.root)

    # Resolve plan file path
    plan_file_arg = Path(args.plan_file)
    plan_fp = root / "rearrange" / "task_plans" / args.task / args.subtask / args.split / args.plan_file
    assert plan_fp.exists(), f"Plan file not found: {plan_fp}"

    # Resolve spawn data path based on task/subtask/split
    spawn_fp = root / "rearrange" / "spawn_data" / args.task / args.subtask / args.split / "spawn_data.pt"
    assert spawn_fp.exists(), f"Spawn data not found: {spawn_fp}"

    # Plans metadata
    plan_data = plan_data_from_file(plan_fp)
    all_plan_count = len(plan_data.plans)

    # Create eval envs
    env_cfg = EnvConfig(
        env_id="PickSubtaskTrain-v0",
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        all_plan_count=all_plan_count,
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
        base_dir = Path(
            f"mshab_exps/PickSubtaskTrain-v0/{args.task}-{args.subtask}/{args.task}-{args.subtask}-{args.agent}-{seed}"
        )
        ckpt_path = base_dir / args.ckpt_name
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt["agent"] if isinstance(ckpt, dict) and "agent" in ckpt else ckpt
        agent.load_state_dict(state_dict)
        agent.eval()

        # Evaluate all plan ids
        records = []
        all_ids = list(range(all_plan_count))

        for start in tqdm(range(0, len(all_ids), args.num_envs), desc=f"Evaluating seed {seed}"):
            chunk = all_ids[start : start + args.num_envs]
            padded_ids = pad_ids(chunk, args.num_envs)
            obs, _ = env.reset(options={"task_plan_idxs": torch.tensor(padded_ids)})

            # Run a full episode rollout for the current batch
            run_eval_episode(env, obs, agent, uid2lbl, uid2scene_id, device)

            # Collect per-episode stats for the unpadded subset
            batch_recs = collect_env_stats(env)[: len(chunk)]
            records.extend(batch_recs)

        # Aggregate results for this seed
        assert len(records) > 0, "No evaluation records collected."
        import numpy as np

        arr = {k: np.array([rec[k] for rec in records]) for k in records[0]}

        summary = {
            "agent": args.agent,
            "task": args.task,
            "subtask": args.subtask,
            "split": args.split,
            "seed": int(seed),
            "ckpt_name": str(ckpt_path.name),
            "n_episodes": int(len(records)),
            "return_per_step": float(arr["rps"].mean()),
            "success_once": float(arr["succ_once"].mean()),
            "success_at_end": float(arr["succ_end"].mean()),
            "plan_file": str(plan_fp),
            "spawn_file": str(spawn_fp),
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_json = out_dir / f"eval_{args.agent}_{args.task}_{args.subtask}_{args.split}_seed{seed}_{timestamp}.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

        print("Metrics saved to", out_json)
        print(json.dumps(summary, indent=2))
        all_results.append(summary)

    # Close env after all seeds processed
    env.close()


if __name__ == "__main__":
    main()


