import os, json, torch, h5py, numpy as np
from datetime import datetime
from pathlib import Path

from mshab.envs.make    import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file
from lang_mapping.agent.baseline.agent_uplift import Agent_uplift
from lang_mapping.dataset import build_object_map, get_object_labels_batch, merge_t_m1

CKPT  = Path("/home/woojeh/Documents/mobile-language-mapping/mshab_exps/PickSubtaskTrain-v0/set_table-rcad-bc-point-pick/bc-pick-all-uplift-local-trajs_per_obj=10/models/latest_agent.pt")
TRAJ  = 370                                   
TASK  = "set_table"                        
MODE  = "pick"

OOD = True
plan_dir = "task_plans_ood" if OOD else "task_plans"

suffix = "_ood" if OOD else ""

ROOT = os.path.expanduser("~/.maniskill/data/scene_datasets/replica_cad_dataset")
H5    = f"{ROOT}/rearrange-dataset/{TASK}/{MODE}/all.h5"
META  = H5.replace(".h5", ".json")
PLAN  = f"{ROOT}/rearrange/{plan_dir}/{TASK}/{MODE}/train/all.json"
SPAWN = f"{ROOT}/rearrange/spawn_data/{TASK}/{MODE}/train/spawn_data.pt"

VID   = Path(__file__).parent / "videos"; VID.mkdir(exist_ok=True, parents=True)
TRAJ_DIR = Path(__file__).parent / "traj" / TASK / MODE
TRAJ_DIR.mkdir(exist_ok=True, parents=True)           
OUT_H5 = TRAJ_DIR / f"traj{suffix}_{TRAJ:03d}.h5"
OUT_JS = TRAJ_DIR / f"traj{suffix}_{TRAJ:03d}.json"

TEXT_PROMPTS        = ["bowl", "apple"]
CLIP_MODEL_NAME     = "EVA02-L-14"
CLIP_WEIGHTS_ID     = "merged2b_s4b_b131k"
CLIP_INPUT_DIM      = 768
CAMERA_INTRINSICS   = (71.9144, 71.9144, 112, 112)
NUM_TRANSFORMER_LAY = 4
NUM_HEADS           = 8


meta = json.load(open(META)); epi = meta["episodes"][TRAJ]
IGNORE = {"obs_mode","reward_mode","control_mode","render_mode",
          "shader_dir","robot_uids","num_envs","sim_backend"}

env_cfg = EnvConfig(
    env_id            = meta["env_info"]["env_id"],
    num_envs          = 1,
    max_episode_steps = meta["env_info"]["max_episode_steps"],
    all_plan_count    = len(plan_data_from_file(PLAN).plans),
    task_plan_fp      = PLAN,
    spawn_data_fp     = SPAWN,
    record_video      = True,
    cat_state         = True,
    cat_pixels        = False,
    frame_stack       = 1,
    env_kwargs        = {k:v for k,v in meta["env_info"]["env_kwargs"].items() if k not in IGNORE},
)

env    = make_env(env_cfg, video_path=VID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# reset env with task specific config
obs, _ = env.reset(
    options=dict(
        reconfigure           = True,
        build_config_idxs     = torch.tensor([epi["build_config_idx"]]),
        init_config_idxs      = torch.tensor([epi["init_config_idx"]]),
        task_plan_idxs        = torch.tensor([epi["task_plan_idx"]]),
        spawn_selection_idxs  = torch.tensor([epi["spawn_selection_idx"]]),
    )
)


uid2lbl = build_object_map(PLAN, TEXT_PROMPTS)
agent = Agent_uplift(
    sample_obs             = obs,
    single_act_shape       = env.single_action_space.shape,
    device                 = device,
    open_clip_model        = (CLIP_MODEL_NAME, CLIP_WEIGHTS_ID),
    text_input             = TEXT_PROMPTS,
    clip_input_dim         = CLIP_INPUT_DIM,
    camera_intrinsics      = CAMERA_INTRINSICS,
    num_heads              = NUM_HEADS,
    num_layers_transformer = NUM_TRANSFORMER_LAY,
).to(device)
agent.load_state_dict(torch.load(CKPT, map_location=device))
agent.eval()
print(f"✓ Loaded weights from {CKPT}")

# buffers for new trajectory
actions_buf, state_buf = [], []

prev_obs = obs
plan0 = env.unwrapped.task_plan[0]
subtask_labels = get_object_labels_batch(uid2lbl, plan0.composite_subtask_uids).to(device)

for _ in range(env.max_episode_steps):
    with torch.no_grad():
        agent_in = merge_t_m1(prev_obs, obs)
        act      = agent(agent_in, subtask_labels)      # (1,1,A)

    # store action and state
    actions_buf.append(act.cpu().numpy())               # (1,1,A)
    state_buf.append(obs["state"].cpu().numpy())        # (1, state_dim)

    prev_obs = obs
    obs, *_  = env.step(act[:, 0, :])

env.close()
print("✓ Episode finished, writing trajectory...")

# MARK: Below for later use, loading trajectory

# save .h5
with h5py.File(OUT_H5, "w") as f:
    g = f.create_group(f"traj_{TRAJ}")
    g.create_dataset("actions", data=np.concatenate(actions_buf, axis=0))  # (T,A)
    g.create_dataset("state",   data=np.concatenate(state_buf,   axis=0))  # (T,S)

print("  → saved", OUT_H5.relative_to(Path.cwd()))

# Save json
json.dump(
    {
        "episode_id"         : TRAJ,
        "task_plan_idx"      : int(epi["task_plan_idx"]),
        "build_config_idx"   : int(epi["build_config_idx"]),
        "init_config_idx"    : int(epi["init_config_idx"]),
        "spawn_selection_idx": int(epi["spawn_selection_idx"]),
        "timestamp_utc"      : datetime.utcnow().isoformat(),
    },
    open(OUT_JS, "w"), indent=2,
)
print("  → meta saved", OUT_JS.relative_to(Path.cwd()))

print("video saved to", VID / "ep_000000.mp4")
