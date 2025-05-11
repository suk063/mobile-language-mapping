import os, json, h5py, torch
from pathlib import Path
from mshab.envs.make    import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file

TRAJ = 118                                                    
ROOT = os.path.expanduser("~/.maniskill/data/scene_datasets/replica_cad_dataset")

H5   = f"{ROOT}/rearrange-dataset/prepare_groceries/pick/all.h5"
META = H5.replace(".h5", ".json")
PLAN = f"{ROOT}/rearrange/task_plans/prepare_groceries/pick/train/all.json"
SPAWN= f"{ROOT}/rearrange/spawn_data/prepare_groceries/pick/train/spawn_data.pt"
VID  = Path(__file__).parent/"videos"; VID.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
meta = json.load(open(META)); epi = meta["episodes"][TRAJ]


with h5py.File(H5) as f:
    g        = f[f"traj_{TRAJ}"]
    actions  = torch.from_numpy(g["actions"][...])           # (T,13)
    qpos0    = torch.from_numpy(g["obs/agent/qpos"][0])
    qvel0    = torch.from_numpy(g["obs/agent/qvel"][0])


IGNORE = {"obs_mode","reward_mode","control_mode","render_mode","shader_dir",
       "robot_uids","num_envs","sim_backend"}
env_cfg = EnvConfig(
    env_id            = meta["env_info"]["env_id"],
    num_envs          = 1,
    max_episode_steps = meta["env_info"]["max_episode_steps"],
    all_plan_count    = len(plan_data_from_file(PLAN).plans),
    task_plan_fp      = PLAN,
    spawn_data_fp     = SPAWN,
    record_video      = True,
    env_kwargs        ={k:v for k,v in meta["env_info"]["env_kwargs"].items()
                        if k not in IGNORE},
)
env = make_env(env_cfg, video_path=VID)

# Reset with configuration
env.reset(
    options = dict(
        reconfigure           = True,
        build_config_idxs     = torch.tensor([epi["build_config_idx"]]),
        init_config_idxs      = torch.tensor([epi["init_config_idx"]]),
        task_plan_idxs        = torch.tensor([epi["task_plan_idx"]]),
        spawn_selection_idxs = torch.tensor([epi["spawn_selection_idx"]])
    ),
)

# Load to gpu
actions = actions.to(env.device)

for a in actions:
    env.step(a.unsqueeze(0))

print("video saved to", VID/"ep_000000.mp4")
env.close()