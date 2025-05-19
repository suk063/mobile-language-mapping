import os, json, torch, h5py, numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Tuple

from mshab.envs.make    import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file
from lang_mapping.agent.agent_global_gridnet_multiepisode_pointtrans import Agent_global_gridnet_multiepisode_pointtrans
from lang_mapping.module.mlp import ImplicitDecoder
from lang_mapping.grid_net import GridNet
from lang_mapping.dataset import build_object_map, get_object_labels_batch, merge_t_m1, build_episode_subtask_maps

CKPT  = Path("/home/woojeh/Documents/mobile-language-mapping/mshab_exps/PickSubtaskTrain-v0/map/latest_agent.pt")
TASK  = "set_table"
MODE  = "pick"
OOD = False
PLAN_DIR = "task_plans_ood" if OOD else "task_plans"
suffix = "_ood" if OOD else ""
PLANS = 100 

# MARK: Change size for vectorized run
NUM_ENVS = 30

ROOT = Path("~/.maniskill/data/scene_datasets/replica_cad_dataset").expanduser()
H5    = f"{ROOT}/rearrange-dataset/{TASK}/{MODE}/all.h5"
META  = H5.replace(".h5", ".json")

PLAN  = ROOT / "rearrange" / PLAN_DIR / TASK / MODE / "train" / "all.json"
SPAWN = ROOT / "rearrange/spawn_data" / TASK / MODE / "train/spawn_data.pt"

# MARK: text prompt lookup
PROMPT_MAP = {
    "set_table": ["bowl", "apple"],
    "tidy_house": [
        "gelatin_box", "bowl", "cracker_box", "pudding_box"
    ],
    "prepare_groceries": [
        "tomato_soup_can", "cracker_box", "tuna_fish_can", "bowl",
        "sugar_box", "master_chef_can", "pudding_box",
        "potted_meat_can", "gelatin_box"
    ],
}
TEXT_PROMPTS = PROMPT_MAP[TASK]

VID   = Path(__file__).parent / "videos"; VID.mkdir(exist_ok=True, parents=True)
TRAJ_DIR = Path(__file__).parent / "traj" / TASK / MODE
TRAJ_DIR.mkdir(exist_ok=True, parents=True)
THIS_DIR = Path(__file__).resolve().parent
CLIP_MODEL_NAME     = "EVA02-L-14"
CLIP_WEIGHTS_ID     = "merged2b_s4b_b131k"
CLIP_INPUT_DIM      = 768
CAMERA_INTRINSICS   = (71.9144, 71.9144, 112, 112)
NUM_TRANSFORMER_LAY = 4
NUM_HEADS           = 8

meta = json.load(open(META))
IGNORE = {"obs_mode","reward_mode","control_mode","render_mode","shader_dir","robot_uids","num_envs","sim_backend"}

plan_data = plan_data_from_file(PLAN)
env_cfg = EnvConfig(
    env_id            = "PickSubtaskTrain-v0",
    num_envs          = NUM_ENVS,
    max_episode_steps = 200,                         
    all_plan_count    = len(plan_data.plans),
    task_plan_fp      = str(PLAN),
    spawn_data_fp     = str(SPAWN),
    record_video      = False,
    cat_state         = True,
    cat_pixels        = False,
    frame_stack       = 1,
)
env = make_env(env_cfg, video_path=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GridDefinition:
    type: str = "regular"
    feature_dim: int = 60
    init_stddev: float = 0.2
    bound: List[List[float]] = field(default_factory=lambda: [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]])
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
class AlgoCfg:
    grid_cfg: GridCfg = field(default_factory=GridCfg)

@dataclass
class Cfg:
    algo: AlgoCfg = field(default_factory=AlgoCfg)

cfg = Cfg()
static_map = GridNet(cfg=asdict(cfg.algo.grid_cfg))
implicit_decoder = ImplicitDecoder(
    voxel_feature_dim=120,
    hidden_dim=512,
    output_dim=CLIP_INPUT_DIM,
).to(device)

uid2lbl = build_object_map(PLAN, TEXT_PROMPTS)

def load_changed_centers(root_dir: str, device: Union[torch.device, str] = "cpu") -> dict:
    centers = {}
    for fp in sorted(Path(root_dir).glob("level*_centers.npz")):
        lvl = int(fp.stem.split("_")[0].lstrip("level"))
        data = np.load(fp, allow_pickle=True)
        centers[lvl] = {int(k): torch.as_tensor(data[k], dtype=torch.float32, device=device) for k in data.files}
    if not centers:
        raise FileNotFoundError(f"No center files found in {root_dir}")
    return centers

def pad_ids(ids: list[int]) -> list[int]:
    return ids + [ids[-1]] * (NUM_ENVS - len(ids))

def subtask_labels_tensor() -> torch.Tensor:
    labs = [
        get_object_labels_batch(uid2lbl, p.composite_subtask_uids)
        for p in env.unwrapped.task_plan
    ]
    return torch.stack(labs, dim=0).to(device)

def collect_env_stats(env):
    ret   = torch.as_tensor(env.return_queue, dtype=torch.float32)
    suc1  = torch.as_tensor(env.success_once_queue, dtype=torch.float32)
    sucE  = torch.as_tensor(env.success_at_end_queue, dtype=torch.float32)
    L     = torch.as_tensor(env.length_queue, dtype=torch.float32)
    env.reset_queues()
    max_steps = env.max_episode_steps
    return [
        dict(rps = (ret[i] / max_steps).item(),
             succ_once = suc1[i].item(),
             succ_end = sucE[i].item(),
             length = L[i].item())
        for i in range(ret.numel())
    ]

def run_episode(env, agent, obs, subtask_labels, epi_ids, save_states=True):
    actions_buf, state_buf = [], []
    prev_obs = obs
    for _ in range(env.max_episode_steps):
        with torch.no_grad():
            agent_in = merge_t_m1(prev_obs, obs)
            act = agent.forward_policy(agent_in, subtask_labels, epi_ids)
        if save_states:
            actions_buf.append(act.cpu().numpy())
            state_buf.append(obs["state"].cpu().numpy())
        prev_obs = obs
        obs, *_ = env.step(act[:, 0, :])
    return obs, actions_buf, state_buf


obs, _ = env.reset(seed=0)

agent = Agent_global_gridnet_multiepisode_pointtrans(
    sample_obs             = obs,
    single_act_shape       = env.single_action_space.shape,
    device                 = device,
    open_clip_model        = (CLIP_MODEL_NAME, CLIP_WEIGHTS_ID),
    text_input             = TEXT_PROMPTS,
    clip_input_dim         = CLIP_INPUT_DIM,
    camera_intrinsics      = CAMERA_INTRINSICS,
    num_heads              = NUM_HEADS,
    num_layers_transformer = NUM_TRANSFORMER_LAY,
    static_map=static_map,
    implicit_decoder=implicit_decoder
).to(device)

agent.load_state_dict(torch.load(CKPT, map_location=device))
agent.eval()
agent.valid_coords = load_changed_centers(root_dir="pre-trained/set_table/13", device=device)
print(f"✓ Loaded weights from {CKPT}")

agent.sample_obs = obs
plan0 = env.unwrapped.task_plan[0]
subtask_labels = get_object_labels_batch(uid2lbl, plan0.composite_subtask_uids).to(device)
episode2subtasks, episode2id, uid2episode_id = build_episode_subtask_maps(PLAN)
epi_ids = torch.tensor([uid2episode_id[uid] for uid in plan0.composite_subtask_uids], device=device, dtype=torch.long)


# MARK: Eval first N plans
all_ids   = list(range(min(PLANS, env_cfg.all_plan_count)))
records   = []

for start in tqdm(range(0, len(all_ids), NUM_ENVS), desc="Evaluating"):
    chunk        = all_ids[start : start + NUM_ENVS]
    padded_ids   = pad_ids(chunk)
    obs, _       = env.reset(options={"task_plan_idxs": torch.tensor(padded_ids)})

    run_episode(env, agent, obs, subtask_labels, epi_ids, save_states=False)                  
    batch_recs   = collect_env_stats(env)[:len(chunk)]
    records.extend(batch_recs)

env.close()

# MARK: Aggregate results
arr = {k: np.array([rec[k] for rec in records]) for k in records[0]}
summary = {
    "task"              : TASK,
    "ood"               : OOD,
    "n_episodes"        : int(len(records)),
    "return_per_step"   : float(arr["rps"].mean()),
    "success_once"      : float(arr["succ_once"].mean()),
    "success_at_end"    : float(arr["succ_end"].mean()),
    "len"               : float(arr["length"].mean()),
    "ckpt"              : str(CKPT),
    "plan_file"         : str(PLAN),
    "timestamp_utc"     : datetime.utcnow().isoformat(),
}
results_dir = THIS_DIR / "results"
results_dir.mkdir(exist_ok=True)
timestamp   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out_json    = results_dir / f"eval_results_{timestamp}.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print("Metrics saved to", out_json)
print(json.dumps(summary, indent=2))