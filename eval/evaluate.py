import json, torch, numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from mshab.envs.make    import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file
from lang_mapping.agent.baseline.agent_uplift import Agent_uplift
from lang_mapping.agent.baseline.agent_3dencoder import Agent_3dencoder
from lang_mapping.agent.baseline.agent_image import Agent_image
from lang_mapping.dataset import build_object_map, get_object_labels_batch, merge_t_m1


# MARK: Choose agent type
AGENT_KIND = "image"          # {"uplift", "3dencoder", "image"}

TASK  = "set_table"            
MODE  = "pick"                 
OOD   = False                  
PLANS = 100       
# MARK: Change size for vectorized run
NUM_ENVS = 30
CKPT  = Path("/home/woojeh/Documents/mobile-language-mapping/mshab_exps/PickSubtaskTrain-v0/set_table-rcad-bc-point-pick/bc-pick-all-uplift-local-trajs_per_obj=10/models/latest_agent.pt")

ROOT = Path("~/.maniskill/data/scene_datasets/replica_cad_dataset").expanduser()
PLAN_DIR = "task_plans_ood" if OOD else "task_plans"
THIS_DIR = Path(__file__).resolve().parent
PLAN_FP  = ROOT / "rearrange" / PLAN_DIR / TASK / MODE / "train" / "all.json"
SPAWN_FP = ROOT / "rearrange/spawn_data" / TASK / MODE / "train/spawn_data.pt"

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


plan_data = plan_data_from_file(PLAN_FP)
env_cfg = EnvConfig(
    env_id            = "PickSubtaskTrain-v0",
    num_envs          = NUM_ENVS,
    max_episode_steps = 200,                         
    all_plan_count    = len(plan_data.plans),
    task_plan_fp      = str(PLAN_FP),
    spawn_data_fp     = str(SPAWN_FP),
    record_video      = False,
    cat_state         = True,
    cat_pixels        = False,
    frame_stack       = 1,
)
env = make_env(env_cfg, video_path=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Successfully created eval environment!")


uid2lbl = build_object_map(str(PLAN_FP), TEXT_PROMPTS)
sample_obs, _ = env.reset(seed=0)


# MARK: Prepare the agent for eval
COMMON_PARAM = dict(
    sample_obs       = sample_obs,
    single_act_shape = env.single_action_space.shape,
    device           = device,
    open_clip_model  = ("EVA02-L-14", "merged2b_s4b_b131k"),
    text_input       = TEXT_PROMPTS,
    clip_input_dim   = 768,
    camera_intrinsics= (71.9144, 71.9144, 112, 112),
    num_heads        = 8,
    num_layers_transformer = 4,
)

def build_agent(kind: str):
    if kind == "uplift":
        return Agent_uplift(**COMMON_PARAM)
    if kind == "3dencoder":
        return Agent_3dencoder(transf_input_dim=768, **COMMON_PARAM)
    if kind == "image":
        return Agent_image(**COMMON_PARAM)
    raise ValueError(f"Unknown AGENT_KIND {kind}")

agent = build_agent(AGENT_KIND).to(device)
agent.load_state_dict(torch.load(CKPT, map_location=device))
agent.eval()
print(f"Using {AGENT_KIND}, weights loaded from {CKPT}")

def pad_ids(ids: list[int]) -> list[int]:
    return ids + [ids[-1]] * (NUM_ENVS - len(ids))

def subtask_labels_tensor() -> torch.Tensor:
    labs = [
        get_object_labels_batch(uid2lbl, p.composite_subtask_uids)
        for p in env.unwrapped.task_plan
    ]
    return torch.stack(labs, dim=0).to(device)

def run_episode(first_obs: dict[str, torch.Tensor]):
    obs      = first_obs
    prev_obs = first_obs

    plan0   = env.unwrapped.task_plan[0]
    labels  = get_object_labels_batch(uid2lbl,
                                      plan0.composite_subtask_uids).to(device)

    for _ in range(env.max_episode_steps):
        agent_obs = merge_t_m1(prev_obs, obs)

        with torch.no_grad():
            act_seq = agent(agent_obs, labels)

        next_obs, _, _, _, _ = env.step(act_seq[:, 0, :])

        prev_obs = obs
        obs      = next_obs

    return obs

def collect_env_stats(env):

    ret   = torch.as_tensor(env.return_queue, dtype=torch.float32)
    suc1  = torch.as_tensor(env.success_once_queue, dtype=torch.float32)
    sucE  = torch.as_tensor(env.success_at_end_queue, dtype=torch.float32)
    L     = torch.as_tensor(env.length_queue, dtype=torch.float32)

    env.reset_queues()
    max_steps = env.max_episode_steps

    records = []
    for i in range(ret.numel()):
        records.append(
            dict(rps       = (ret[i] / max_steps).item(),
                 succ_once = suc1[i].item(),
                 succ_end  = sucE[i].item(),
                 length    = L[i].item())
        )
    return records


# MARK: Eval first N plans
all_ids   = list(range(min(PLANS, env_cfg.all_plan_count)))
records   = []

for start in tqdm(range(0, len(all_ids), NUM_ENVS), desc="Evaluating"):
    chunk        = all_ids[start : start + NUM_ENVS]
    padded_ids   = pad_ids(chunk)
    obs, _       = env.reset(options={"task_plan_idxs": torch.tensor(padded_ids)})

    run_episode(obs)                   
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
    "plan_file"         : str(PLAN_FP),
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