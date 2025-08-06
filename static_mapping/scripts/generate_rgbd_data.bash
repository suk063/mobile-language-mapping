#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_DEVICE=${CUDA_DEVICE:-0}
SEED=1
TASK=${TASK:-set_table}
SUBTASK=${SUBTASK:-pick}
SPLIT=train
OBJ=${OBJ:-all}

# shader_dir: minimal, default, rt, rt-med, rt-fast. See: ManiSkill/mani_skill/render/shaders.py#L41

MS_ASSET_DIR="${HOME}/.maniskill/data"
TASK_PLAN_FP="${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange/task_plans/${TASK}/${SUBTASK}/${SPLIT}/${OBJ}.json"
SPWAN_DATA_FP="${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange/spawn_data/${TASK}/${SUBTASK}/${SPLIT}/spawn_data.pt"
EPISODE_FP="${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange-dataset/${TASK}/${SUBTASK}/${OBJ}.json"
TRAJ_H5="${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange-dataset/${TASK}/${SUBTASK}/${OBJ}.h5"
#OUTPUT_FILE="${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange-dataset/${TASK}/${SUBTASK}/${OBJ}_static.h5"
OUTPUT_FILE="${HOME}/Data/mobile_language_mapping/${TASK}/${SUBTASK}/${OBJ}_static.pt"

args=(
    "seed=${SEED}"
    "eval_env.env_id=Static${SUBTASK^}Env"
    "eval_env.make_env=True"
    "eval_env.num_envs=100"
    "eval_env.task_plan_fp=${TASK_PLAN_FP}"
    "eval_env.spawn_data_fp=${SPWAN_DATA_FP}"
    "eval_env.frame_stack=1"
    "eval_env.obs_mode=rgbd"
    "eval_env.render_mode=all"
    "eval_env.shader_dir=minimal"
    "load_agent=False"
    "hide_episode_objects=False"
    "truncate_trajectory_at_success=True"
    "fixed_task_plan_idx=-1"
    "fixed_init_config_idx=-1"
    "fixed_spawn_selection_idx=-1"
    "visualize=False"
    "visualize_rgb_gt=False"
    "episode_fp=${EPISODE_FP}"
    "traj_h5=${TRAJ_H5}"
    "output_file=${OUTPUT_FILE}"
)

set -x
if [ ! -f ${OUTPUT_FILE} ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python generate_rgbd_data.py \
        ${SCRIPT_DIR}/../configs/experiment/01_bc_pick_static.yml \
        "${args[@]}"
fi
