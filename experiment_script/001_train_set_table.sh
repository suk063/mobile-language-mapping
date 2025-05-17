#!/usr/bin/bash

SEED=1

TRAJS_PER_OBJ=1000

TASK=set_table
SUBTASK=pick
SPLIT=train
OBJ=all_10

# Change according to the task
# ALL_PLAN_COUNT=244
ALL_PLAN_COUNT=342
NUM_ENVS=20

# shellcheck disable=SC2001 
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps"
GROUP=$TASK-map_set_table-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/bc-$SUBTASK-$OBJ-map_set_table-trajs_per_obj=$TRAJS_PER_OBJ"
# shellcheck disable=SC2001
PROJECT_NAME="map_set_table"

WANDB=True
TENSORBOARD=True
MS_ASSET_DIR="$HOME/.maniskill/data"

RESUME_LOGDIR="$WORKSPACE/$EXP_NAME"
RESUME_CONFIG="$RESUME_LOGDIR/config.yml"

MAX_CACHE_SIZE=0   # safe num for about 64 GiB system memory

if [[ $SUBTASK == "open" || $SUBTASK == "close" || $SUBTASK == "pick" || $SUBTASK == "place" ]]; then
    data_dir_fp="$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset/$TASK/$SUBTASK/$OBJ.h5"
else
    data_dir_fp="$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset/$TASK/$SUBTASK"
fi

args=(
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "eval_env.env_id=$ENV_ID"
    "eval_env.task_plan_fp=$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
    "eval_env.spawn_data_fp=$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
    "eval_env.frame_stack=1"
    "eval_env.all_plan_count=$ALL_PLAN_COUNT"
    "algo.trajs_per_obj=$TRAJS_PER_OBJ"
    "algo.data_dir_fp=$data_dir_fp"
    "algo.max_cache_size=$MAX_CACHE_SIZE"
    "algo.eval_freq=1"
    "algo.log_freq=1"
    "algo.save_freq=1"
    "eval_env.make_env=True"
    "eval_env.num_envs=$NUM_ENVS"
    "eval_env.max_episode_steps=200"
    "eval_env.record_video=True"
    "eval_env.info_on_video=True"
    "eval_env.save_video_freq=1"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
)

if [ -f "$RESUME_CONFIG" ] && [ -f "$RESUME_LOGDIR/models/latest.pt" ]; then
    echo "RESUMING"
    SAPIEN_NO_DISPLAY=1 python -m experiment.001_train_set_table "$RESUME_CONFIG" RESUME_LOGDIR="$RESUME_LOGDIR" \
        logger.clear_out="False" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"

else
    echo "STARTING"
    SAPIEN_NO_DISPLAY=1 python -m experiment.001_train_set_table configs/001_train_set_table.yml \
        logger.clear_out="True" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"
        
fi
