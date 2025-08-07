#!/usr/bin/bash

SEED=1

TRAJS_PER_OBJ=1000

REPRESENTATION=point
TASK=prepare_groceries
SUBTASK=pick
SPLIT=train
OBJ=all_13

# Change according to the task
NUM_ENVS=10

# shellcheck disable=SC2001 
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps"
GROUP=$TASK-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/$TASK-$SUBTASK-point"
# shellcheck disable=SC2001
PROJECT_NAME="$TASK-$SUBTASK-point"

WANDB=True
TENSORBOARD=True
MS_ASSET_DIR="$HOME/.maniskill/data"

RESUME_LOGDIR="/sh-vol/mobile-language-mapping/$WORKSPACE/$EXP_NAME"
RESUME_CONFIG="$RESUME_LOGDIR/config.yml"


MAX_IMAGE_CACHE_SIZE=0   # safe num for about 64 GiB system memory
NUM_DATALOAD_WORKERS=2
data_dir_fp="$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset/$TASK/$SUBTASK/$OBJ.h5"

args=(
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "eval_env.env_id=$ENV_ID"
    "eval_env.task_plan_fp=$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
    "eval_env.spawn_data_fp=$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
    "eval_env.frame_stack=1"
    "algo.representation=$REPRESENTATION"
    "algo.trajs_per_obj=$TRAJS_PER_OBJ"
    "algo.data_dir_fp=$data_dir_fp"
    "algo.max_image_cache_size=$MAX_IMAGE_CACHE_SIZE"
    "algo.num_dataload_workers=$NUM_DATALOAD_WORKERS"
    "algo.eval_freq=1"
    "algo.log_freq=1"
    "algo.save_freq=1"
    "algo.save_backup_ckpts=True"
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
    SAPIEN_NO_DISPLAY=1 python -m experiment.train_bc "$RESUME_CONFIG" \
        resume_logdir="$RESUME_LOGDIR" \
        logger.clear_out="False" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"
else
    echo "STARTING"
    SAPIEN_NO_DISPLAY=1 python -m experiment.train_bc configs/point/train_${REPRESENTATION}_${TASK}.yml \
        logger.clear_out="True" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"
fi
