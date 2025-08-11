#!/usr/bin/bash

set -euo pipefail

# User-tunable params (can be overridden via environment variables before calling)
AGENT=${AGENT:-map}                 # one of: map, image, uplifted, point
TASK=${TASK:-set_table}             # one of: set_table, prepare_groceries, tidy_house
SUBTASK=${SUBTASK:-pick}
SPLIT=${SPLIT:-train}
OBJ=${OBJ:-all_13}
PLAN_FILE=${PLAN_FILE:-${OBJ}.json}

ROOT=${ROOT:-"$HOME/.maniskill/data/scene_datasets/replica_cad_dataset"}
NUM_ENVS=${NUM_ENVS:-16}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-200}
DEVICE=${DEVICE:-cuda}
CKPT_NAME=${CKPT_NAME:-best_eval_success_once_ckpt.pt}
SCENE_IDS_YAML=${SCENE_IDS_YAML:-pretrained/scene_ids.yaml}
OUT_DIR=${OUT_DIR:-eval/results}

# Optional: single seed or multiple seeds (space-separated)
if [ "${SEEDS:-}" != "" ]; then
  # shellcheck disable=SC2206
  SEED_ARGS=(--seeds ${SEEDS})
elif [ "${SEED:-}" != "" ]; then
  SEED_ARGS=(--seed "${SEED}")
else
  SEED_ARGS=(--seed 0)
fi

# Map-specific artifacts
STATIC_MAP_PATH=${STATIC_MAP_PATH:-pretrained/hash_voxel_sparse.pt}
IMPLICIT_DECODER_PATH=${IMPLICIT_DECODER_PATH:-pretrained/implicit_decoder.pt}

# Build common args
ARGS=(
  --agent "${AGENT}"
  --task "${TASK}"
  --subtask "${SUBTASK}"
  --plan-file "${PLAN_FILE}"
  --split "${SPLIT}"
  --root "${ROOT}"
  --num-envs "${NUM_ENVS}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --device "${DEVICE}"
  --scene-ids-yaml "${SCENE_IDS_YAML}"
  --ckpt-name "${CKPT_NAME}"
  --out-dir "${OUT_DIR}"
)

# Add map-only args if needed
if [ "${AGENT}" = "map" ]; then
  ARGS+=(
    --static-map-path "${STATIC_MAP_PATH}"
    --implicit-decoder-path "${IMPLICIT_DECODER_PATH}"
  )
fi

echo "Running eval with: ${ARGS[*]} ${SEED_ARGS[*]}"
python -m experiment.eval_bc "${ARGS[@]}" "${SEED_ARGS[@]}"

