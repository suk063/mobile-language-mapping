#!/bin/bash

set -euo pipefail

EXPS_ROOT="/sh-vol/mobile-language-mapping/mshab_exps"
OUTPUT_DIR="/sh-vol/mobile-language-mapping/eval_results"
SUBTASK="pick"
OBJS=("all_13" "all_10")
CHECKPOINTS=("best_eval_success_once_ckpt.pt" "final_ckpt.pt" "best_eval_return_per_step_ckpt.pt")
BALL_QUERY_K=2
USE_REL_POS=false

if [ ! -d "${EXPS_ROOT}" ]; then
    echo "Experiments root directory not found: ${EXPS_ROOT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Find all experiment directories. We assume they don't contain spaces.
find "${EXPS_ROOT}" -mindepth 1 -maxdepth 1 -type d | while read -r exp_dir; do
    dir_name=$(basename "${exp_dir}")
    echo "Processing experiment: ${dir_name}"

    # Parse TASK, AGENT, SEED from directory name
    # e.g., prepare_groceries-map-0-no-rel-PE-k=2 -> TASK=prepare_groceries, AGENT=map, SEED=0
    # e.g., set_table-uplifted-1 -> TASK=set_table, AGENT=uplifted, SEED=1
    
    TASK=$(echo "$dir_name" | cut -d'-' -f1)
    AGENT=$(echo "$dir_name" | cut -d'-' -f2)
    SEED=$(echo "$dir_name" | cut -d'-' -f3)

    if [ -z "${TASK}" ] || [ -z "${AGENT}" ] || [ -z "${SEED}" ]; then
        echo "Could not parse TASK/AGENT/SEED from ${dir_name}. Skipping."
        continue
    fi
    
    echo "  TASK: ${TASK}, AGENT: ${AGENT}, SEED: ${SEED}"

    # Only run for 'prepare_groceries' task
    if [ "${TASK}" != "prepare_groceries" ]; then
        echo "  Skipping task '${TASK}', only running for 'prepare_groceries'."
        continue
    fi

    # Only run for 'map' agent
    if [ "${AGENT}" != "map" ]; then
        echo "  Skipping agent '${AGENT}', only running for 'map'."
        continue
    fi

    model_dir="${exp_dir}/models"
    if [ ! -d "${model_dir}" ]; then
        echo "  Model directory not found: ${model_dir}. Skipping."
        continue
    fi

    exp_has_error=false
    for ckpt in "${CHECKPOINTS[@]}"; do
        if [ "${exp_has_error}" = true ]; then
            echo "  Skipping remaining checkpoints for ${dir_name} due to previous error."
            break
        fi

        ckpt_path="${model_dir}/${ckpt}"
        if [ ! -f "${ckpt_path}" ]; then
            echo "  Checkpoint not found: ${ckpt_path}. Skipping."
            continue
        fi

        for obj in "${OBJS[@]}"; do
            plan_file="${obj}.json"
            
            # Create a directory for logs
            mkdir -p "${OUTPUT_DIR}/logs"
            LOG_FILE="${OUTPUT_DIR}/logs/${dir_name}---${ckpt}---${obj}.log"
            
            echo "  Running evaluation for CKPT: ${ckpt}, OBJ: ${obj}. Log: ${LOG_FILE}"

            # Prepare common args for eval script
            # Note: other args like --root, --num-envs will use defaults from eval_bc.py
            ARGS=(
              --agent "${AGENT}"
              --task "${TASK}"
              --subtask "${SUBTASK}"
              --plan-file "${plan_file}"
              --seed "${SEED}"
              --model-dir "${model_dir}"
              --ckpt-name "${ckpt}"
            )

            # Add map-only args if needed
            if [ "${AGENT}" = "map" ]; then
              ARGS+=(
                --static-map-path "pretrained/hash_voxel_sparse.pt"
                --implicit-decoder-path "pretrained/implicit_decoder.pt"
                --ball-query-k "${BALL_QUERY_K}"
              )
              if [ "${USE_REL_POS}" = true ]; then
                  ARGS+=(--use-rel-pos)
              fi
            fi

            # Run evaluation and redirect all output to log file
            if ! python -m experiment.eval_bc "${ARGS[@]}" > "${LOG_FILE}" 2>&1; then
                echo "  [ERROR] Evaluation failed for ${dir_name} with CKPT ${ckpt}. Skipping rest of this experiment."
                echo "  See full error log in: ${LOG_FILE}"
                exp_has_error=true
                break
            fi
            echo "  Finished evaluation for CKPT: ${ckpt}, OBJ: ${obj}"
        done
    done
    echo "Finished processing experiment: ${dir_name}"
    echo "-----------------------------------------------------"
done

echo "All evaluations finished."

RESULTS_DIR="${OUTPUT_DIR}/eval_results_$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="${RESULTS_DIR}.tar.gz"

echo "Collecting results into ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Find all experiment directories again to copy results
find "${EXPS_ROOT}" -mindepth 1 -maxdepth 1 -type d | while read -r exp_dir; do
    dir_name=$(basename "${exp_dir}")
    model_dir="${exp_dir}/models"
    
    if [ -d "${model_dir}" ]; then
        # Find all json files and copy them
        json_files=$(find "${model_dir}" -name "*.json")
        if [ -n "${json_files}" ]; then
            # Create a corresponding directory in the results folder
            mkdir -p "${RESULTS_DIR}/${dir_name}/models"
            echo "  Copying results from ${dir_name}"
            find "${model_dir}" -name "*.json" -exec cp -t "${RESULTS_DIR}/${dir_name}/models/" {} +
        fi
    fi
done

echo "Creating archive: ${ARCHIVE_NAME}"
tar -czvf "${ARCHIVE_NAME}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")"

echo "-----------------------------------------------------"
echo "Done. Results are in ${ARCHIVE_NAME}"
echo "You can now download this file to your local machine."
echo "For example, using scp:"
echo "scp user@cluster_address:${ARCHIVE_NAME} /path/on/your/local/machine/"
echo ""
echo "If you want to upload to a specific drive, let me know the details."
