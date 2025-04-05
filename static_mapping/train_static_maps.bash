#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/home/daizhirui/results/mobile_language_mapping/static_mapping
export PYTHONPATH=${SCRIPT_DIR}/..

python ${SCRIPT_DIR}/train_static_map.py ${SCRIPT_DIR}/train_config.yaml \
    depth_downsample_method="nearest" \
    output_dir=${OUTPUT_DIR}/nearest

python ${SCRIPT_DIR}/train_static_map.py ${SCRIPT_DIR}/train_config.yaml \
    depth_downsample_method="avg3d" \
    output_dir=${OUTPUT_DIR}/avg3d

python ${SCRIPT_DIR}/train_static_map.py ${SCRIPT_DIR}/train_config.yaml \
    depth_downsample_method="nearest-exact" \
    output_dir=${OUTPUT_DIR}/nearest-exact

python ${SCRIPT_DIR}/train_static_map.py ${SCRIPT_DIR}/train_config.yaml \
    depth_downsample_method="avg2d" \
    output_dir=${OUTPUT_DIR}/avg2d
