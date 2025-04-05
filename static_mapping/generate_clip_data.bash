#!/usr/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/daizhirui/Data/mobile_language_mapping"
export PYTHONPATH=${SCRIPT_DIR}/..
for file in $(find ${DATA_DIR} -name "*.pt"); do
    python \
        "${SCRIPT_DIR}/generate_clip_data.py" \
        --input-pt-path ${file} \
        --output-pt-path ${file%.pt}_clip.pt
done
