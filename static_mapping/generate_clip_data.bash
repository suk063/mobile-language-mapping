#!/usr/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HOME}/Data/mobile_language_mapping"
export PYTHONPATH=${SCRIPT_DIR}/..
for file in $(find ${DATA_DIR}/ -name "all_static.pt"); do
    echo ${file}
    output_pt_path="${file%.pt}_clip.pt"
    if [ -f "${output_pt_path}" ]; then
        echo "Skip existing file: ${output_pt_path}"
        continue
    fi

    set -x
    python \
        "${SCRIPT_DIR}/generate_clip_data.py" \
        --input-pt-path ${file} \
        --output-pt-path ${file%.pt}_clip.pt
done
