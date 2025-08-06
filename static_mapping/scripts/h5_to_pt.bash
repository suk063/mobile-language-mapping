#!/usr/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/daizhirui/Data/mobile_language_mapping"

find ${DATA_DIR} -name "*.h5" | xargs -I {} python \
    "${SCRIPT_DIR}/h5_to_pt.py" \
    --h5-file={}
