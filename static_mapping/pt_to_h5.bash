#!/usr/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/daizhirui/Data/mobile_language_mapping"

find ${DATA_DIR} -name "*.pt" | xargs -I {} python \
    "${SCRIPT_DIR}/pt_to_h5.py" \
    --input-pt-path={}
