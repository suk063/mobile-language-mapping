#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="${SCRIPT_DIR}/.."
TASK=set_table SUBTASK=pick ./generate_rgbd_data.bash &
TASK=set_table SUBTASK=place ./generate_rgbd_data.bash &
TASK=tidy_house SUBTASK=pick ./generate_rgbd_data.bash &
TASK=tidy_house SUBTASK=place ./generate_rgbd_data.bash &
TASK=prepare_groceries SUBTASK=pick ./generate_rgbd_data.bash &
TASK=prepare_groceries SUBTASK=place ./generate_rgbd_data.bash &
