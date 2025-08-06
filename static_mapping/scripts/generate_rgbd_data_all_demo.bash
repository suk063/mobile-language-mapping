#!/usr/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="${SCRIPT_DIR}/.."

TASK=set_table OBJ=all_10 TASK_PLAN_IDX=107 INIT_CONFIG_IDX=53 SPAWN_SELECTION_IDX=76 ./generate_rgbd_data_demo.bash
TASK=set_table OBJ=all_13 TASK_PLAN_IDX=51 INIT_CONFIG_IDX=13 SPAWN_SELECTION_IDX=18 ./generate_rgbd_data_demo.bash
TASK=tidy_house OBJ=all_10 TASK_PLAN_IDX=331 INIT_CONFIG_IDX=51 SPAWN_SELECTION_IDX=18 ./generate_rgbd_data_demo.bash
TASK=tidy_house OBJ=all_13 TASK_PLAN_IDX=111 INIT_CONFIG_IDX=127 SPAWN_SELECTION_IDX=18 ./generate_rgbd_data_demo.bash
TASK=prepare_groceries OBJ=all_10 TASK_PLAN_IDX=315 INIT_CONFIG_IDX=39 SPAWN_SELECTION_IDX=88 ./generate_rgbd_data_demo.bash
TASK=prepare_groceries OBJ=all_13 TASK_PLAN_IDX=174 INIT_CONFIG_IDX=45 SPAWN_SELECTION_IDX=37 ./generate_rgbd_data_demo.bash
