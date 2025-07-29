#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/daizhirui/Data/mobile_language_mapping"
export PYTHONPATH=${SCRIPT_DIR}/..
for file in $(find ${DATA_DIR}/ -name "all_static.pt"); do
    echo ${file}
    output_pt_path="${file%.pt}_masked_clip.pt"
    if [ -f "${output_pt_path}" ]; then
        echo "Skip existing file: ${output_pt_path}"
        continue
    fi

    set -x
    python \
        "${SCRIPT_DIR}/generate_clip_data.py" \
        --input-pt-path ${file} \
        --output-pt-path ${file%.pt}_masked_clip.pt \
        --mask-out-classes \
            root_arm_1_link_1 \
            root_arm_1_link_2 \
            base_link \
            r_wheel_link \
            l_wheel_link \
            torso_lift_link \
            head_pan_link \
            head_tilt_link \
            head_camear_link \
            head_camera_rgb_frame \
            head_camera_rgb_optical_frame \
            head_camera_depth_frame \
            head_camera_depth_optical_frame \
            shoulder_pan_link \
            shoulder_lift_link \
            upperarm_roll_link \
            elbow_flex_link \
            forearm_roll_link \
            wrist_flex_link \
            wrist_roll_link \
            gripper_link \
            r_gripper_finger_link \
            l_gripper_finger_link \
            bellows_link \
            bellows_link2 \
            estop_link \
            laser_link \
            torso_fixed_link
done
