#!/usr/bin/env python3
"""
For every trajectory in .h5 file (or in an entire directory tree) it
computes:

* final camera pose (xyz + roll‑pitch‑yaw) for head & hand sensors
* distance‑to‑goal stats (min / final)
* VISIBILITY‑centric measures that highlight “goal out‑of‑view” periods
    - vis_frac_*            : fraction of frames where the goal is inside FOV
    - longest_invis_gap_*   : longest contiguous run of frames with the goal out‑of‑view
    - t_since_view_to_grasp : # steps between last sight of goal and grasp time
* task outcome flags
"""

import argparse
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
#  Geometry helpers                                                           #
# --------------------------------------------------------------------------- #
def rpy_deg(R_3x3: np.ndarray):
    """3×3 rotation matrix  → roll‑pitch‑yaw in *degrees* (XYZ intrinsic)."""
    return R.from_matrix(R_3x3).as_euler("xyz", degrees=True)


def cam_pose(extrinsic_3x4: np.ndarray):
    """
    Split a 3×4 OpenCV extrinsic [R | t] (cam→base) into position (xyz) and
    orientation (r,p,y°) expressed in the *base* frame.
    """
    T = np.eye(4, dtype=float)
    T[:3, :4] = extrinsic_3x4
    xyz = T[:3, 3]
    rpy = rpy_deg(T[:3, :3])
    return xyz, rpy


# --------------------------------------------------------------------------- #
#  Visibility helper                                                          #
# --------------------------------------------------------------------------- #
def in_view(goal_xyz, ext_3x4, K, img_h=224, img_w=224):
    """
    Returns True if the 3‑D goal projects inside the image bounds of this
    camera frame.

    goal_xyz : (3,)         in *base* frame
    ext_3x4  : 3×4 cam→base OpenCV matrix (right‑handed, Z forward)
    K        : 3×3 intrinsics (fx, fy, cx, cy)
    """
    # Build homogeneous transform and invert to get base→cam
    T_cb = np.eye(4);  T_cb[:3, :4] = ext_3x4
    T_bc = np.linalg.inv(T_cb)

    # Transform goal into camera frame
    g_cam = T_bc[:3, :3] @ goal_xyz + T_bc[:3, 3]           # (3,)

    if g_cam[2] <= 0:                    # behind camera
        return False

    # Project
    u = (K[0, 0] * g_cam[0] / g_cam[2]) + K[0, 2]
    v = (K[1, 1] * g_cam[1] / g_cam[2]) + K[1, 2]
    return (0 <= u < img_w) and (0 <= v < img_h)


# --------------------------------------------------------------------------- #
#  Metric extraction for a single trajectory                                  #
# --------------------------------------------------------------------------- #
def visibility_series(goal, ext, K):
    """Return visibility stats for an entire trajectory (bool per frame)."""
    vis = np.array([in_view(g, e, K) for g, e in zip(goal, ext)], dtype=bool)
    # longest contiguous False stretch
    if (~vis).any():
        # run‑length encoding of visibility changes
        edges = np.where(np.concatenate(([vis[0]], vis[:-1] != vis[1:], [True])))[0]
        gap_lengths = np.diff(edges)[::2]  # lengths of False segments
        longest_gap = int(gap_lengths.max())
    else:
        longest_gap = 0

    first_seen  = int(np.argmax(vis)) if vis.any() else -1
    last_seen   = int(len(vis) - 1 - np.argmax(vis[::-1])) if vis.any() else -1
    return vis.mean(), longest_gap, first_seen, last_seen, vis


def extract_metrics(traj):
    """Compute all metrics for a single h5py.Group representing one trajectory."""
    goal   = traj["obs/extra/goal_pos_wrt_base"][:]          # (T,3)
    objpos = traj["obs/extra/obj_pose_wrt_base"][:, :3]      # (T,3)
    grasp  = traj["obs/extra/is_grasped"][:]                 # (T,)
    succ   = traj["success"][:]                              # (T,)

    head_ext = traj["obs/sensor_param/fetch_head/extrinsic_cv"][:]  # (T,3,4)
    hand_ext = traj["obs/sensor_param/fetch_hand/extrinsic_cv"][:]

    # Distance series
    head_xyz = head_ext[..., 3]                              # (T,3)
    hand_xyz = hand_ext[..., 3]
    head_dist = np.linalg.norm(goal - head_xyz, axis=1)
    hand_dist = np.linalg.norm(goal - hand_xyz, axis=1)

    # Intrinsics (assumed constant)
    K_head = traj["obs/sensor_param/fetch_head/intrinsic_cv"][0]
    K_hand = traj["obs/sensor_param/fetch_hand/intrinsic_cv"][0]

    # Visibility metrics
    head_vis_frac, head_long_gap, head_first, head_last, _ = \
        visibility_series(goal, head_ext, K_head)
    hand_vis_frac, hand_long_gap, hand_first, hand_last, _ = \
        visibility_series(goal, hand_ext, K_hand)

    # Time from last view to grasp
    if grasp.any():
        grasp_step = int(np.argmax(grasp))
        t_since_view_head = grasp_step - head_last
        t_since_view_hand = grasp_step - hand_last
    else:
        grasp_step, t_since_view_head, t_since_view_hand = -1, -1, -1

    final = -1  # last timestep
    head_xyz_f, head_rpy_f = cam_pose(head_ext[final])
    hand_xyz_f, hand_rpy_f = cam_pose(hand_ext[final])

    return dict(
        traj_id                  = traj.name,
        # final camera pose
        head_xyz_final           = head_xyz_f,
        head_rpy_final_deg       = head_rpy_f,
        hand_xyz_final           = hand_xyz_f,
        hand_rpy_final_deg       = hand_rpy_f,
        # distance stats
        min_head_goal_dist       = float(head_dist.min()),
        min_hand_goal_dist       = float(hand_dist.min()),
        final_head_goal_dist     = float(head_dist[final]),
        final_hand_goal_dist     = float(hand_dist[final]),
        # visibility stats
        vis_frac_head            = float(head_vis_frac),
        vis_frac_hand            = float(hand_vis_frac),
        longest_invis_gap_head   = head_long_gap,
        longest_invis_gap_hand   = hand_long_gap,
        t_since_view_to_grasp_h  = t_since_view_head,
        t_since_view_to_grasp_d  = t_since_view_hand,
        # outcome
        grasp_success_step       = grasp_step,
        overall_success          = bool(succ[-1]),
        traj_length              = int(len(goal)),
    )


default_demo = Path.home() / ".maniskill/data/scene_datasets/replica_cad_dataset" \
                            "/rearrange-dataset/set_table/pick/all.h5"

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default=str(default_demo),
                help="trajectory *.h5 or directory (default: %(default)s)")
ap.add_argument("-o", "--out_csv", default="trajectory_metrics.csv",
                help="save metrics CSV (default: %(default)s)")
args = ap.parse_args()

src = Path(args.path)
files = [src] if src.is_file() else sorted(src.rglob("*.h5"))
if not files:
    raise FileNotFoundError(f"No *.h5 under {src}")

rows = []
for fp in files:
    with h5py.File(fp, "r") as f:
        for traj in f.values():
            rows.append(extract_metrics(traj))

df = pd.DataFrame(rows)


print("\n=== FINAL CAMERA POSES ===")
print(df[["traj_id",
            "head_xyz_final", "head_rpy_final_deg",
            "hand_xyz_final", "hand_rpy_final_deg"]])

print("\n=== SUMMARY METRICS ===")
cols = ["traj_id",
        "vis_frac_head", "vis_frac_hand",
        "longest_invis_gap_head", "longest_invis_gap_hand",
        "t_since_view_to_grasp_h", "t_since_view_to_grasp_d",
        "final_head_goal_dist", "final_hand_goal_dist",
        "min_head_goal_dist", "min_hand_goal_dist",
        "overall_success"]
print(df[cols].to_string(index=False))

# optional CSV
if args.out_csv:
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV to {args.out_csv}")
