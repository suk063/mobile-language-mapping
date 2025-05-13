import argparse
import itertools
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
#  Geometry helpers                                                           #
# --------------------------------------------------------------------------- #
def rpy_deg(R_3x3: np.ndarray):
    """3×3 rotation matrix  → roll‑pitch‑yaw in *degrees* (XYZ intrinsic)."""
    return R.from_matrix(R_3x3).as_euler("xyz", degrees=True)


def cam_pose(ext_3x4: np.ndarray):
    """
    Interpret a 3×4 OpenCV extrinsic [R | t] as *base→camera* and invert
    it to get *camera→base*. Then extract the camera position (xyz) and
    orientation (r,p,y in degrees) in the *base* frame.

    ext_3x4 : 3×4 matrix that takes a point from base frame to camera frame.
    Returns:
        xyz: (3,) camera position in base frame
        rpy: (3,) camera orientation (roll, pitch, yaw in degrees)
    """
    # 1) Make a 4×4
    T_bc = np.eye(4, dtype=float)
    T_bc[:3, :4] = ext_3x4  # base→camera

    # 2) Invert to get camera→base
    T_cb = np.linalg.inv(T_bc)

    # 3) Camera position/orientation in base frame
    xyz = T_cb[:3, 3]
    rpy = rpy_deg(T_cb[:3, :3])
    return xyz, rpy

def camera_forward_axis_in_base(ext_3x4: np.ndarray):
    """
    Given base->camera extrinsic, return camera's +Z axis in the base frame.
    If R_bc = ext_3x4[:3, :3], columns of R_bc are camera axes in the base frame.
    So R_bc[:, 2] is the camera's forward axis (+Z).
    """
    R_bc = ext_3x4[:3, :3]
    return R_bc[:, 2]

# --------------------------------------------------------------------------- #
#  Visibility helper                                                          #
# --------------------------------------------------------------------------- #
def in_view(goal_xyz, ext_3x4, K, img_h=224, img_w=224):
    """
    Returns True if the 3D goal (in base frame) projects inside the image
    bounds of this camera. We assume `ext_3x4` is a base→camera transform
    under the OpenCV convention (i.e. R right-handed, +Z forward).

    Args:
        goal_xyz : (3,) in base frame
        ext_3x4  : 3×4 base→camera matrix (OpenCV extrinsic)
        K        : 3×3 intrinsics (fx, fy, cx, cy)
        img_h, img_w : image size
    """
    # --- Convert from base to camera frame directly (no inversion) ---
    R_bc = ext_3x4[:3, :3]  # rotation
    t_bc = ext_3x4[:3, 3]   # translation
    g_cam = R_bc @ goal_xyz + t_bc  # (3,)

    # behind camera?
    if g_cam[2] <= 0:
        return False

    # --- Project to 2D pixel coords
    u = (K[0, 0] * g_cam[0] / g_cam[2]) + K[0, 2]
    v = (K[1, 1] * g_cam[1] / g_cam[2]) + K[1, 2]

    return (0 <= u < img_w) and (0 <= v < img_h)

def view_angle_deg(goal_xyz, ext_3x4):
    """
    Computes the angle (in degrees) between the camera's +Z axis and the
    direction from camera center to `goal_xyz`, all in the base frame.
    """
    # Get camera center in base
    cam_xyz, _ = cam_pose(ext_3x4)

    # Get camera's +Z axis in base
    forward_axis = camera_forward_axis_in_base(ext_3x4)

    # Direction from camera to goal
    vec_cam2goal = goal_xyz - cam_xyz
    norm = np.linalg.norm(vec_cam2goal)
    if norm < 1e-8:
        # The goal and camera center are basically the same point
        return 0.0

    vec_cam2goal /= norm

    # Angle
    dot_prod = np.dot(vec_cam2goal, forward_axis)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)  # safeguard numerical issues
    angle_rad = np.arccos(dot_prod)
    return float(np.degrees(angle_rad))

# --------------------------------------------------------------------------- #
#  Metric extraction for a single trajectory                                  #
# --------------------------------------------------------------------------- #
def visibility_series(goal, ext, K):
    """
    Return:
        vis_frac      : fraction of frames goal is in FOV   (0–1)
        longest_gap   : longest run of *invisible* frames   (int)
        first_seen    : first timestep goal becomes visible (–1 if never)
        last_seen     : last  timestep goal is visible      (–1 if never)
        vis           : boolean array (T,)
    """
    vis = np.array([in_view(g, e, K) for g, e in zip(goal, ext)], dtype=bool)

    if vis.all():                        # always visible
        longest_gap = 0
    elif (~vis).all():                   # never visible
        longest_gap = len(vis)
    else:
        # find longest contiguous False segment
        longest_gap = max(len(list(group))
                          for val, group in itertools.groupby(vis) if not val)

    first_seen = int(np.argmax(vis)) if vis.any() else -1
    last_seen  = int(len(vis) - 1 - np.argmax(vis[::-1])) if vis.any() else -1

    return vis.mean(), longest_gap, first_seen, last_seen, vis


def extract_metrics(traj):
    """
    Compute metrics for a single h5py.Group representing one trajectory.
    """
    goal   = traj["obs/extra/goal_pos_wrt_base"][:]    # (T,3)
    objpos = traj["obs/extra/obj_pose_wrt_base"][:, :3]# (T,3) not used here
    grasp  = traj["obs/extra/is_grasped"][:]           # (T,)
    succ   = traj["success"][:]                        # (T,)

    head_ext = traj["obs/sensor_param/fetch_head/extrinsic_cv"][:]  # (T,3,4)
    hand_ext = traj["obs/sensor_param/fetch_hand/extrinsic_cv"][:]

    # Distance series (camera center to goal)
    head_xyz = head_ext[..., 3]  # (T,3)
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

    # Final camera poses
    final = -1  # last timestep
    head_xyz_f, head_rpy_f = cam_pose(head_ext[final])
    hand_xyz_f, hand_rpy_f = cam_pose(hand_ext[final])

    # Orientation-based metric:
    # angle between camera's +Z and (camera->goal) direction at final step
    final_head_view_angle = view_angle_deg(goal[final], head_ext[final])
    final_hand_view_angle = view_angle_deg(goal[final], hand_ext[final])

    return dict(
        traj_id = traj.name,
        # final camera pose in base frame
        head_xyz_final = head_xyz_f,
        head_rpy_final_deg = head_rpy_f,
        hand_xyz_final = hand_xyz_f,
        hand_rpy_final_deg = hand_rpy_f,

        # distance stats
        min_head_goal_dist = float(head_dist.min()),
        min_hand_goal_dist = float(hand_dist.min()),
        final_head_goal_dist = float(head_dist[final]),
        final_hand_goal_dist = float(hand_dist[final]),

        # visibility stats
        vis_frac_head = float(head_vis_frac),
        vis_frac_hand = float(hand_vis_frac),
        longest_invis_gap_head = head_long_gap,
        longest_invis_gap_hand = hand_long_gap,
        t_since_view_to_grasp_h = t_since_view_head,
        t_since_view_to_grasp_d = t_since_view_hand,

        # orientation stats (added)
        final_head_goal_view_deg = final_head_view_angle,
        final_hand_goal_view_deg = final_hand_view_angle,

        # outcome
        grasp_success_step = grasp_step,
        overall_success = bool(succ[-1]),
        traj_length = int(len(goal)),
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
            metrics = extract_metrics(traj)
            # Only store if overall success
            if metrics["overall_success"]:
                rows.append(metrics)

df = pd.DataFrame(rows)


print("\n=== FINAL CAMERA POSES ===")
print(df[[
    "traj_id",
    "head_xyz_final",
    "head_rpy_final_deg",
    "hand_xyz_final",
    "hand_rpy_final_deg"
]])

print("\n=== SUMMARY METRICS ===")
cols = [
    "traj_id",
    "vis_frac_head", "vis_frac_hand",
    "longest_invis_gap_head", "longest_invis_gap_hand",
    "t_since_view_to_grasp_h", "t_since_view_to_grasp_d",
    "final_head_goal_dist", "final_hand_goal_dist",
    "min_head_goal_dist", "min_hand_goal_dist",
    "final_head_goal_view_deg", "final_hand_goal_view_deg",
    "overall_success"
]
print(df[cols].to_string(index=False))

# optional CSV
if args.out_csv:
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV to {args.out_csv}")
