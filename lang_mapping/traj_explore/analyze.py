import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("csv", default="trajectory_metrics.csv",
                help="metrics CSV produced by observe_traj.py")
ap.add_argument("--top", type=int, default=10,
                help="How many top demos to list & plot")
args = ap.parse_args()

csv_path = Path(args.csv)
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------ #
#  1. Compute Visibility and Orientation Stats                       #
# ------------------------------------------------------------------ #
# Average Visibility Fraction
df["avf"] = 0.5 * (df["vis_frac_head"] + df["vis_frac_hand"])

# Visibility-Scarcity Score (VSS)
#   (1 - avf) encourages lower average visibility => higher VSS
#    + fraction of frames that are a long invis gap for head & hand
df["vss"] = (1 - df["avf"]) \
        + df["longest_invis_gap_head"] / df["traj_length"] \
        + df["longest_invis_gap_hand"] / df["traj_length"]

# “Best” final camera angle = smaller angle is better
df["best_final_view_deg"] = df[["final_head_goal_view_deg", 
                                "final_hand_goal_view_deg"]].min(axis=1)
# Simple Orientation Alignment Score (OAS),  in [0,1]
df["oas"] = 1 - df["best_final_view_deg"].clip(0,180) / 180.0

# Combined score example: 
#   combined_score = OAS - 0.5 * VSS
ALPHA = 0.5
df["combined_score"] = df["oas"] - ALPHA * df["vss"]

# Worst final view angle (for mapping-need style metrics)
df["worst_final_view_deg"] = df[["final_head_goal_view_deg",
                                "final_hand_goal_view_deg"]].max(axis=1)
# Mapping Need Score (MNS): higher => more “out of view” or poorly aligned
#   MNS = VSS + (worst_final_view_deg / 180)
df["mns"] = df["vss"] + (df["worst_final_view_deg"] / 180.0)

# ------------------------------------------------------------------ #
#  2. Create an Aggregate Ranking & Select Top 10                    #
# ------------------------------------------------------------------ #
# We want to rank by 3 metrics: vss, mns, combined_score
# Higher is more "interesting" or "challenging" for all three.
# Pandas 'rank' defaults to ascending=True, so we set ascending=False
# so that a higher metric => rank=1 means best/hardest.

# Only rank among successful trajectories
df_success = df[df["overall_success"] == True].copy()

df_success["rank_vss"] = df_success["vss"].rank(method="min", ascending=False)
df_success["rank_mns"] = df_success["mns"].rank(method="min", ascending=False)
df_success["rank_combined"] = df_success["combined_score"].rank(method="min", ascending=False)

# Summation of ranks => overall "score_agg" (lower = better across all)
df_success["score_agg"] = (df_success["rank_vss"] 
                        + df_success["rank_mns"] 
                        + df_success["rank_combined"])

# Sort by the aggregator, ascending => best overall is top of the list
df_success.sort_values("score_agg", inplace=True)

# Keep only the top N
df_top = df_success.head(args.top).copy()

print(f"\n=== Top {args.top} Demos by combined ranking of (VSS, MNS, CombinedScore) ===")
cols_print = ["traj_id","score_agg","vss","mns","combined_score",
                "rank_vss","rank_mns","rank_combined","best_final_view_deg"]
print(df_top[cols_print].to_string(index=False))

# ------------------------------------------------------------------ #
#  3. Plot directory                                                #
# ------------------------------------------------------------------ #
plot_dir = csv_path.with_suffix("") / "plots_top10only"
plot_dir.mkdir(parents=True, exist_ok=True)

# Note: We'll plot ONLY df_top in all subsequent plots.

# ------------------------------------------------------------------ #
#  4. Histogram of VSS (only top demos)                              #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.hist(df_top["vss"], bins=5, edgecolor='black')
plt.xlabel("Visibility-Scarcity Score (VSS)")
plt.ylabel("Count (Top 10 only)")
plt.title("Top 10 Demos: VSS Distribution")
plt.tight_layout()
plt.savefig(plot_dir / "hist_vss_top10.png")
plt.close()

# ------------------------------------------------------------------ #
#  5. Scatter: AVF vs. longest invis gap (only top demos)           #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5.5,4.5))
sc = plt.scatter(df_top["avf"], 
                df_top["longest_invis_gap_head"] / df_top["traj_length"],
                c=df_top["vss"], cmap="viridis", s=80, edgecolor='black')
cb = plt.colorbar(sc)
cb.set_label("VSS")
plt.xlabel("Average Visibility Fraction (AVF)")
plt.ylabel("Longest invisible gap (fraction of episode)")
plt.title("Top 10 Demos")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_avf_vs_gap_colVSS_top10.png")
plt.close()

# ------------------------------------------------------------------ #
#  6. Bar chart: longest invis gap (top 10)                          #
# ------------------------------------------------------------------ #
# We'll just reuse df_top sorted by the gap
df_top_gap = df_top.sort_values("longest_invis_gap_head", ascending=False)
plt.figure(figsize=(6,4))
plt.bar(df_top_gap["traj_id"], df_top_gap["longest_invis_gap_head"], color='orange')
plt.xticks(rotation=45, ha="right")
plt.ylabel("Longest invisible gap (frames)")
plt.title("Head-cam longest invisibility (Top 10 only)")
plt.tight_layout()
plt.savefig(plot_dir / "bar_longest_gap_top10.png")
plt.close()

# ------------------------------------------------------------------ #
#  7. Histograms for vis_frac_head and vis_frac_hand (top 10 only)   #
# ------------------------------------------------------------------ #
plt.figure(figsize=(6,4))
plt.hist(df_top["vis_frac_head"], bins=5, alpha=0.6, label="Head-cam Visibility", edgecolor='black')
plt.hist(df_top["vis_frac_hand"], bins=5, alpha=0.6, label="Hand-cam Visibility", edgecolor='black')
plt.xlabel("Visibility Fraction")
plt.ylabel("Count (Top 10 only)")
plt.title("Top 10: Distribution of Visibility Fractions")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "hist_vis_frac_head_and_hand_top10.png")
plt.close()

# ------------------------------------------------------------------ #
#  8. Scatter: vis_frac_head vs. vis_frac_hand (top 10 only)         #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.scatter(df_top["vis_frac_head"], 
        df_top["vis_frac_hand"], 
        alpha=0.8, c=df_top["vss"], cmap="viridis", s=80, edgecolor='black')
cb = plt.colorbar()
cb.set_label("VSS")
plt.xlabel("Head-cam Visibility Fraction")
plt.ylabel("Hand-cam Visibility Fraction")
plt.title("Top 10: Head vs. Hand Visibility")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_visfrac_head_vs_hand_top10.png")
plt.close()

# ------------------------------------------------------------------ #
#  9. Scatter: VSS vs. worst_final_view_deg, color by MNS           #
# ------------------------------------------------------------------ #
plt.figure(figsize=(6,5))
sc = plt.scatter(
df_top["vss"],
df_top["worst_final_view_deg"],
c=df_top["mns"],
cmap="viridis",
s=80,
edgecolor='black'
)
cb = plt.colorbar(sc)
cb.set_label("Mapping Need Score (MNS)")
plt.xlabel("Visibility-Scarcity Score (VSS)")
plt.ylabel("Worst Final Camera Angle (deg)")
plt.title("Top 10 Only: VSS vs. Worst Angle")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_vss_vs_worstAngle_colMNS_top10.png")
plt.close()

# ------------------------------------------------------------------ #
# 10. Histogram of MNS (top 10 only)                                 #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.hist(df_top["mns"], bins=5, color="orange", alpha=0.8, edgecolor='black')
plt.xlabel("Mapping Need Score (MNS)")
plt.ylabel("Count (Top 10 only)")
plt.title("Top 10: Distribution of MNS")
plt.tight_layout()
plt.savefig(plot_dir / "hist_mns_top10.png")
plt.close()

# ------------------------------------------------------------------ #
# 11. Histograms of final orientation angles (top 10 only)           #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.hist(df_top["final_head_goal_view_deg"], bins=5, alpha=0.6, label="Head final angle", edgecolor='black')
plt.hist(df_top["final_hand_goal_view_deg"], bins=5, alpha=0.6, label="Hand final angle", edgecolor='black')
plt.xlabel("Angle to goal (degrees)")
plt.ylabel("Count (Top 10 only)")
plt.title("Top 10: Distribution of final camera‑goal angles")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "hist_final_view_angle_top10.png")
plt.close()

# ------------------------------------------------------------------ #
# 12. Scatter: VSS vs. best_final_view_deg (color by OAS)           #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
sc = plt.scatter(
df_top["vss"],
df_top["best_final_view_deg"],
c=df_top["oas"],  # orientation alignment
cmap="viridis",
s=80,
edgecolor='black'
)
cb = plt.colorbar(sc)
cb.set_label("Orientation Alignment Score (OAS)")
plt.xlabel("Visibility-Scarcity Score (VSS)")
plt.ylabel("Best Final Camera Angle (deg)")
plt.title("Top 10: VSS vs. Orientation")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_vss_vs_orientation_top10.png")
plt.close()

print(f"\nPlots saved in {plot_dir.resolve()}")