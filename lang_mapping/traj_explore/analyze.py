import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("csv", default="trajectory_metrics.csv",
                help="metrics CSV produced by observe_traj.py")
ap.add_argument("--top", type=int, default=10,
                help="how many top demos to list")
args = ap.parse_args()

csv_path = Path(args.csv)
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------ #
#  1.  Visibility metrics                                            #
# ------------------------------------------------------------------ #
df["avf"] = 0.5 * (df["vis_frac_head"] + df["vis_frac_hand"])
df["vss"] = (1 - df["avf"]) \
          + df["longest_invis_gap_head"] / df["traj_length"] \
          + df["longest_invis_gap_hand"] / df["traj_length"]

rank = (df[df["overall_success"] == True]
        .sort_values("vss", ascending=False))

print(f"\n=== Top {args.top} mapping-worthy demos (by VSS) ===")
cols = ["traj_id", "avf", "vss",
        "vis_frac_head", "vis_frac_hand",
        "longest_invis_gap_head", "longest_invis_gap_hand"]
print(rank.head(args.top)[cols].to_string(index=False))

# ------------------------------------------------------------------ #
#  2.  Plot directory                                                #
# ------------------------------------------------------------------ #
plot_dir = csv_path.with_suffix("") / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  3.  Histogram of visibility scarcity                              #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.hist(df["vss"], bins=20)
plt.xlabel("Visibility-Scarcity Score (VSS)")
plt.ylabel("Count")
plt.title("How hard are the demos? (higher = less visibility)")
plt.tight_layout()
plt.savefig(plot_dir / "hist_vss.png")
plt.close()

# ------------------------------------------------------------------ #
#  4.  Scatter: AVF vs. longest invis gap                            #
#      – point colour encodes VSS                                    #
# ------------------------------------------------------------------ #
plt.figure(figsize=(5.5,4.5))
sc = plt.scatter(df["avf"], df["longest_invis_gap_head"] / df["traj_length"],
                 c=df["vss"], cmap="viridis", alpha=0.8)
cb = plt.colorbar(sc); cb.set_label("VSS")
plt.xlabel("Average Visibility Fraction (AVF)")
plt.ylabel("Longest invisible gap  (fraction of episode)")
plt.title("Where mapping is needed: bottom-right, dark colours")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_avf_vs_gap_colVSS.png")
plt.close()

# ------------------------------------------------------------------ #
#  5.  Bar chart: longest invis gap (top-N)                          #
# ------------------------------------------------------------------ #
topN = rank.head(args.top)
plt.figure(figsize=(6,4))
plt.bar(topN["traj_id"], topN["longest_invis_gap_head"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Longest invisible gap (frames)")
plt.title(f"Head-cam longest invisibility (top {args.top})")
plt.tight_layout()
plt.savefig(plot_dir / "bar_longest_gap_top.png")
plt.close()

# ------------------------------------------------------------------ #
#  6.  ## NEW CODE ## Histograms for vis_frac_head and vis_frac_hand
# ------------------------------------------------------------------ #
plt.figure(figsize=(6,4))
plt.hist(df["vis_frac_head"], bins=20, alpha=0.6, label="Head-cam Visibility")
plt.hist(df["vis_frac_hand"], bins=20, alpha=0.6, label="Hand-cam Visibility")
plt.xlabel("Visibility Fraction")
plt.ylabel("Count")
plt.title("Distribution of Visibility Fractions (Head vs. Hand)")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "hist_vis_frac_head_and_hand.png")
plt.close()

# ------------------------------------------------------------------ #
#  7.  ## NEW CODE ## Scatter comparing vis_frac_head vs. vis_frac_hand
# ------------------------------------------------------------------ #
plt.figure(figsize=(5,4))
plt.scatter(df["vis_frac_head"], df["vis_frac_hand"], alpha=0.7, c=df["vss"], cmap="viridis")
cb = plt.colorbar(); cb.set_label("VSS")
plt.xlabel("Head-cam Visibility Fraction")
plt.ylabel("Hand-cam Visibility Fraction")
plt.title("Head vs. Hand Visibility")
plt.tight_layout()
plt.savefig(plot_dir / "scatter_visfrac_head_vs_hand.png")
plt.close()

print(f"\nPlots saved in {plot_dir.resolve()}")