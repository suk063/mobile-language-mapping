"""gridnet_visualization.py – Open3D visualizer for GridNet
Shows voxel centres coloured either by:
* **PCA** of **decoded** features (after `ImplicitDecoder`).
* **CLIP‑text similarity** (unchanged).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from lang_mapping.grid_net import GridNet
from lang_mapping.module import ImplicitDecoder
import open_clip

# -----------------------------------------------------------------------------
# Default GridNet cfg ----------------------------------------------------------
# -----------------------------------------------------------------------------

def default_grid_cfg() -> Dict:
    return {
        "name": "grid_net",
        "spatial_dim": 3,
        "grid": {
            "type": "regular",
            "feature_dim": 60,
            "init_stddev": 0.2,
            "bound": [[-2.6, 4.6], [-8.1, 4.7], [0.0, 3.1]],
            "base_cell_size": 0.4,
            "per_level_scale": 2.0,
            "n_levels": 2,
            "n_scenes": 122,
            "second_order_grid_sample": False,
        },
    }

# -----------------------------------------------------------------------------
# Utility helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _flatten_grid(feat: torch.Tensor) -> torch.Tensor:
    """(C,Z,Y,X) → (N,C)"""
    return feat.permute(1, 2, 3, 0).reshape(-1, feat.shape[0])


def extract_grid_centers_features(
    gridnet: GridNet,
    *,
    scene_id: int,
    level: int,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return voxel centres & raw features for a *full* grid (thresholded)."""
    device = gridnet.bound.device

    grid = gridnet.features[scene_id][level]
    feat = grid.feature.squeeze(0).to(device)  # (C,Z,Y,X)

    # Build voxel index tensor on same device
    C, Z, Y, X = feat.shape
    idx = (
        torch.stack(
            torch.meshgrid(
                torch.arange(Z, device=device),
                torch.arange(Y, device=device),
                torch.arange(X, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .float()
    )

    cell_size = grid.cell_size
    bound_min = gridnet.bound[:, 0]
    centres = ((idx + 0.5) * cell_size + bound_min).cpu().numpy()  # (N,3)

    feats = _flatten_grid(feat).cpu()
    mask = torch.linalg.vector_norm(feats, dim=1) > threshold

    centres_np = centres
    feats_np = feats.detach().numpy()
    mask_np = mask.numpy()

    return centres_np[mask_np], feats_np[mask_np]

# -----------------------------------------------------------------------------
# Changed‑centres helper -------------------------------------------------------
# -----------------------------------------------------------------------------

def load_changed_centers_npz(root_dir: Path, level: int) -> Dict[int, np.ndarray]:
    file = root_dir / f"level{level}_centers.npz"
    if not file.exists():
        raise FileNotFoundError(file)
    data = np.load(file, allow_pickle=True)
    return {int(k): data[k] for k in data.files}

# -----------------------------------------------------------------------------
# Colour mapping --------------------------------------------------------------
# -----------------------------------------------------------------------------

def colour_pca(features: np.ndarray) -> np.ndarray:
    """RGB via 3‑component PCA (features assumed already decoded)."""
    norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    rgb = MinMaxScaler().fit_transform(PCA(n_components=3).fit_transform(norm))
    return rgb  # (N,3)


def colour_similarity(
    features: np.ndarray,
    *,
    implicit_dec: ImplicitDecoder,
    clip_model,
    tokenizer,
    query: str | list[str],
) -> np.ndarray:
    """
    - query      : 문자열 또는 문자열 리스트
    - returns    : (N,3) RGB in [0,1]
    """
    # --------------------------------------------------------------
    # 1. prep text tokens  (append "" for redundant baseline)
    # --------------------------------------------------------------
    if isinstance(query, str):
        text_input = [query]
    else:
        text_input = list(query)
    text_input += [""]                       # <-- baseline token

    device = next(clip_model.parameters()).device
    text_tok  = tokenizer(text_input).to(device)

    # --------------------------------------------------------------
    # 2. embed + L2-normalise + baseline subtraction
    # --------------------------------------------------------------
    with torch.no_grad():
        text_emb = F.normalize(clip_model.encode_text(text_tok), dim=-1, p=2)

    redundant_emb  = text_emb[-1:, :]        # (“”)
    text_emb       = text_emb[:-1, :] - redundant_emb
    text_emb = F.normalize(text_emb)

    # --------------------------------------------------------------
    # 3. decode voxel features → similarity
    # --------------------------------------------------------------
    pts_feat = torch.as_tensor(features, device=device)
    with torch.no_grad():
        decoded = F.normalize(implicit_dec(pts_feat), dim=-1, p=2)

    sim  = (decoded * text_emb).sum(-1).cpu().numpy()    # cosine similarity
    sim  = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

    return cm.get_cmap("viridis")(sim)[:, :3]
# -----------------------------------------------------------------------------
# Open3D viewer ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def show_o3d(points: np.ndarray,
             colors: np.ndarray,
             title: str,
             point_size: float = 5.0):      # ← default is noticeably larger
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = point_size          # ★ the only new line that matters

    vis.run()
    vis.destroy_window()
# -----------------------------------------------------------------------------
# Model loader ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_models(ckpt_dir: Path, device: torch.device):
    ckpt_dir = ckpt_dir.expanduser().resolve()

    gridnet = GridNet(cfg=default_grid_cfg(), device=device)
    gridnet.load_state_dict(torch.load(ckpt_dir / "latest_static_map.pt", map_location=device))
    gridnet.eval()

    fdim = default_grid_cfg()["grid"]["feature_dim"] * default_grid_cfg()["grid"]["n_levels"]
    implicit_dec = ImplicitDecoder(voxel_feature_dim=fdim, hidden_dim=512, output_dim=768).to(device)
    implicit_dec.load_state_dict(torch.load(ckpt_dir / "latest_decoder.pt", map_location=device))
    implicit_dec.eval()

    clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-L-14", pretrained="merged2b_s4b_b131k")
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("EVA02-L-14")

    return gridnet, implicit_dec, clip_model, tokenizer

# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("GridNet Open3D visualizer")
    parser.add_argument("--ckpt_dir", type=Path, default="pre-trained")
    parser.add_argument("--mode", choices=["pca", "sim"], default="sim")
    parser.add_argument("--query", type=str, default="apple")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--changed_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gridnet, implicit_dec, clip_model, tokenizer = build_models(args.ckpt_dir, device)

    # ------------------------------------------------------------------
    # Gather voxel centres + raw voxel features
    # ------------------------------------------------------------------
    if args.changed_only:
        cntrs_dict = load_changed_centers_npz(args.ckpt_dir, args.level)
        pts_all, feats_all = [], []
        for s_id, pts in cntrs_dict.items():
            sid_tensor = torch.full((pts.shape[0], 1), s_id, device=device, dtype=torch.long)
            with torch.no_grad():
                feats = gridnet.query_feature(torch.as_tensor(pts, device=device), sid_tensor).cpu().numpy()
            pts_all.append(pts)
            feats_all.append(feats)
        points = np.concatenate(pts_all, axis=0)
        raw_features = np.concatenate(feats_all, axis=0)
    else:
        points, raw_features = extract_grid_centers_features(gridnet, scene_id=args.scene, level=args.level)

    # ------------------------------------------------------------------
    # Colour mapping
    # ------------------------------------------------------------------
    if args.mode == "pca":
        # Decode FIRST, then PCA → RGB
        with torch.no_grad():
            decoded = implicit_dec(torch.as_tensor(raw_features, device=device)).cpu().numpy()
        colours = colour_pca(decoded)
        title = "GridNet PCA (changed centres)" if args.changed_only else "GridNet PCA"
        show_o3d(points, colours, title, point_size=args.point_size)
    else:
        colours = colour_similarity(
            raw_features,
            implicit_dec=implicit_dec,
            clip_model=clip_model,
            tokenizer=tokenizer,
            query=args.query,
        )
        title = f'Similarity → "{args.query}"' + (" (changed centres)" if args.changed_only else "")
        show_o3d(points, colours, title, point_size=args.point_size)


if __name__ == "__main__":
    main()
