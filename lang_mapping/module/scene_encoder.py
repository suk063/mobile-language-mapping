import torch
import torch.nn as nn
from torch_geometric.nn.pool import fps, radius
from torch_geometric.nn import MLP, PointTransformerConv
from torch_geometric.utils import to_dense_batch


from lang_mapping.module.transformer import TransformerLayer


class PointTransformerBlock(nn.Module):
    """
    Point Transformer block with Set Abstraction (FPS, radius grouping, and PointTransformerConv).
    """

    def __init__(self, in_dim, out_dim, ratio, radius_val, nsample, heads=8, dropout=0.1):
        super().__init__()
        self.ratio = ratio
        self.r = radius_val
        self.k = nsample

        # For bipartite PointTransformerConv, in_channels should be a tuple if dims differ.
        # Here, they are the same after the initial projection.
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        pos_nn = MLP([3, out_dim, out_dim], plain_last=False, batch_norm=False)
        attn_nn = MLP([out_dim, out_dim], plain_last=False, batch_norm=False)

        self.conv = PointTransformerConv(
            in_channels=out_dim, out_channels=out_dim,
            pos_nn=pos_nn, attn_nn=attn_nn
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x, pos, batch):
        """
        Forward pass.
        :param x: (N, C_in) features
        :param pos: (N, 3) coordinates
        :param batch: (N,) batch indices
        :return: (M, C_out) features, (M, 3) coordinates, (M,) batch indices of downsampled points
        """
        x = self.in_proj(x)
        
        # Furthest Point Sampling
        if self.ratio < 1.0:
            idx = fps(pos, batch, ratio=self.ratio)
        else: # No downsampling
            idx = torch.arange(pos.shape[0], device=pos.device)

        # Radius grouping
        row, col = radius(x=pos, y=pos[idx], r=self.r, batch_x=batch, batch_y=batch[idx], max_num_neighbors=self.k)
        edge_index = torch.stack([col, row], dim=0)

        # Bipartite convolution
        x_out = self.conv(x=(x, x[idx]), pos=(pos, pos[idx]), edge_index=edge_index)
        x_out = self.out_norm(x_out)

        return x_out, pos[idx], batch[idx]


class GlobalSceneEncoder(nn.Module):
    def __init__(self, in_dim=384, out_dim=384, heads=8, dropout=0.1):
        super().__init__()
        cfg = dict(
            sa1=(in_dim, out_dim, 1.0, 16, 0.25),
            sa2=(out_dim, out_dim, 2.0, 16, 0.25),
            sa3=(out_dim, out_dim, 4.0, 16, 0.25),
            sa4=(out_dim, out_dim, 8.0, 16, 0.25)
        )
        (id1,od1,r1,k1,p1), (id2,od2,r2,k2,p2), (id3,od3,r3,k3,p3), (id4,od4,r4,k4,p4) = cfg.values()

        self.sa1 = PointTransformerBlock(id1, od1, p1, r1, k1, heads, dropout)
        self.sa2 = PointTransformerBlock(id2, od2, p2, r2, k2, heads, dropout)
        self.sa3 = PointTransformerBlock(id3, od3, p3, r3, k3, heads, dropout)
        self.sa4 = PointTransformerBlock(id4, od4, p4, r4, k4, heads, dropout)

        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.LayerNorm(out_dim))

    def forward(self, pts, pad=None):
        """
        pts : (B, N, 3 + in_dim)
        pad : (B, N) bool – True = PAD
        """
        B, N, _ = pts.shape
        xyz, feat = pts[..., :3].contiguous(), pts[..., 3:].contiguous()

        # Convert to PyG batch format
        if pad is not None:
            mask = ~pad
            batch_vec_full = torch.arange(B, device=pts.device).unsqueeze(1).expand(B, N)
            pos, x = xyz[mask], feat[mask]
            batch = batch_vec_full[mask]
        else:
            pos, x = xyz.view(-1, 3), feat.view(-1, feat.shape[-1])
            batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        x, pos, batch = self.sa4(x, pos, batch)

        feat = self.proj(x)
        
        # Convert back to dense tensor format (with padding)
        xyz_out, _ = to_dense_batch(pos, batch, fill_value=1e6) # Pad with large value
        feat_out, pad_mask = to_dense_batch(feat, batch, fill_value=0.0)

        return xyz_out, feat_out, ~pad_mask
    