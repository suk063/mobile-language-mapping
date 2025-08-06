import torch
import torch.nn as nn
from pytorch3d.ops import ball_query, sample_farthest_points
from lang_mapping.module.transformer import TransformerLayer

class SATransformer(nn.Module):
    """
    Set-abstraction block with point-wise self-attention.
    • FPS (or mask-aware FPS) to choose centroids
    • Ball-query to gather K neighbours
    • Local self-attention → max-pool → output features
    """
    def __init__(self, in_dim, out_dim,
                 radius, nsample, sampling_ratio=0.25,
                 heads=8, dropout=0.1):
        super().__init__()
        self.radius, self.nsample, self.sampling_ratio = radius, nsample, sampling_ratio

        # self.in_proj = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.LayerNorm(out_dim),
        #     nn.ReLU(inplace=True),
        # )
        #  CHG: TransformerLayer uses 'key_padding_mask'
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        self.local_attn = TransformerLayer(out_dim, heads, out_dim * 4, dropout, use_xformers=True)
        self.out_norm   = nn.LayerNorm(out_dim)

    # ------------------------- FPS ------------------------------ #
    @torch.no_grad()
    def _sample(self, xyz, pad_mask):
        """
        Mask-aware farthest-point sampling
        Returns
        -------
        idx        : (B, M)   sampled centroid indices
        keep_mask  : (B, M)   True → valid centroid, False → padded centroid
        """
        B, N, _ = xyz.shape
        if pad_mask is None:                     # no padding in input
            M = max(1, int(round(N * self.sampling_ratio)))
            _, idx = sample_farthest_points(xyz, K=M)
            return idx, None

        # move valid points to the front so FPS sees a packed list
        perm      = torch.argsort(pad_mask.int(), dim=1)
        xyz_front = xyz.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))
        lengths   = (~pad_mask).sum(dim=1)                           # valid counts
        K_each    = torch.clamp((lengths.float() * self.sampling_ratio)
                                .round().long(), min=1)
        K_max     = int(K_each.max())

        _, idx_all = sample_farthest_points(xyz_front,
                                            lengths=lengths, K=K_max)
        arange_mat = torch.arange(K_max, device=xyz.device).expand(B, K_max)
        keep_mask  = arange_mat < K_each.unsqueeze(1)
        idx_local  = torch.where(keep_mask, idx_all,
                                 idx_all[:, :1].expand(-1, K_max))
        idx_global = perm.gather(1, idx_local)                      # back-map
        return idx_global, keep_mask                               # (B,M), (B,M)

    # ------------------------------ forward -------------------------------- #
    def forward(self,
                xyz:   torch.Tensor,         # (B, N, 3)
                feats: torch.Tensor,         # (B, N, Cin)
                pad_mask: torch.Tensor|None = None  # (B, N) bool – True=PAD
                ):
        B, N, _ = xyz.shape
        feats = self.in_proj(feats)
        # 1) choose centroids ------------------------------------------------
        ctr_idx, keep_mask = self._sample(xyz, pad_mask)            # (B,M)
        ctr_xyz   = xyz  [torch.arange(B)[:, None], ctr_idx]        # (B,M,3)

        # 2) ball-query neighbours -----------------------------------------
        neigh = ball_query(ctr_xyz, xyz, K=self.nsample,
                           radius=self.radius, return_nn=False)
        # neigh.idx = -1 where invalid
        neigh_idx = torch.where(
            neigh.idx < 0,
            ctr_idx.unsqueeze(-1).expand(-1, -1, self.nsample),
            neigh.idx,
        )                                                           # (B,M,K)

        batch_idx   = torch.arange(B, device=xyz.device
                           ).view(B, 1, 1).expand_as(neigh_idx)
        neigh_xyz   = xyz  [batch_idx, neigh_idx]                   # (B,M,K,3)
        neigh_feats = feats[batch_idx, neigh_idx]                   # (B,M,K,Cin)

        # 3) build padding mask for neighbours -----------------------------  # NEW
        neigh_invalid = neigh.idx < 0                               # (B,M,K)
        pad_neigh     = neigh_invalid.view(-1, self.nsample)        # (B*M,K)

        # 4) local self-attention ------------------------------------------
        BM, K, C = neigh_feats.size(0) * neigh_feats.size(1), \
                   neigh_feats.size(2), neigh_feats.size(3)

        feats_out = self.local_attn(
            neigh_feats.view(BM, K, C),
            coords_src           = neigh_xyz.view(BM, K, 3),
            key_padding_mask     = pad_neigh                       # NEW
        )                                                           # (BM,K,C)

            # 5) pooling & post-norm -------------------------------------------
        pooled = feats_out.max(dim=1).values.view(B, ctr_idx.shape[1], C)
        pooled = self.out_norm(pooled)

        # 6) propagate centroid-level padding mask -------------------------
        centroid_pad = None
        if keep_mask is not None:
            centroid_pad = ~keep_mask                               # (B,M)
            pooled  = pooled .masked_fill(centroid_pad.unsqueeze(-1), 0.0)
            ctr_xyz = ctr_xyz.masked_fill(centroid_pad.unsqueeze(-1), 1e6)

        return ctr_xyz, pooled, centroid_pad


class GlobalSceneEncoder(nn.Module):
    def __init__(self, in_dim=384, out_dim=384, heads=8, dropout=0.1):
        super().__init__()
        cfg = dict(
            sa1=(1.0,  16, 0.25),
            sa2=(2.0,  16, 0.25),
            sa3=(4.0,  16, 0.25),
            sa4=(8.0, 16, 0.0)
        )
        (r1,k1,p1), (r2,k2,p2), (r3,k3,p3), (r4,k4,p4) = cfg.values()

        self.sa1 = SATransformer(in_dim, out_dim, r1, k1, p1, heads, dropout)
        self.sa2 = SATransformer(out_dim, out_dim, r2, k2, p2, heads, dropout)
        self.sa3 = SATransformer(out_dim, out_dim, r3, k3, p3, heads, dropout)
        self.sa4 = SATransformer(out_dim, out_dim, r4, k4, p4, heads, dropout)

        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.LayerNorm(out_dim))

    def forward(self, pts, pad=None):
        """
        pts : (B, N, 3 + in_dim)
        pad : (B, N) bool – True = PAD
        """
        
        xyz, feat = pts[..., :3], pts[..., 3:]

        xyz, feat, pad = self.sa1(xyz, feat, pad)
        xyz, feat, pad = self.sa2(xyz, feat, pad)
        xyz, feat, pad = self.sa3(xyz, feat, pad)
        xyz, feat, pad = self.sa4(xyz, feat, pad)

        feat = self.proj(feat)                     # (B, ≤100, out_dim)
        return xyz, feat
    