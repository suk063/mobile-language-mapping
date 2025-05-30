import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from pytorch3d.ops import ball_query, sample_farthest_points

from ..module.transformer import TransformerLayer
from ..utils import rotary_pe_3d

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class SetAbstraction(nn.Module):
    """
    FPS → Ball Query → point-wise MLP → masked Max-Pool
    """
    def __init__(self, in_dim, out_dim, radius, nsample, sampling_ratio=0.25):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.sampling_ratio = sampling_ratio

        # (Cin+xyz) → Cout/2 → Cout
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 3, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    # --------------------------------------------------------------------- #
    # FPS with lengths ----------------------------------------------------- #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _sample(self, xyz: torch.Tensor, pad_mask: torch.BoolTensor | None):
        B, N, _ = xyz.shape
        if pad_mask is None:
            M = max(1, int(round(N * self.sampling_ratio)))
            _, idx = sample_farthest_points(xyz, K=M)
            return idx, None

        # (1) pad 제거를 위해 재배열
        perm      = torch.argsort(pad_mask.int(), dim=1)         # (B, N)
        xyz_front = xyz.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))

        # (2) 클라우드별 실제 길이
        lengths = (~pad_mask).sum(dim=1)                         # (B,)

        # (3) 클라우드마다 목표 샘플 수 K_i 계산
        K_each = torch.clamp(                                    # (B,)
            (lengths.float() * self.sampling_ratio).round().long(),
            min=1
        )
        K_max = int(K_each.max())                                # 전체에서 가장 큰 K

        # (4) 한번에 FPS → (B, K_max)
        _, idx_all = sample_farthest_points(
            xyz_front, lengths=lengths, K=K_max
        )

        # (5) 필요 없는 칸은 첫 인덱스로 채워 두기(어차피 이후 ball-query에서 PAD)
        arange_mat = torch.arange(K_max, device=xyz.device).expand(B, K_max)
        keep_mask  = arange_mat < K_each.unsqueeze(1)            # (B, K_max)
        idx_local  = torch.where(
            keep_mask, idx_all, idx_all[:, :1].expand(-1, K_max)
        )                                                        # (B, K_max)

        # (6) 원본 인덱스로 복원
        idx_global = perm.gather(1, idx_local)
        return idx_global, keep_mask

    # --------------------------------------------------------------------- #
    # forward -------------------------------------------------------------- #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        xyz: torch.Tensor,           # (B, N, 3)
        feats: torch.Tensor,         # (B, N, Cin)
        pad_mask: torch.BoolTensor | None = None  # (B, N) – True = PAD
    ):
        B, N, _ = xyz.shape

        # 1) FPS centroids (lengths aware)
        centroids_idx, keep_mask = self._sample(xyz, pad_mask)            # (B, M)
        centroids_xyz = xyz[torch.arange(B)[:, None], centroids_idx]

        # 2) Ball-query neighbours
        neigh_info = ball_query(
            centroids_xyz, xyz, K=self.nsample,
            radius=self.radius, return_nn=False
        )
        neigh_idx = neigh_info.idx                             # (B, M, K)
        pad_from_ball = neigh_idx < 0
        neigh_idx = torch.where(
            pad_from_ball,
            centroids_idx.unsqueeze(-1).expand(-1, -1, self.nsample),
            neigh_idx,
        )

        # 3) gather
        batch_idx  = torch.arange(B, device=xyz.device).view(B, 1, 1).expand_as(neigh_idx)
        neigh_xyz   = xyz[batch_idx, neigh_idx]                # (B, M, K, 3)
        neigh_feats = feats[batch_idx, neigh_idx]              # (B, M, K, Cin)

        # 4) point-wise MLP
        local_f = self.mlp(
            torch.cat([neigh_feats, neigh_xyz - centroids_xyz.unsqueeze(2)], dim=-1)
        )                                                      # (B, M, K, Cout)

        pooled_f = local_f.max(dim=2).values          # (B, M, Cout)

        centroid_pad = None
        if keep_mask is not None:
            centroid_pad = ~keep_mask                 # (B,M)
            pooled_f = pooled_f.masked_fill(centroid_pad.unsqueeze(-1), 0.0)
            centroids_xyz = centroids_xyz.masked_fill(centroid_pad.unsqueeze(-1), 1e6)

        return centroids_xyz, pooled_f, centroid_pad
    
class HierarchicalSceneEncoder(nn.Module):
    def __init__(self,
                 in_dim:  int = 768,
                 hid_dim: int = 1024,
                 out_dim: int = 768):
        super().__init__()

        # (radius, nsample, sampling_ratio)
        cfg = dict(
            sa1 = (1.0,  16, 0.25),  
            sa2 = (2.0,  32, 0.25),  
            sa3 = (4.0,  64, 0.25),
            sa4 = (100.0, 10000, 0.0)  
        )
        r1,k1,p1 = cfg["sa1"]; r2,k2,p2 = cfg["sa2"]
        r3,k3,p3 = cfg["sa3"]; r4,k4,p4 = cfg["sa4"]; 


        self.sa1 = SetAbstraction(in_dim,  hid_dim, r1, k1, p1)
        self.sa2 = SetAbstraction(hid_dim, hid_dim, r2, k2, p2)
        self.sa3 = SetAbstraction(hid_dim, hid_dim, r3, k3, p3)
        self.sa4 = SetAbstraction(hid_dim, hid_dim, r4, k4, p4)

        self.proj = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    # ------------------------------------------------------------------ #
    def forward(self,
                pts: torch.Tensor,                 # (B, N, in_dim+3)
                pad: torch.BoolTensor | None = None):
        xyz, feat = pts[..., :3], pts[..., 3:]
        
        
        xyz, feat, pad = self.sa1(xyz, feat, pad)
        xyz, feat, pad = self.sa2(xyz, feat, pad)
        xyz, feat, pad = self.sa3(xyz, feat, pad)
        xyz, feat, pad = self.sa4(xyz, feat, pad)
   
        feat = self.proj(feat)                      # (B, ≤100, out_dim)
        return xyz, feat

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

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        #  CHG: TransformerLayer already supports src_padding_mask
        self.local_attn = TransformerLayer(out_dim, heads, out_dim * 4, dropout)
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

        # 3) feature projection --------------------------------------------
        neigh_feats = self.in_proj(neigh_feats)                    # (B,M,K,Cout)

        # 4) build padding mask for neighbours -----------------------------  # NEW
        neigh_invalid = neigh.idx < 0                               # (B,M,K)
        pad_neigh     = neigh_invalid.view(-1, self.nsample)        # (B*M,K)

        # 5) local self-attention ------------------------------------------
        BM, K, C = neigh_feats.size(0) * neigh_feats.size(1), \
                   neigh_feats.size(2), neigh_feats.size(3)

        feats_out = self.local_attn(
            neigh_feats.view(BM, K, C),
            coords_src           = neigh_xyz.view(BM, K, 3),
            src_padding_mask     = pad_neigh                       # NEW
        )                                                           # (BM,K,C)

        # 6) pooling & post-norm -------------------------------------------
        pooled = feats_out.max(dim=1).values.view(B, ctr_idx.shape[1], C)
        pooled = self.out_norm(pooled)

        # 7) propagate centroid-level padding mask -------------------------
        centroid_pad = None
        if keep_mask is not None:
            centroid_pad = ~keep_mask                               # (B,M)
            pooled  = pooled .masked_fill(centroid_pad.unsqueeze(-1), 0.0)
            ctr_xyz = ctr_xyz.masked_fill(centroid_pad.unsqueeze(-1), 1e6)

        return ctr_xyz, pooled, centroid_pad


class HierarchicalSceneTransformer(nn.Module):
    def __init__(self, in_dim=768, hid_dim=768, out_dim=768,
                 heads=8, dropout=0.1):
        super().__init__()
        cfg = dict(
            sa1=(1.0,  16, 0.25),
            sa2=(2.0,  16, 0.25),
            sa3=(4.0,  16, 0.25),
            sa4=(8.0, 16, 0.0)
        )
        (r1,k1,p1), (r2,k2,p2), (r3,k3,p3), (r4,k4,p4) = cfg.values()

        self.sa1 = SATransformer(in_dim,  hid_dim, r1, k1, p1, heads, dropout)
        self.sa2 = SATransformer(hid_dim, hid_dim, r2, k2, p2, heads, dropout)
        self.sa3 = SATransformer(hid_dim, hid_dim, r3, k3, p3, heads, dropout)
        self.sa4 = SATransformer(hid_dim, hid_dim, r4, k4, p4, heads, dropout)

        self.proj = nn.Sequential(nn.Linear(hid_dim, out_dim),
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
    
    
# -------------------------------------------------------------
# helper: split/merge heads
# -------------------------------------------------------------
def _split_heads(x: torch.Tensor, h: int) -> torch.Tensor:  # (B,L,C) -> (B,h,L,C//h)
    B, L, C = x.shape
    return x.view(B, L, h, C // h).transpose(1, 2)

def _merge_heads(x: torch.Tensor) -> torch.Tensor:          # (B,h,L,d) -> (B,L,h*d)
    B, h, L, d = x.shape
    return x.transpose(1, 2).reshape(B, L, h * d)

# -------------------------------------------------------------
# single-direction cross-attention with RoPE
# -------------------------------------------------------------
class CrossAttnRoPE(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.h = heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out    = nn.Linear(dim, dim, bias=False)
        self.scale  = (dim // heads) ** -0.5

    def forward(
        self,
        Q_in: torch.Tensor,     # (B,L_q,C)
        K_in: torch.Tensor,     # (B,L_k,C)
        V_in: torch.Tensor,     # (B,L_k,C)
        coords_q: torch.Tensor, # (B,L_q,3)
        coords_k: torch.Tensor, # (B,L_k,3)
    ) -> torch.Tensor:

        q = _split_heads(self.q_proj(Q_in), self.h)  # (B,h,Lq,d)
        k = _split_heads(self.k_proj(K_in), self.h)  # (B,h,Lk,d)
        v = _split_heads(self.v_proj(V_in), self.h)

        # --- RoPE ------------------------------------------------------
        q = rotary_pe_3d(q, coords_q)                                   # (B,h,Lq,d)
        k = rotary_pe_3d(k, coords_k)

        # --- scaled dot-product --------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B,h,Lq,Lk)
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                            # (B,h,Lq,d)

        out  = self.out(_merge_heads(out))                         # (B,Lq,C)
        return out

class BiCrossAttnLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn_L = CrossAttnRoPE(dim, heads)     # Q=L, K/V=G
        self.attn_G = CrossAttnRoPE(dim, heads)     # Q=G, K/V=L
        self.gate_L = nn.Parameter(torch.full((1, 1, dim), -5.0))
        self.gate_G = nn.Parameter(torch.full((1, 1, dim), -5.0))

    def forward(self, L, G, coords_L, coords_G):
        L = L + torch.sigmoid(self.gate_L) * self.attn_L(L, G, G, coords_L, coords_G)
        G = G + torch.sigmoid(self.gate_G) * self.attn_G(G, L, L, coords_G, coords_L)
        return L, G

# -------------------------------------------------------------
# bidirectional wrapper
# -------------------------------------------------------------
class BiCrossAttnRoPE(nn.Module):
    """
    L ↔ G bidirectional gated cross-attention with RoPE.
      • L_in : (B,S_L,C)   local / visual
      • G_in : (B,S_G,C)   global
    """
    def __init__(self, dim: int, heads: int = 8, num_layers: int = 2):
        super().__init__()

        self.layers = nn.ModuleList([BiCrossAttnLayer(dim, heads) for _ in range(num_layers)])
        self.apply(init_weights_kaiming)

    def forward(self, L, G, coords_L, coords_G):
        for layer in self.layers:
            L, G = layer(L, G, coords_L, coords_G)
        return L, G
    

# ─────────────────────────────────────────────
#   Local self-attention based feature fusion
# ─────────────────────────────────────────────
class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        radius: float = 0.2,
        k: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.radius, self.k = radius, k
        self.attn = TransformerLayer(
            d_model=dim,
            n_heads=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
        )

    # ----------------------------------------------------------
    # Find neighbor indices within <radius>; pad with query itself
    # ----------------------------------------------------------
    def _neigh_indices(
        self,
        q_xyz: torch.Tensor,           # (B, N, 3)  – query coordinates
        kv_xyz: torch.Tensor,          # (B, L, 3)  – scene coordinates
        kv_pad: Optional[torch.Tensor] # (B, L) bool – True → padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        idx     : (B, N, k) long  – neighbor indices (query-padded)
        invalid : (B, N, k) bool  – True → padding slot
        """
        dist = torch.cdist(q_xyz, kv_xyz)                      # (B, N, L)
        if kv_pad is not None:
            dist = dist.masked_fill(kv_pad[:, None, :], float("inf"))

        # keep only points ≤ radius
        dist = torch.where(dist <= self.radius, dist, float("inf"))
        k = self.k

        # 1) take top-k closest (up to k). If fewer, remaining are arbitrary for now.
        _, idx_topk = dist.topk(k, largest=False, dim=-1)      # (B, N, k)

        # 2) mark invalid (padding) slots
        gather_dist = dist.gather(-1, idx_topk)                # (B, N, k)
        invalid = gather_dist.isinf()                          # True → padding slot

        # 3) overwrite padding slots with dummy index 0 (will be replaced by query itself)
        query_idx = torch.zeros_like(idx_topk)                 # value 0 is arbitrary
        idx = torch.where(invalid, query_idx, idx_topk)        # (B, N, k)

        return idx, invalid

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        q_xyz:   torch.Tensor,                # (B, N, 3)
        q_feat:  torch.Tensor,                # (B, N, C)
        kv_xyz:  torch.Tensor,                # (B, L, 3)
        kv_feat: torch.Tensor,                # (B, L, C)
        kv_pad:  Optional[torch.Tensor] = None  # (B, L) bool
    ) -> torch.Tensor:
        B, N, C = q_feat.shape
        idx, invalid = self._neigh_indices(q_xyz, kv_xyz, kv_pad)  # (B, N, k)

        # gather neighbor coordinates / features
        batch = torch.arange(B, device=q_feat.device).view(B, 1, 1)
        neigh_xyz  = kv_xyz[batch.expand_as(idx), idx]             # (B, N, k, 3)
        neigh_feat = kv_feat[batch.expand_as(idx), idx]            # (B, N, k, C)

        # replace padding slots with the query point itself
        neigh_xyz [invalid] = q_xyz .unsqueeze(2).expand(-1, -1, self.k, -1)[invalid]
        neigh_feat[invalid] = q_feat.unsqueeze(2).expand(-1, -1, self.k, -1)[invalid]

        # concatenate query token with neighbor tokens
        tokens = torch.cat([q_feat.unsqueeze(2), neigh_feat], dim=2)  # (B, N, k+1, C)
        coords = torch.cat([q_xyz.unsqueeze(2), neigh_xyz],  dim=2)   # (B, N, k+1, 3)

        # key-padding mask for attention (True → ignore)
        pad_mask = torch.cat(
            [
                torch.zeros_like(invalid[..., :1]),  # query token (#0) is always valid
                invalid
            ],
            dim=-1
        ).view(B * N, self.k + 1)                    # (B*N, k+1)

        # reshape to (B*N, S, C) for the transformer layer
        BM = B * N
        fused = self.attn(
            tokens.view(BM, self.k + 1, C).contiguous(),
            coords_src           = coords.view(BM, self.k + 1, 3).contiguous(),
            src_padding_mask = pad_mask,
        )                                            # (BM, k+1, C)

        # return only the query position (index 0 within each group)
        fused_q = fused[:, 0, :].view(B, N, C) + q_feat
        # fused_q = fused[:, 0, :].view(B, N, C) 
        
        return fused_q
