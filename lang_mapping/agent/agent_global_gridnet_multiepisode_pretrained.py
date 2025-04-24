import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ..module.transformer import ActionTransformerDecoder
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj, ConcatMLPFusion
from lang_mapping.grid_net import GridNet

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip
import math
from typing import Optional
import timm

# ──────────────────────────────────────────────────────────────────────
# 2.  Perceiver cross-attention (learnable query, RoPE, padding mask)
# ──────────────────────────────────────────────────────────────────────
class ScenePerceiver(nn.Module):
    def __init__(self,
        input_dim:          int = 240,
        nhead:              int = 8,
        num_layers:         int = 4,
        hidden_dim:         int = 1024,
        num_learn_tokens:   int = 16,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_learn_tokens, input_dim))
        nn.init.xavier_uniform_(self.query_tokens)

        self.layers = nn.ModuleList([
            PerceiverAttentionLayer(
                dim=input_dim, nhead=nhead, dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        kv_coords:      torch.Tensor,    # (B,L_max,3)
        kv_feats:       torch.Tensor,    # (B,L_max,C)
        kv_pad_mask:    torch.Tensor,    # (B,L_max)  True=pad
        head_trans:     torch.Tensor,    # (B,3)
    ) -> torch.Tensor:                   # (B,num_learn_tokens,C)
        B, _, C = kv_feats.shape
        q       = self.query_tokens.expand(B, -1, -1)          # (B,T,C)  T = num_learn_tokens
        # every query token uses same head-translation coord
        coords_q = head_trans.unsqueeze(1).expand_as(q[..., :3])  # (B,T,3)

        x = q
        for layer in self.layers:
            x = layer(
                q            = x,
                k            = kv_feats,
                v            = kv_feats,
                coords_q     = coords_q,
                coords_kv    = kv_coords,
                kv_pad_mask  = kv_pad_mask      # see modified layer below
            )

        return x                                   # (B,T,C)

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,             # (B, S, d_model)
        coords_src: torch.Tensor = None,  # (B, S, 3) or None
        causal_mask=None,
    ) -> torch.Tensor:
        # src shape: (B, S, d_model)
        B, S, _ = src.shape
        
        # 1) Q, K, V projections
        q = self.W_q(src)  # (B, S, d_model)
        k = self.W_k(src)
        v = self.W_v(src)
        
        # 2) Reshape and transpose for multi-head
        # => (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3) Apply RoPE if coords_src is provided
        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
            # v is often unchanged in RoPE
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
        attn = torch.matmul(F.softmax(scores, -1), v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.d_model)
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        
        return self.norm2(src2 + self.dropout_ff(ff))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=240,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        num_learn_tokens = 4,
    ):
        super().__init__()
        
        self.num_learn_tokens = num_learn_tokens
        
        self.tokens_m1 = nn.Parameter(torch.zeros(1, num_learn_tokens, input_dim))
        self.tokens_t  = nn.Parameter(torch.zeros(1, num_learn_tokens, input_dim))
        nn.init.xavier_uniform_(self.tokens_m1)
        nn.init.xavier_uniform_(self.tokens_t)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=input_dim,
                n_heads=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
                
        self.apply(init_weights_kaiming)
               
    def forward(
        self,
        hand_token_t, head_token_t, hand_token_m1, head_token_m1,
        coords_hand_t=None, coords_head_t=None,
        coords_hand_m1=None, coords_head_m1=None,
        state_t=None, state_m1=None,    
        trans_head_t=None, trans_head_m1=None
    ):
        B, N, D = hand_token_t.shape
        tokens, coords = [], []
        
        zeros1 = lambda: torch.zeros(B, 1, 3, device=hand_token_t.device)
        
        
        # ── (A) m-1 segment ───────────────────────────────────────────
        if state_m1 is not None:
            tokens.append(state_m1.unsqueeze(1))
            coords.append(trans_head_m1.unsqueeze(1) if trans_head_m1 is not None else zeros1())
            
        tokens += [hand_token_m1, head_token_m1]
        coords += [coords_hand_m1, coords_head_m1]
        
        tok_m1   = self.tokens_m1.expand(B, -1, -1)                      # (B,ℓ,D)
        coord_m1 = trans_head_m1.unsqueeze(1).expand(-1, self.num_learn_tokens, -1)
        tokens.append(tok_m1)
        coords.append(coord_m1)
        
        # 6)  current state
        if state_t is not None:
            tokens.append(state_t.unsqueeze(1))
            coords.append(trans_head_t.unsqueeze(1)  if trans_head_t  is not None else zeros1())
        
        # 9‑10) current visual
        tokens += [hand_token_t, head_token_t]
        coords += [coords_hand_t, coords_head_t]
        
        tok_t   = self.tokens_t.expand(B, -1, -1)                        # (B,ℓ,D)
        coord_t = trans_head_t.unsqueeze(1).expand(-1, self.num_learn_tokens, -1)
        tokens.append(tok_t)
        coords.append(coord_t)
        
        src        = torch.cat(tokens,  1)                               # (B,S,D)
        coords_src = torch.cat(coords, 1) if coords_hand_t is not None else None
        
        len_m1 = self.num_learn_tokens                  # learnable m-1
        if state_m1 is not None: len_m1 += 1
        len_m1 += hand_token_m1.size(1) * 2

        len_t  = self.num_learn_tokens                  # learnable t
        if state_t is not None: len_t  += 1
        len_t  += hand_token_t.size(1) * 2

        causal_mask = build_causal_mask(len_m1, len_t, src.device) 

        for layer in self.layers:
            src = layer(
                src, coords_src=coords_src,
                causal_mask=causal_mask
            )
        
        return src[:,-self.num_learn_tokens:,:]    # (B, N*? + 1, D)

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def build_causal_mask(n_m1, n_t, device):
    S    = n_m1 + n_t
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)

    # m‑1 Query / t Key
    mask[0:n_m1, n_m1:] = True

    return mask


class PerceiverAttentionLayer(nn.Module):
    def __init__(
        self, 
        dim: int = 256, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        assert dim % nhead == 0, "dim must be divisible by nhead"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Output projection after attention
        self.out_proj = nn.Linear(dim, dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim),
        )

        # LayerNorm layers
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, q, k, v,
                coords_q=None, coords_kv=None,
                kv_pad_mask: Optional[torch.Tensor] = None):      # (B,KV)
        B, Q_len, _  = q.shape
        _, KV_len, _ = k.shape

        q_proj = self.W_q(q).view(B, Q_len, self.nhead, self.head_dim).transpose(1,2)
        k_proj = self.W_k(k).view(B, KV_len, self.nhead, self.head_dim).transpose(1,2)
        v_proj = self.W_v(v).view(B, KV_len, self.nhead, self.head_dim).transpose(1,2)

        if coords_q  is not None: q_proj = rotary_pe_3d(q_proj, coords_q)
        if coords_kv is not None: k_proj = rotary_pe_3d(k_proj, coords_kv)

        scores = torch.matmul(q_proj, k_proj.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if kv_pad_mask is not None:                                   # NEW
            scores = scores.masked_fill(kv_pad_mask[:,None,None,:],   # broadcast
                                         float('-inf'))

        attn = torch.matmul(scores.softmax(-1), v_proj)
        attn = attn.transpose(1,2).reshape(B, Q_len, self.dim)

        x = self.ln1(q + self.dropout_attn(self.out_proj(attn)))
        x = self.ln2(x + self.dropout_ff(self.ffn(x)))
        return x
 
# --------------------------------------------------------------------- #
#                       feature–text gating helper                      #
# --------------------------------------------------------------------- #
def gate_with_text(feats: torch.Tensor,
                   text_embed: torch.Tensor) -> torch.Tensor:
    """
    Residual gating: feats ← feats + feats ✕ cos_sim(feats,text)

    feats       : (B, N, C) 
    text_embed  : (B, 768)
    proj        : nn.Linear that projects 768 → C if dims differ
    """

    # -- match dimensions ----------------------------------------------
    txt = F.normalize(text_embed, dim=-1).unsqueeze(1)                      # (B,1,C)

    # -- cosine‑similarity gating --------------------------------------
    score = (F.normalize(feats, dim=-1) * txt).sum(-1, keepdim=True)  # (B,N,1)
    gated = feats + feats* score                                   # residual

    return gated    

class Agent_global_gridnet_multiepisode_pretrained(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        transf_input_dim: int = 240,
        state_mlp_dim: int = 128,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers_transformer: int = 4,
        num_layers_perceiver: int = 4,
        num_learnable_tokens: int = 16,
        num_action_layer: int = 6,
        action_horizon: int = 16
    ):
        super().__init__()

        self.device = device

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        pose_flat_dim      = int(np.prod(sample_obs["pixels"]["fetch_hand_pose"].shape[2:]))
        raw_state_dim      = sample_obs["state"].shape[1]        # 42
        
        state_dim = raw_state_dim + pose_flat_dim
        
        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        # Text embeddings and projection
        if text_input:
            text_input += [""]
        
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            self.text_embeddings = text_embeddings - redundant_emb


        # Reduce CLIP feature dimension
        self.dim_reducer_global = nn.Linear(clip_input_dim, transf_input_dim)
        
        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=1024,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
            num_learn_tokens=num_learnable_tokens
        )
        
        # Action MLP
        self.action_dim = np.prod(single_act_shape)
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=transf_input_dim,         
            nhead=8,
            num_decoder_layers=num_action_layer,   
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=self.action_dim,
            action_horizon=action_horizon
        )

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder
        
        self.state_proj_transf =  nn.Linear(state_dim, transf_input_dim)
        self.state_proj_percei =  nn.Linear(state_dim, transf_input_dim) 

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.state_mlp_for_action = nn.Linear(state_dim, transf_input_dim).to(self.device)
        
        self.transf_input_dim   = transf_input_dim
        self.scene_perceiver    = ScenePerceiver(
            input_dim         = transf_input_dim,
            nhead             = num_heads,
            num_layers        = num_layers_perceiver,
            hidden_dim        = 1024,
            num_learn_tokens  = num_learnable_tokens,
        )
        
        self.visual_fuser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model      = transf_input_dim,
                nhead        = num_heads,
                dim_feedforward = 1024,  
                dropout      = 0.1,
                batch_first  = True,
            ),
            num_layers = 4,     
        )
    
    @staticmethod
    def _flatten_pose(p):            # p: [B, 1, 3, 4]
        return p.squeeze(1).reshape(p.size(0), -1)      # → [B, 12]

    def _gather_scene_kv(
        self,
        batch_episode_ids: torch.Tensor,         # [B]
        text_emb:          torch.Tensor,         # [B,768]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        kv_coords     : (B,  L_max, 3)   padded with 0
        kv_feats      : (B,  L_max, C)   padded with 0
        kv_pad_mask   : (B,  L_max)      True ⟹ pad
        """
        B                              = batch_episode_ids.size(0)
        scene_ids                       = batch_episode_ids.tolist()
        per_scene_coords, per_scene_len = [], []

        # ── 1) collect coordinates per scene ──────────────────────────────
        for sid in scene_ids:
            c = self.valid_coords[int(sid)].to(self.device)     # (L_i,3)
            per_scene_coords.append(c)
            per_scene_len.append(c.size(0))

        L_max = max(per_scene_len)

        kv_coords   = torch.zeros(B, L_max, 3,                   device=self.device)
        kv_feats    = torch.zeros(B, L_max, self.transf_input_dim, device=self.device)
        kv_pad_mask = torch.ones( B, L_max, dtype=torch.bool,    device=self.device)

        # ── 2) query voxel features scene-wise ────────────────────────────
        for b, (sid, coords) in enumerate(zip(scene_ids, per_scene_coords)):
            L                     = coords.size(0)
            kv_coords  [b, :L]    = coords
            kv_pad_mask[b, :L]    = False                         # not pad

            scene_ids_tensor      = torch.full((L, 1), sid, device=self.device)
            with torch.no_grad():
                vox_raw           = self.static_map.query_feature(coords, scene_ids_tensor)
                vox_feat          = self.implicit_decoder(vox_raw)          # (L,F_dec)

            # text-gating + dim reduce  → 240-D (transf_input_dim)
            vox_feat              = gate_with_text(vox_feat.unsqueeze(0), text_emb[b:b+1]).squeeze(0)
            kv_feats   [b, :L]    = vox_feat

        return kv_coords, kv_feats, kv_pad_mask
    
    def forward_mapping(self, observations, is_grasp, batch_episode_ids):
        
        bool_mask = (is_grasp < 0.5)  
        if bool_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 1) Extract data
        hand_rgb_t   = observations["pixels"]["fetch_hand_rgb"][bool_mask]
        head_rgb_t   = observations["pixels"]["fetch_head_rgb"][bool_mask]

        hand_depth_t  = observations["pixels"]["fetch_hand_depth"][bool_mask]
        head_depth_t  = observations["pixels"]["fetch_head_depth"][bool_mask]

        hand_pose_t   = observations["pixels"]["fetch_hand_pose"][bool_mask]
        head_pose_t   = observations["pixels"]["fetch_head_pose"][bool_mask]

        batch_valid_episode_ids = batch_episode_ids[bool_mask]

        B = hand_rgb_t.shape[0]
        
        # If needed, permute hand_rgb_t so channel=3
        if hand_rgb_t.shape[2] != 3:
            hand_rgb_t = hand_rgb_t.permute(0, 1, 4, 2, 3)
            head_rgb_t = head_rgb_t.permute(0, 1, 4, 2, 3)
        
        # Flatten frames
        _, fs, d, H, W = hand_rgb_t.shape
        hand_rgb_t = hand_rgb_t.reshape(B, fs * d, H, W)
        head_rgb_t = head_rgb_t.reshape(B, fs * d, H, W)
        
        # Transform to [0,1], apply normalization
        hand_rgb_t = transform(hand_rgb_t.float() / 255.0)
        head_rgb_t = transform(head_rgb_t.float() / 255.0)

        with torch.no_grad():
            hand_visfeat_t = get_visual_features(self.clip_model, hand_rgb_t)
            head_visfeat_t = get_visual_features(self.clip_model, head_rgb_t)
        
        # Handle depth (reshape, interpolate)
        hand_depth_t = hand_depth_t / 1000.0
        head_depth_t = head_depth_t / 1000.0
        
        if hand_depth_t.dim() == 5:
            _, fs, d2, H, W = hand_depth_t.shape
            hand_depth_t = hand_depth_t.view(B, fs * d2, H, W)
            head_depth_t = head_depth_t.view(B, fs * d2, H, W)
            hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest-exact")
            head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest-exact")

        # 3D world coords
        hand_coords_world_t, hand_coords_camera_t = get_3d_coordinates(
            hand_depth_t, hand_pose_t, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_t, head_coords_camera_t = get_3d_coordinates(
            head_depth_t, head_pose_t,
            self.fx, self.fy, self.cx, self.cy
        )

        # Reduce CLIP dimension for hand/head
        _, C_, Hf, Wf = hand_coords_world_t.shape
        N = Hf * Wf

        feats_hand_t = hand_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_t = head_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
                
        feats_hand_flat_t = feats_hand_t.reshape(B*N, -1)
        feats_head_flat_t = feats_head_t.reshape(B*N, -1)                
                
        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        hand_coords_camera_flat_t = hand_coords_camera_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_camera_flat_t = head_coords_camera_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        # filtering out points
        hand_depth_flat_t = hand_depth_t.reshape(B*N)
        head_depth_flat_t = head_depth_t.reshape(B*N)

        depth_mask_hand = hand_depth_flat_t > 0.3  
        depth_mask_head = head_depth_flat_t > 0.6
        
        scene_ids_flat = batch_valid_episode_ids.view(-1, 1).expand(-1, N).reshape(-1, 1)       # (B*N,1)


        # Query voxel features and cos simeilarity
        voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t, scene_ids_flat)
        voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t, scene_ids_flat)

        # Implicit decoder
        # hand
        voxel_feat_points_hand_masked = voxel_feat_points_hand_flat_t[depth_mask_hand]
        coords_camera_hand_masked     = hand_coords_camera_flat_t[depth_mask_hand]
        feats_hand_masked            = feats_hand_flat_t[depth_mask_hand]

        if voxel_feat_points_hand_masked.shape[0] > 0:
            voxel_feat_points_hand_final = self.implicit_decoder(voxel_feat_points_hand_masked)
            cos_sim_hand = F.cosine_similarity(voxel_feat_points_hand_final, feats_hand_masked, dim=-1)
            cos_loss_hand = 1.0 - cos_sim_hand.mean()
        else:
            cos_loss_hand = 0.0

        # head
        voxel_feat_points_head_masked = voxel_feat_points_head_flat_t[depth_mask_head]
        coords_camera_head_masked     = head_coords_camera_flat_t[depth_mask_head]
        feats_head_masked            = feats_head_flat_t[depth_mask_head]

        if voxel_feat_points_head_masked.shape[0] > 0:
            voxel_feat_points_head_final = self.implicit_decoder(voxel_feat_points_head_masked)
            cos_sim_head = F.cosine_similarity(voxel_feat_points_head_final, feats_head_masked, dim=-1)
            cos_loss_head = 1.0 - cos_sim_head.mean()
        else:
            cos_loss_head = 0.0
    
        if cos_loss_hand == 0 and cos_loss_head == 0:
            return 0
        elif cos_loss_hand == 0:
            return cos_loss_head
        elif cos_loss_head == 0:
            return cos_loss_hand
        else:
            return (cos_loss_hand + cos_loss_head) /2
        
    
    def forward_policy(self, observations, object_labels, batch_episode_ids):

        # 1) Extract data
        hand_rgb_t   = observations["pixels"]["fetch_hand_rgb"]
        head_rgb_t   = observations["pixels"]["fetch_head_rgb"]

        hand_depth_t  = observations["pixels"]["fetch_hand_depth"]
        head_depth_t  = observations["pixels"]["fetch_head_depth"]

        hand_pose_t   = observations["pixels"]["fetch_hand_pose"]
        head_pose_t   = observations["pixels"]["fetch_head_pose"]

        hand_rgb_m1  = observations["pixels"]["fetch_hand_rgb_m1"]
        head_rgb_m1  = observations["pixels"]["fetch_head_rgb_m1"]

        hand_depth_m1 = observations["pixels"]["fetch_hand_depth_m1"]
        head_depth_m1 = observations["pixels"]["fetch_head_depth_m1"]

        hand_pose_m1  = observations["pixels"]["fetch_hand_pose_m1"]
        head_pose_m1  = observations["pixels"]["fetch_head_pose_m1"]

        head_pose_flat_t   = self._flatten_pose(head_pose_t)     # [B,16]
        head_pose_flat_m1  = self._flatten_pose(head_pose_m1)

        state_t  = observations["state"]
        state_m1 = observations["state_m1"]
        
        state_t_cat  = torch.cat([state_t,  head_pose_flat_t],  dim=1)
        state_m1_cat = torch.cat([state_m1, head_pose_flat_m1], dim=1)

        B = hand_rgb_t.shape[0]
    
        # If needed, permute hand_rgb_t so channel=3
        if hand_rgb_t.shape[2] != 3:
            hand_rgb_t = hand_rgb_t.permute(0, 1, 4, 2, 3)
            head_rgb_t = head_rgb_t.permute(0, 1, 4, 2, 3)
            hand_rgb_m1 = hand_rgb_m1.permute(0, 1, 4, 2, 3)
            head_rgb_m1 = head_rgb_m1.permute(0, 1, 4, 2, 3)
        
        # Flatten frames
        _, fs, d, H, W = hand_rgb_t.shape
        hand_rgb_t = hand_rgb_t.reshape(B, fs * d, H, W)
        head_rgb_t = head_rgb_t.reshape(B, fs * d, H, W)
        hand_rgb_m1 = hand_rgb_m1.reshape(B, fs * d, H, W)
        head_rgb_m1 = head_rgb_m1.reshape(B, fs * d, H, W)

        # Transform to [0,1], apply normalization
        hand_rgb_t = transform(hand_rgb_t.float() / 255.0)
        head_rgb_t = transform(head_rgb_t.float() / 255.0)
        hand_rgb_m1 = transform(hand_rgb_m1.float() / 255.0)
        head_rgb_m1 = transform(head_rgb_m1.float() / 255.0)
        
        with torch.no_grad():
            hand_visfeat_t = get_visual_features(self.clip_model, hand_rgb_t)
            head_visfeat_t = get_visual_features(self.clip_model, head_rgb_t)
            hand_visfeat_m1 = get_visual_features(self.clip_model, hand_rgb_m1)
            head_visfeat_m1 = get_visual_features(self.clip_model, head_rgb_m1)
        
        # Handle depth (reshape, interpolate)
        
        hand_depth_t = hand_depth_t / 1000.0
        head_depth_t = head_depth_t / 1000.0
        hand_depth_m1 = hand_depth_m1 / 1000.0
        head_depth_m1 = head_depth_m1 / 1000.0
        
        if hand_depth_t.dim() == 5:
            _, fs, d2, H, W = hand_depth_t.shape
            hand_depth_t = hand_depth_t.view(B, fs * d2, H, W)
            head_depth_t = head_depth_t.view(B, fs * d2, H, W)
            hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest-exact")
            head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest-exact")
            
            hand_depth_m1 = hand_depth_m1.view(B, fs * d2, H, W)
            head_depth_m1 = head_depth_m1.view(B, fs * d2, H, W)
            hand_depth_m1 = F.interpolate(hand_depth_m1, (16, 16), mode="nearest-exact")
            head_depth_m1 = F.interpolate(head_depth_m1, (16, 16), mode="nearest-exact")

        # 3D world coords
        hand_coords_world_t, _ = get_3d_coordinates(
            hand_depth_t, hand_pose_t, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_t, _ = get_3d_coordinates(
            head_depth_t, head_pose_t,
            self.fx, self.fy, self.cx, self.cy
        )
        
        hand_coords_world_m1, _ = get_3d_coordinates(
            hand_depth_m1, hand_pose_m1, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_m1, _ = get_3d_coordinates(
            head_depth_m1, head_pose_m1,
            self.fx, self.fy, self.cx, self.cy
        )

        # Reduce CLIP dimension for hand/head
        _, C_, Hf, Wf = hand_coords_world_t.shape
        N = Hf * Wf

        feats_hand_t = hand_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_t = head_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_hand_m1 = hand_visfeat_m1.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_m1 = head_visfeat_m1.permute(0, 2, 3, 1).reshape(B, N, -1)

        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)        
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        # Query voxel features and cos simeilarity
        scene_ids_flat = batch_episode_ids.view(-1, 1).expand(-1, N).reshape(-1, 1)
        
        # with torch.no_grad():
        #     voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t, scene_ids_flat)
        #     voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t, scene_ids_flat)
            
        #     voxel_feat_points_hand_flat_m1 = self.static_map.query_feature(hand_coords_world_flat_m1, scene_ids_flat)
        #     voxel_feat_points_head_flat_m1 = self.static_map.query_feature(head_coords_world_flat_m1, scene_ids_flat)
    
        # voxel_feat_points_hand_flat_final_t = self.implicit_decoder(voxel_feat_points_hand_flat_t)
        # voxel_feat_points_head_flat_final_t = self.implicit_decoder(voxel_feat_points_head_flat_t)
        
        # voxel_feat_points_hand_flat_final_m1 = self.implicit_decoder(voxel_feat_points_hand_flat_m1)
        # voxel_feat_points_head_flat_final_m1 = self.implicit_decoder(voxel_feat_points_head_flat_m1)
        
        # --------------------------------------------------------------------- #
        # 1)  text embeddings for this batch
        # --------------------------------------------------------------------- #
        text_emb = self.text_embeddings[object_labels]        # (B,768)

        # --------------------------------------------------------------------- #
        # 2)  gate BOTH voxel and visual features (t and m‑1, hand & head)
        # --------------------------------------------------------------------- #
        # -- visual CLIP feats (dim = 768) ------------------------------------
        feats_hand_t  = gate_with_text(feats_hand_t,  text_emb)        # (B*N,768)
        feats_head_t  = gate_with_text(feats_head_t,  text_emb)
        feats_hand_m1 = gate_with_text(feats_hand_m1, text_emb)
        feats_head_m1 = gate_with_text(feats_head_m1, text_emb)

        # -- voxel features from implicit decoder (dim = voxel_feature_dim) ---
        # voxel_feat_points_hand_final_t  = gate_with_text(
        #     voxel_feat_points_hand_flat_final_t.view(B, N, -1),  text_emb)
        # voxel_feat_points_head_final_t  = gate_with_text(
        #     voxel_feat_points_head_flat_final_t.view(B, N, -1),  text_emb)
        # voxel_feat_points_hand_final_m1 = gate_with_text(
        #     voxel_feat_points_hand_flat_final_m1.view(B, N, -1), text_emb)
        # voxel_feat_points_head_final_m1 = gate_with_text(
        #     voxel_feat_points_head_flat_final_m1.view(B, N, -1), text_emb)

        # fused_hand_t = self.feature_fusion_hand(feats_hand_t, voxel_feat_points_hand_final_t)
        # fused_head_t = self.feature_fusion_head(feats_head_t, voxel_feat_points_head_final_t)
        # fused_hand_m1 = self.feature_fusion_hand(feats_hand_m1, voxel_feat_points_hand_final_m1)
        # fused_head_m1 = self.feature_fusion_head(feats_head_m1, voxel_feat_points_head_final_m1)  

        state_proj_transf_t = self.state_proj_transf(state_t_cat)
        state_proj_transf_m1 = self.state_proj_transf(state_m1_cat)      
        
        # global voxels
        trans_head_t   = head_pose_t[:, 0, :3, 3]      # (B,3)
        trans_head_m1  = head_pose_m1[:, 0, :3, 3]
        
        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=feats_hand_t,
            head_token_t=feats_head_t,
            hand_token_m1=feats_hand_m1,
            head_token_m1=feats_head_m1,
            coords_hand_t=hand_coords_world_flat_t.view(B, N, 3),
            coords_head_t=head_coords_world_flat_t.view(B, N, 3),
            coords_hand_m1=hand_coords_world_flat_m1.view(B, N, 3),
            coords_head_m1=head_coords_world_flat_m1.view(B, N, 3),
            state_t=state_proj_transf_t,
            state_m1=state_proj_transf_m1,
            trans_head_t=trans_head_t,          
            trans_head_m1=trans_head_m1
        ) # [B, N, 240]
        
        kv_coords, kv_feats, kv_pad = self._gather_scene_kv(batch_episode_ids, text_emb)
        perceiver_out = self.scene_perceiver(     # nn.Module registered in __init__
            kv_coords = kv_coords,
            kv_feats  = kv_feats,
            kv_pad_mask = kv_pad,
            head_trans  = trans_head_t,
        )    
        
        visual_tokens = torch.cat([out_transformer,  perceiver_out],  dim=1)
        visual_tokens = self.visual_fuser(visual_tokens)
        
        state_t_proj  = self.state_mlp_for_action(state_t_cat).unsqueeze(1)    # [B, 240]
        
        action_out = self.action_transformer(visual_tokens, state_t_proj)
        
        return action_out
    

