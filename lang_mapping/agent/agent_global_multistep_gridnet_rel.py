import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports

from lang_mapping.module.transformer import TransformerEncoder, GlobalPerceiver, LocalSelfAttentionFusion, ActionTransformerDecoder
from lang_mapping.module.mlp import ImplicitDecoder, DimReducer, StateProj
from lang_mapping.grid_net import GridNet
from lang_mapping.utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip
import math

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_causal_mask(seq_len: int, past_len: int, device):
    """tri‑mask that hides K from the ‘past_len’ first tokens."""
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    if past_len > 0:                       # hide K<i=past_len for Q>=past_len
        m[past_len:, :past_len] = True
    return m         

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
    ):
        super().__init__()
        
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
        hand_pose_m1=None, head_pose_m1=None,   # (B, D) each
        hand_pose_t=None,  head_pose_t=None,    # (B, D) each
    ):
        B, N, D = hand_token_t.shape
        tokens, coords = [], []
        
        zeros1 = lambda: torch.zeros(B, 1, 3, device=hand_token_t.device)
        
        # # 0) text
        # tokens.append(text_token.unsqueeze(1))
        # coords.append(torch.zeros(B, 1, 3, device=text_token.device))
        
        # 1) m‑1 state
        if state_m1 is not None:
            tokens.append(state_m1.unsqueeze(1))
            coords.append(zeros1())

        # 2‑3) m‑1 poses
        tokens += [hand_pose_m1.unsqueeze(1), head_pose_m1.unsqueeze(1)]
        coords += [zeros1(), zeros1()]

        # 4‑5) m‑1 visual
        tokens += [hand_token_m1, head_token_m1]
        coords += [coords_hand_m1, coords_head_m1]
        
        # 6)  current state
        if state_t is not None:
            tokens.append(state_t.unsqueeze(1))
            coords.append(zeros1())
        
        # 7‑8) current poses
        tokens += [hand_pose_t.unsqueeze(1), head_pose_t.unsqueeze(1)]
        coords += [zeros1(), zeros1()]
        
        # 9‑10) current visual
        tokens += [hand_token_t, head_token_t]
        coords += [coords_hand_t, coords_head_t]

        src = torch.cat(tokens, 1)                # (B, S, D)
        coords_src = torch.cat(coords, 1) if coords_hand_t is not None else None
        S = src.size(1)
        
        # build masks ---------------------------------------------------------
        past_len = 1 + 2 + N * 2        # state_m1 + 2 poses + 2*visual_m1
        causal_mask = make_causal_mask(S, past_len, device=src.device)

        for layer in self.layers:
            src = layer(
                src, coords_src=coords_src,
                causal_mask=causal_mask
            )
        # discard m‑1 tokens => keep everything after visual_m1
        start_idx = 1 + 2 + N * 2   # state_m1 + 2 poses + 2*visual_m1
        return src[:, start_idx:, :]    # (B, N*? + 3, D)  (3 = state_t + 2 poses)

# --------------------------------------------------------------------- #
#                       feature–text gating helper                      #
# --------------------------------------------------------------------- #
def gate_with_text(feats: torch.Tensor,
                   text_embed: torch.Tensor) -> torch.Tensor:
    """
    Residual gating: feats ← feats + feats ✕ cos_sim(feats,text)

    feats       : (B, N, C) **or** (B*N, C)
    text_embed  : (B, 768)
    proj        : nn.Linear that projects 768 → C if dims differ
    """

    # -- match dimensions ----------------------------------------------
    txt = F.normalize(text_embed, dim=-1).unsqueeze(1)                      # (B,1,C)

    # -- cosine‑similarity gating --------------------------------------
    score = (F.normalize(feats, dim=-1) * txt).sum(-1, keepdim=True)  # (B,N,1)
    gated = feats + feats* score                                   # residual

    return gated                                 # restore shape

class Agent_global_multistep_gridnet_rel(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        voxel_feature_dim: int = 128,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers_transformer: int = 4,
        num_layers_perceiver: int = 2,
        num_learnable_tokens: int = 8,
        action_horizon: int = 8
    ):
        super().__init__()

        self.device = device
        
        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        if text_input:
            text_input += [""]
        text_tokens = self.tokenizer(text_input).to(device)
        with torch.no_grad():
            te = F.normalize(self.clip_model.encode_text(text_tokens), -1)
        self.text_embeddings = te[:-1] - te[-1]
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim)
        
        # --- pose projection -------------------------------------------------
        # hand/head pose sample: torch.Size([B, 1, 3, 4])  ⇒ flat_dim = 12
        pose_flat_dim = int(np.prod(sample_obs["pixels"]["fetch_hand_pose"].shape[2:]))  # 3*4 = 12
        self.pose_proj_hand = nn.Linear(pose_flat_dim, voxel_feature_dim)
        self.pose_proj_head = nn.Linear(pose_flat_dim, voxel_feature_dim)
        
        # Reduce CLIP feature dimension
        self.dim_reducer_hand = nn.Linear(clip_input_dim, voxel_feature_dim)
        self.dim_reducer_head = nn.Linear(clip_input_dim, voxel_feature_dim)
        
        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=1024,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )
        
        self.global_perceiver = GlobalPerceiver(
            input_dim=voxel_feature_dim,
            nhead=num_heads,
            num_layers=num_layers_perceiver,
            out_dim=voxel_feature_dim,
            num_learnable_tokens=num_learnable_tokens
        )
        
        # Action MLP
        self.action_dim = np.prod(single_act_shape)
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=240,         
            nhead=8,
            num_decoder_layers=6,   
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=self.action_dim,
        ).to(self.device)
        
        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder
        
        self.feature_fusion_attn_hand = LocalSelfAttentionFusion(feat_dim=clip_input_dim)
        self.feature_fusion_attn_head = LocalSelfAttentionFusion(feat_dim=clip_input_dim)

        self.state_proj_transf = nn.Linear(
            sample_obs["state"].shape[1], voxel_feature_dim
        ) 

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.state_mlp_for_action = nn.Linear(state_dim, voxel_feature_dim).to(self.device)
        
    # --------------------------------------------------------------------- #
    #                     helper: flatten SE(3) matrix                      #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _flatten_pose(p):            # p: [B, 1, 3, 4]
        return p.squeeze(1).reshape(p.size(0), -1)      # → [B, 12]


    def forward_mapping(self, observations, is_grasp):
        
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
            # hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest")
            # head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest")
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

        # Query voxel features and cos simeilarity
        voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t)
        voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t)

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
            
        total_cos_loss = cos_loss_hand + cos_loss_head
        return total_cos_loss
    
    def forward_policy(self, observations, object_labels):
        
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

        state_t  = observations["state"]
        state_m1 = observations["state_m1"]

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
        voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t)
        voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t)
        
        voxel_feat_points_hand_flat_m1 = self.static_map.query_feature(hand_coords_world_flat_m1)
        voxel_feat_points_head_flat_m1 = self.static_map.query_feature(head_coords_world_flat_m1)
    
        voxel_feat_points_hand_flat_final_t = self.implicit_decoder(voxel_feat_points_hand_flat_t)
        voxel_feat_points_head_flat_final_t = self.implicit_decoder(voxel_feat_points_head_flat_t)
        
        voxel_feat_points_hand_flat_final_m1 = self.implicit_decoder(voxel_feat_points_hand_flat_m1)
        voxel_feat_points_head_flat_final_m1 = self.implicit_decoder(voxel_feat_points_head_flat_m1)
        
        # --------------------------------------------------------------------- #
        # 1)  text embeddings for this batch
        # --------------------------------------------------------------------- #
        text_emb = self.text_embeddings[object_labels]        # (B,768)

        # --------------------------------------------------------------------- #
        # 2)  gate BOTH voxel and visual features (t and m‑1, hand & head)
        # --------------------------------------------------------------------- #
        # -- visual CLIP feats (dim = 768) ------------------------------------
        feats_hand_t  = gate_with_text(feats_hand_t,  text_emb)          # (B,N,768)
        feats_head_t  = gate_with_text(feats_head_t,  text_emb)
        feats_hand_m1 = gate_with_text(feats_hand_m1, text_emb)
        feats_head_m1 = gate_with_text(feats_head_m1, text_emb)

        # -- voxel features from implicit decoder (dim = voxel_feature_dim) ---
        voxel_feat_points_hand_final_t  = gate_with_text(
            voxel_feat_points_hand_flat_final_t.view(B, N, -1),  text_emb)
        voxel_feat_points_head_final_t  = gate_with_text(
            voxel_feat_points_head_flat_final_t.view(B, N, -1),  text_emb)
        voxel_feat_points_hand_final_m1 = gate_with_text(
            voxel_feat_points_hand_flat_final_m1.view(B, N, -1), text_emb)
        voxel_feat_points_head_final_m1 = gate_with_text(
            voxel_feat_points_head_flat_final_m1.view(B, N, -1), text_emb)

        fused_hand_t = feats_hand_t
        fused_head_t = feats_head_t
        fused_hand_m1 = feats_hand_m1
        fused_head_m1 = feats_head_m1
        
        # fused_hand_t = self.feature_fusion_attn_hand(
        #     voxel_feat_points_hand_final_t,
        #     feats_hand_t,
        # ).reshape(B*N, -1)
        
        # fused_head_t = self.feature_fusion_attn_head(
        #     voxel_feat_points_head_final_t,
        #     feats_head_t,
        # ).reshape(B*N, -1)
        
        # fused_hand_m1 = self.feature_fusion_attn_hand(
        #     voxel_feat_points_hand_final_m1,
        #     feats_hand_m1,
        # ).reshape(B*N, -1)
        
        # fused_head_m1 = self.feature_fusion_attn_head(
        #     voxel_feat_points_head_final_m1,
        #     feats_head_m1,
        # ).reshape(B*N, -1)
        
        fused_hand_reduced_t = self.dim_reducer_hand(fused_hand_t).view(B, N, -1)
        fused_head_reduced_t = self.dim_reducer_head(fused_head_t).view(B, N, -1)
        
        fused_hand_reduced_m1 = self.dim_reducer_hand(fused_hand_m1).view(B, N, -1)
        fused_head_reduced_m1 = self.dim_reducer_head(fused_head_m1).view(B, N, -1)                

        state_proj_transf_t = self.state_proj_transf(state_t)
        state_proj_transf_m1 = self.state_proj_transf(state_m1)

        hand_pose_m1_proj = self.pose_proj_hand(self._flatten_pose(
            observations["pixels"]["fetch_hand_pose_m1"]
        ))
        head_pose_m1_proj = self.pose_proj_head(self._flatten_pose(
            observations["pixels"]["fetch_head_pose_m1"]
        ))
        hand_pose_t_proj = self.pose_proj_hand(self._flatten_pose(
            observations["pixels"]["fetch_hand_pose"]
        ))
        head_pose_t_proj = self.pose_proj_head(self._flatten_pose(
            observations["pixels"]["fetch_head_pose"]
        ))
        
        # text_token_proj = self.text_proj(text_emb)          

        coords_hand_t     = hand_coords_world_flat_t.view(B, N, 3)
        coords_head_t     = head_coords_world_flat_t.view(B, N, 3)
        coords_hand_m1    = hand_coords_world_flat_m1.view(B, N, 3)
        coords_head_m1    = head_coords_world_flat_m1.view(B, N, 3)

        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=fused_hand_reduced_t,
            head_token_t=fused_head_reduced_t,
            hand_token_m1=fused_hand_reduced_m1,
            head_token_m1=fused_head_reduced_m1,
            coords_hand_t=coords_hand_t,
            coords_head_t=coords_head_t,
            coords_hand_m1=coords_hand_m1,
            coords_head_m1=coords_head_m1,
            state_t=state_proj_transf_t,
            state_m1=state_proj_transf_m1,
            hand_pose_m1=hand_pose_m1_proj,
            head_pose_m1=head_pose_m1_proj,
            hand_pose_t=hand_pose_t_proj,
            head_pose_t=head_pose_t_proj, 
        ) # [B, N, 240]
        
        state_t_proj  = self.state_mlp_for_action(state_t).unsqueeze(1)   # [B, 240]
        action_out = self.action_transformer(out_transformer, state_t_proj)
        
        return action_out