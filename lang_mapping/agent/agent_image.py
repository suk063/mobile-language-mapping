import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from lang_mapping.module.transformer import TransformerEncoder, ActionTransformerDecoder
from ..module.mlp import ActionMLP, StateProj, DimReducer

from ..utils import get_3d_coordinates, get_visual_features, transform

import open_clip

class Agent_image(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        voxel_feature_dim: int = 240,
        state_mlp_dim: int = 128,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        num_heads: int = 8,
        num_layers_transformer: int = 4
    ):
        """
        Maintains a voxel-hash representation for 3D scenes and uses a CLIP-based
        feature extractor plus an implicit decoder for mapping.
        """
        super().__init__()

        self.device = device

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # MLP for raw state
        self.state_mlp = nn.Linear(state_dim, state_mlp_dim).to(self.device)

        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)

        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        # Text embeddings and projection
        # if text_input:
        #     text_input += [""]
        
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            # Remove the last embedding (blank token) to avoid repetition
            # text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            # text_embeddings = text_embeddings - redundant_emb
            # self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        # Reduce CLIP feature dimension
        self.dim_reducer_hand = DimReducer(clip_input_dim, voxel_feature_dim, L=0)
        self.dim_reducer_head = DimReducer(clip_input_dim, voxel_feature_dim, L=0)

        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=1024,
            num_layers=num_layers_transformer,
            num_heads=num_heads
        )

        self.action_dim = np.prod(single_act_shape)
        self.action_transformer = ActionTransformerDecoder(
            d_model=240,         
            nhead=8,
            num_decoder_layers=6,   
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=self.action_dim
        ).to(self.device)

        self.state_proj_transformer =  StateProj(state_dim=state_dim, output_dim=voxel_feature_dim).to(self.device)   

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

        self.state_mlp_for_action = nn.Linear(state_dim, voxel_feature_dim).to(self.device)
        
    def forward(self, observations, object_labels):
        """
        Forward pass that processes time t and t-1 in a single batch:
         1) Merge data of t and t-1 to form batch=2B
         2) Do feature extraction, depth, coordinates, voxel
         3) Split back into t and t-1
         4) Compute action_t = MLP(state_t, out_t, out_t-1)
        """
        # 1) Extract data
        hand_rgb_t   = observations["pixels"]["fetch_hand_rgb"]
        hand_rgb_m1  = observations["pixels"]["fetch_hand_rgb_m1"]
        head_rgb_t   = observations["pixels"]["fetch_head_rgb"]
        head_rgb_m1  = observations["pixels"]["fetch_head_rgb_m1"]

        hand_depth_t  = observations["pixels"]["fetch_hand_depth"]
        hand_depth_m1 = observations["pixels"]["fetch_hand_depth_m1"]
        head_depth_t  = observations["pixels"]["fetch_head_depth"]
        head_depth_m1 = observations["pixels"]["fetch_head_depth_m1"]

        hand_pose_t   = observations["pixels"]["fetch_hand_pose"]
        hand_pose_m1  = observations["pixels"]["fetch_hand_pose_m1"]
        head_pose_t   = observations["pixels"]["fetch_head_pose"]
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

        feats_hand_t_norm = F.normalize(feats_hand_t, dim=-1, p=2)
        feats_head_t_norm = F.normalize(feats_head_t, dim=-1, p=2)
        feats_hand_m1_norm = F.normalize(feats_hand_m1, dim=-1, p=2)
        feats_head_m1_norm = F.normalize(feats_head_m1, dim=-1, p=2)

        text_embed_norm = F.normalize(self.text_embeddings, dim=-1, p=2)
        text_embed_batch = text_embed_norm[object_labels, :].unsqueeze(1)
        
        gating_score_hand_t = (feats_hand_t_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_head_t = (feats_head_t_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_hand_m1 = (feats_hand_m1_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_head_m1 = (feats_head_m1_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]

        feats_hand_t_gated = feats_hand_t + feats_hand_t * gating_score_hand_t
        feats_head_t_gated = feats_head_t + feats_head_t * gating_score_head_t
        feats_hand_m1_gated = feats_hand_m1 + feats_hand_m1 * gating_score_hand_m1
        feats_head_m1_gated = feats_head_m1 + feats_head_m1 * gating_score_head_m1
                
        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_hand_flat_t = feats_hand_t_gated.reshape(B*N, -1)
        feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_t)
        feats_hand_reduced_t = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_t = feats_head_t_gated.reshape(B*N, -1)
        feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_t)
        feats_head_reduced_t = feats_head_reduced_flat.view(B, N, -1)
        
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_hand_flat_m1 = feats_hand_m1_gated.reshape(B*N, -1)
        feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_m1)
        feats_hand_reduced_m1 = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_m1 = feats_head_m1_gated.reshape(B*N, -1)
        feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_m1)
        feats_head_reduced_m1 = feats_head_reduced_flat.view(B, N, -1)

        state_proj_transformer_t = self.state_proj_transformer(state_t)
        state_proj_transformer_m1 = self.state_proj_transformer(state_m1)

        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=feats_hand_reduced_t,
            head_token_t=feats_head_reduced_t,
            hand_token_m1=feats_hand_reduced_m1,
            head_token_m1=feats_head_reduced_m1,
            state_t=state_proj_transformer_t,
            state_m1=state_proj_transformer_m1, 
        ) # [B, N, 240]
        
        state_t_proj  = self.state_mlp_for_action(state_t).unsqueeze(1)   # [B, 240]
        action_out = self.action_transformer(out_transformer, state_t_proj)
        
        return action_out