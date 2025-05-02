import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ...module.transformer import TransformerEncoder, ActionTransformerDecoder
from ...module.mlp import StateProj

from ...utils import get_3d_coordinates, transform

import open_clip

class DP3Encoder(nn.Module):
    """
    DP3 Encoder: 3-layer MLP + LayerNorm + Max-Pooling + Projection
    - Input  : (B, N, in_dim) 
    - Output : (B, out_dim) 
    """
    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        
        self.proj = nn.Linear(256, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, in_dim)
        return: (B, out_dim)
        """
        feats = self.mlp(pts)                 # (B, N, out_dim)
        feats_out = self.ln(self.proj(feats))    # (B, out_dim)
        
        return feats_out

class Agent_3dencoder(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        transf_input_dim: int = 768,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        num_heads: int = 8,
        num_layers_transformer: int = 4
    ):
        super().__init__()

        self.device = device

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # MLP for raw state
        self.dp3_encoder = DP3Encoder(in_dim=3, out_dim=transf_input_dim)

        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)

        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            self.text_embeddings = text_embeddings - redundant_emb


        self.state_proj =  StateProj(state_dim=state_dim, output_dim=transf_input_dim)   

        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=transf_input_dim * 4,
            num_layers=num_layers_transformer,
            num_heads=num_heads
        )

        self.action_dim = np.prod(single_act_shape)
        self.action_transformer = ActionTransformerDecoder(
            d_model=transf_input_dim,         
            nhead=8,
            num_decoder_layers=6,   
            dim_feedforward=transf_input_dim * 4,
            dropout=0.1,
            action_dim=self.action_dim
        )

        self.state_proj =  StateProj(state_dim=state_dim, output_dim=transf_input_dim)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

        self.state_mlp_action = StateProj(state_dim, transf_input_dim)
        
    def forward(self, observations, object_labels):
        
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

        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        # Prepare coords for transformer
        batch_hand_coords_t = hand_coords_world_flat_t.view(B, N, 3)
        batch_head_coords_t = head_coords_world_flat_t.view(B, N, 3)
        batch_hand_coords_m1 = hand_coords_world_flat_m1.view(B, N, 3)
        batch_head_coords_m1= head_coords_world_flat_m1.view(B, N, 3)
        
        hand_dp3_feat_t = self.dp3_encoder(batch_hand_coords_t) 
        head_dp3_feat_t = self.dp3_encoder(batch_head_coords_t) 
        hand_dp3_feat_m1 = self.dp3_encoder(batch_hand_coords_m1) 
        head_dp3_feat_m1 = self.dp3_encoder(batch_head_coords_m1) 
        
        state_proj_t = self.state_proj(state_t)
        state_proj_m1 = self.state_proj(state_m1)
        
        coords_hand_t = hand_coords_world_flat_t.view(B, N, 3)
        coords_head_t = head_coords_world_flat_t.view(B, N, 3)   

        coords_hand_m1 = hand_coords_world_flat_m1.view(B, N, 3)
        coords_head_m1 = head_coords_world_flat_m1.view(B, N, 3)   

        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=hand_dp3_feat_t,
            head_token_t=head_dp3_feat_t,
            hand_token_m1=hand_dp3_feat_m1,
            head_token_m1=head_dp3_feat_m1,
            coords_hand_t=coords_hand_t,
            coords_head_t=coords_head_t,
            coords_hand_m1=coords_hand_m1,
            coords_head_m1=coords_head_m1,
            state_t=state_proj_t.unsqueeze(1),
            state_m1=state_proj_m1.unsqueeze(1), 
        ) # [B, N, 240]

        state_t_proj  = self.state_mlp_action(state_t).unsqueeze(1)  
        action_out = self.action_transformer(out_transformer, state_t_proj)

        return action_out