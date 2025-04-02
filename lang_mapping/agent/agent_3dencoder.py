import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module import *
from ..utils import get_3d_coordinates, get_visual_features, transform

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
        voxel_feature_dim: int = 120,
        state_mlp_dim: int = 1024,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
    ):
        """
        Maintains a voxel-hash representation for 3D scenes and uses a CLIP-based
        feature extractor plus an implicit decoder for mapping.
        """
        super().__init__()

        self.device = device
        self.epoch = 0

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # MLP for raw state
        self.state_mlp = nn.Linear(state_dim, state_mlp_dim).to(self.device)
        self.dp3_encoder = DP3Encoder(in_dim=3, out_dim=voxel_feature_dim).to(self.device)

        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)

        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])
        
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            output_dim=state_mlp_dim
        )

        self.state_to_voxeldim = nn.Linear(42, voxel_feature_dim).to(self.device)

        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

    def forward_policy(self, observations, object_labels, step_nums):
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]

        # Depth resizing
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")
            
        else:
            b2, _, _, _ = hand_depth.shape

        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        hand_visfeat = head_visfeat = torch.randn(b2, 768, 16, 16).cuda()

        # Compute 3D coords
        hand_coords_world, _ = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, _ = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # Get text embeddings (projected)
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]


        # Prepare coords for transformer
        batch_hand_coords = hand_coords_world_flat.view(B_, N, 3)
        batch_head_coords = head_coords_world_flat.view(B_, N, 3)

        hand_dp3_feat = self.dp3_encoder(batch_hand_coords) 
        head_dp3_feat = self.dp3_encoder(batch_head_coords) 
        
        state_voxel_dim = self.state_to_voxeldim(state)
        
        # Transformer
        visual_token = self.transformer(
            hand=hand_dp3_feat,
            head=head_dp3_feat,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state_voxel_dim,
            text_embeddings=selected_text_reduced
        )

        # visual_token = torch.cat([hand_dp3_feat, head_dp3_feat], dim=1)

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred
