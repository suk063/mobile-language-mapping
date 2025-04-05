import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from lang_mapping.module.transformer import TransformerEncoder
from ..module.mlp import ActionMLP

from ..utils import get_3d_coordinates, get_visual_features, transform

import open_clip

class Agent_uplift(nn.Module):
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
        self.clip_dim_reducer = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)

        self.state_to_voxeldim = nn.Linear(42, voxel_feature_dim).to(self.device)

        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            output_dim=state_mlp_dim
        )

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

        # Reshape RGB
        hand_rgb = pixels["fetch_hand_rgb"]
        head_rgb = pixels["fetch_head_rgb"]
        if hand_rgb.shape[2] != 3:
            hand_rgb = hand_rgb.permute(0, 1, 4, 2, 3)
            head_rgb = head_rgb.permute(0, 1, 4, 2, 3)
        B, fs, d, H, W = hand_rgb.shape
        hand_rgb = hand_rgb.reshape(B, fs * d, H, W)
        head_rgb = head_rgb.reshape(B, fs * d, H, W)

        hand_rgb = transform(hand_rgb.float() / 255.0)
        head_rgb = transform(head_rgb.float() / 255.0)

        # Depth resizing
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")

        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        # Fine-tuning CLIP
        hand_visfeat = get_visual_features(self.clip_model, hand_rgb)
        head_visfeat = get_visual_features(self.clip_model, head_rgb)

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

        # Flatten CLIP feats
        hand_visfeat = hand_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        head_visfeat = head_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        feats_hand_flat = hand_visfeat.reshape(B_ * N, -1)
        feats_head_flat = head_visfeat.reshape(B_ * N, -1)

        # Reduce CLIP dimension 
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat)
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat)

        # Get text embeddings (projected)
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]

        # Prepare coords for transformer
        batch_hand_coords = hand_coords_world_flat.view(B_, N, 3)
        batch_head_coords = head_coords_world_flat.view(B_, N, 3)

        batch_feats_hand_flat_reduced = feats_hand_flat_reduced.view(B_, N, -1)
        batch_feats_head_flat_reduced = feats_head_flat_reduced.view(B_, N, -1)

        state_voxel_dim = self.state_to_voxeldim(state)

        # Transformer
        visual_token = self.transformer(
            hand=batch_feats_hand_flat_reduced,
            head=batch_feats_head_flat_reduced,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state_voxel_dim,
            text_embeddings=selected_text_reduced
        )

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred
