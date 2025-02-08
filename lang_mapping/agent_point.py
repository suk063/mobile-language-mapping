import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from .policy import TransformerEncoder, LocalSelfAttentionFusion, ActionMLP
from .mapping import ImplicitDecoder, VoxelHashTable
from .utils import get_3d_coordinates, get_visual_features, transform

import open_clip

class Agent_point(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        # Configuration parameters
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        state_mlp_dim: int = 1024,
        voxel_feature_dim: int = 120,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        max_time_steps: int = 201,
        temporal_emb_dim: int = 12,
        clip_loss_coef: float = 0.01,
        hash_voxel: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
    ):
        """
        Agent that maintains a voxel hash representation for 3D scenes and uses
        CLIP-based visual features together with an implicit decoder for mapping.
        """
        super().__init__()

        if text_input:
            text_input += [""]

        self.device = device
        self.clip_loss_coef = clip_loss_coef
        self.epoch = 0

        # State dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # State MLP (for the raw state vector)
        self.state_mlp = nn.Linear(state_dim, state_mlp_dim).to(self.device)

        # CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        # Text
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            # Subtract the last embedding to avoid repetition in the blank token
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            text_embeddings = text_embeddings - redundant_emb
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        # Projection to reduce CLIP feature dim
        self.clip_dim_reducer = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)

        # Transformer
        self.transformer = TransformerEncoder(input_dim=voxel_feature_dim, hidden_dim=256, num_layers=2, num_heads=8, output_dim=state_mlp_dim)  

        # Action MLP is now a separate class
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hash table
        self.hash_voxel = hash_voxel
        self.implicit_decoder = implicit_decoder
   
        self.used_voxel_idx_set = set()
        self.voxel_grid = None
        self.voxel_points_dict = None

        # Time embeddings
        self.temporal_emb_table_hand = nn.Parameter(
            torch.randn(max_time_steps, temporal_emb_dim, device=self.device) * 0.01
        )
        self.temporal_emb_table_head = nn.Parameter(
            torch.randn(max_time_steps, temporal_emb_dim, device=self.device) * 0.01
        )

        # Additional transforms for voxel features
        self.voxel_proj = nn.Linear(voxel_feature_dim, voxel_feature_dim).to(self.device)
        self.feature_fusion = LocalSelfAttentionFusion(feat_dim=voxel_feature_dim)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

    def get_or_build_voxel_grid(self):
        """Return a dict of (ix, iy, iz) -> voxel_feature for used voxels."""
        if self.voxel_grid is not None:
            return self.voxel_grid

        voxel_grid = {}
        for i_voxel in self.used_voxel_idx_set:
            coords_3d = self.hash_voxel.voxel_coords[i_voxel]
            ix, iy, iz = torch.floor(coords_3d / self.hash_voxel.resolution).to(torch.int64)
            voxel_grid[(ix.item(), iy.item(), iz.item())] = self.hash_voxel.voxel_features[i_voxel]
        self.voxel_grid = voxel_grid
        return voxel_grid

    def get_or_build_voxel_points_dict(self):
        """
        Return a dict of (ix, iy, iz) -> [points].
        Each entry has up to a certain number of 3D points that fell into that voxel.
        """
        if self.voxel_points_dict is not None:
            return self.voxel_points_dict

        voxel_points_dict = {}
        for v_idx, points_list in self.hash_voxel.voxel_points.items():
            if v_idx < 0:
                continue
            coords_3d = self.hash_voxel.voxel_coords[v_idx]
            ix, iy, iz = torch.floor(coords_3d / self.hash_voxel.resolution).to(torch.int64)
            voxel_points_dict[(ix.item(), iy.item(), iz.item())] = points_list
        self.voxel_points_dict = voxel_points_dict
        return voxel_points_dict

    def forward(self, observations, object_labels, step_nums):
        """
        1. Process RGB/Depth from observations.
        2. Extract CLIP features.
        3. Look up voxel features.
        4. Compute reconstruction loss (cosine similarity).
        5. Fuse features, pass through Transformer, and then Action MLP.
        """
        # Unpack inputs
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]  # [B, state_dim]

        # Preprocess RGB
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

        # Depth
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")

        # Camera poses
        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        # Freeze CLIP feature extraction
        with torch.no_grad():
            hand_visfeat = get_visual_features(self.clip_model, hand_rgb)  # [B, clip_input_dim, Hf, Wf]
            head_visfeat = get_visual_features(self.clip_model, head_rgb)

        # 3D world coords
        hand_coords_world, hand_coords_cam = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, head_coords_cam = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        # Flatten coordinates
        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # Flatten CLIP features
        hand_visfeat = hand_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        head_visfeat = head_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)

        feats_hand_flat = hand_visfeat.reshape(B_ * N, -1)
        feats_head_flat = head_visfeat.reshape(B_ * N, -1)

        # Reduce feature dim
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat)  # [B*N, voxel_feature_dim]
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat)

        feats_hand_reduced = feats_hand_flat_reduced.reshape(B_, N, -1)  # [B, N, voxel_feature_dim]
        feats_head_reduced = feats_head_flat_reduced.reshape(B_, N, -1)

        # Voxel lookup
        return_indices = (self.epoch == 0)
        voxel_feat_for_points_hand, hand_indices = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, return_indices=return_indices
        )
        voxel_feat_for_points_head, head_indices = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, return_indices=return_indices
        )

        # Temporal embeddings
        temporal_emb_hand = self.temporal_emb_table_hand[step_nums]  # [B, temporal_emb_dim]
        temporal_emb_head = self.temporal_emb_table_head[step_nums]
        temporal_emb_hand = temporal_emb_hand.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        temporal_emb_head = temporal_emb_head.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # Decoder for loss (cosine similarity)
        dec_hand = self.implicit_decoder(voxel_feat_for_points_hand, hand_coords_world_flat)
        cos_sim_hand = F.cosine_similarity(dec_hand, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head = self.implicit_decoder(voxel_feat_for_points_head, head_coords_world_flat)
        cos_sim_head = F.cosine_similarity(dec_head, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        total_cos_loss = self.clip_loss_coef * (cos_loss_hand + cos_loss_head)

        # Update voxel points if epoch == 0
        if return_indices:
            # Hand
            valid_mask_hand = hand_indices >= 0
            if valid_mask_hand.any():
                valid_hand_idx = hand_indices[valid_mask_hand]
                valid_hand_pts = hand_coords_world_flat[valid_mask_hand]
                self.hash_voxel.add_points(valid_hand_idx, valid_hand_pts)
                for idx in valid_hand_idx.detach().cpu().numpy():
                    self.used_voxel_idx_set.add(int(idx))

            # Head
            valid_mask_head = head_indices >= 0
            if valid_mask_head.any():
                valid_head_idx = head_indices[valid_mask_head]
                valid_head_pts = head_coords_world_flat[valid_mask_head]
                self.hash_voxel.add_points(valid_head_idx, valid_head_pts)
                for idx in valid_head_idx.detach().cpu().numpy():
                    self.used_voxel_idx_set.add(int(idx))

        # Transformer input
        voxel_feat_for_points_hand_proj = self.voxel_proj(voxel_feat_for_points_hand)
        voxel_feat_for_points_head_proj = self.voxel_proj(voxel_feat_for_points_head)

        voxel_feat_for_points_hand_batched = voxel_feat_for_points_hand_proj.view(B, N, -1)
        voxel_feat_for_points_head_batched = voxel_feat_for_points_head_proj.view(B, N, -1)

        # Fuse voxel features and reduced CLIP features
        fused_hand = self.feature_fusion(voxel_feat_for_points_hand_batched, feats_hand_reduced)
        fused_head = self.feature_fusion(voxel_feat_for_points_head_batched, feats_head_reduced)

        # Select text embedding
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]

        batch_hand_coords = hand_coords_world_flat.view(B, N, 3)
        batch_head_coords = head_coords_world_flat.view(B, N, 3)

        # Pass through Transformer
        visual_token = self.transformer(
            hand=fused_hand,
            head=fused_head,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state,
            text_embeddings=selected_text_reduced
        )

        # Final action MLP
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)  # [B, state_mlp_dim * 2]
        action_pred = self.action_mlp(inp)                   # [B, action_dim]

        return action_pred, total_cos_loss
    
class Agent_point_dynamic(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        # Configuration parameters
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        state_mlp_dim: int = 1024,
        voxel_feature_dim: int = 120,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        max_time_steps: int = 201,
        temporal_emb_dim: int = 12,
        clip_loss_coef: float = 0.01,
        hash_voxel: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
    ):
        """
        Agent that maintains a voxel hash representation for 3D scenes and uses
        CLIP-based visual features together with an implicit decoder for mapping.
        """
        super().__init__()

        if text_input:
            text_input += [""]

        self.device = device
        self.clip_loss_coef = clip_loss_coef
        self.epoch = 0

        # State dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # State MLP (for the raw state vector)
        self.state_mlp = nn.Linear(state_dim, state_mlp_dim).to(self.device)

        # CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(open_clip_model[0])

        # Text
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            # Subtract the last embedding to avoid repetition in the blank token
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            text_embeddings = text_embeddings - redundant_emb
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        # Projection to reduce CLIP feature dim
        self.clip_dim_reducer = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)

        # Transformer
        self.transformer = TransformerEncoder(input_dim=voxel_feature_dim, hidden_dim=256, num_layers=2, num_heads=8, output_dim=state_mlp_dim)  

        # Action MLP is now a separate class
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hash table
        self.hash_voxel = hash_voxel
        self.implicit_decoder = implicit_decoder
   
        self.used_voxel_idx_set = set()
        self.voxel_grid = None
        self.voxel_points_dict = None

        # Time embeddings
        self.temporal_emb_table_hand = nn.Parameter(
            torch.randn(max_time_steps, temporal_emb_dim, device=self.device) * 0.01
        )
        self.temporal_emb_table_head = nn.Parameter(
            torch.randn(max_time_steps, temporal_emb_dim, device=self.device) * 0.01
        )

        # Additional transforms for voxel features
        self.voxel_proj = nn.Linear(voxel_feature_dim, voxel_feature_dim).to(self.device)
        self.feature_fusion = LocalSelfAttentionFusion(feat_dim=voxel_feature_dim)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

    def get_or_build_voxel_grid(self):
        """Return a dict of (ix, iy, iz) -> voxel_feature for used voxels."""
        if self.voxel_grid is not None:
            return self.voxel_grid

        voxel_grid = {}
        for i_voxel in self.used_voxel_idx_set:
            coords_3d = self.hash_voxel.voxel_coords[i_voxel]
            ix, iy, iz = torch.floor(coords_3d / self.hash_voxel.resolution).to(torch.int64)
            voxel_grid[(ix.item(), iy.item(), iz.item())] = self.hash_voxel.voxel_features[i_voxel]
        self.voxel_grid = voxel_grid
        return voxel_grid

    def get_or_build_voxel_points_dict(self):
        """
        Return a dict of (ix, iy, iz) -> [points].
        Each entry has up to a certain number of 3D points that fell into that voxel.
        """
        if self.voxel_points_dict is not None:
            return self.voxel_points_dict

        voxel_points_dict = {}
        for v_idx, points_list in self.hash_voxel.voxel_points.items():
            if v_idx < 0:
                continue
            coords_3d = self.hash_voxel.voxel_coords[v_idx]
            ix, iy, iz = torch.floor(coords_3d / self.hash_voxel.resolution).to(torch.int64)
            voxel_points_dict[(ix.item(), iy.item(), iz.item())] = points_list
        self.voxel_points_dict = voxel_points_dict
        return voxel_points_dict

    def forward(self, observations, object_labels, step_nums):
        """
        1. Process RGB/Depth from observations.
        2. Extract CLIP features.
        3. Look up voxel features.
        4. Compute reconstruction loss (cosine similarity).
        5. Fuse features, pass through Transformer, and then Action MLP.
        """
        # Unpack inputs
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]  # [B, state_dim]

        # Preprocess RGB
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

        # Depth
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")

        # Camera poses
        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        # Freeze CLIP feature extraction
        with torch.no_grad():
            hand_visfeat = get_visual_features(self.clip_model, hand_rgb)  # [B, clip_input_dim, Hf, Wf]
            head_visfeat = get_visual_features(self.clip_model, head_rgb)

        # 3D world coords
        hand_coords_world, hand_coords_cam = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, head_coords_cam = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        # Flatten coordinates
        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # Flatten CLIP features
        hand_visfeat = hand_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        head_visfeat = head_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)

        feats_hand_flat = hand_visfeat.reshape(B_ * N, -1)
        feats_head_flat = head_visfeat.reshape(B_ * N, -1)

        # Reduce feature dim
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat)  # [B*N, voxel_feature_dim]
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat)

        feats_hand_reduced = feats_hand_flat_reduced.reshape(B_, N, -1)  # [B, N, voxel_feature_dim]
        feats_head_reduced = feats_head_flat_reduced.reshape(B_, N, -1)

        # Voxel lookup
        times_t = step_nums  # shape [B], we will broadcast to [B*N]
        times_t_expanded = times_t.unsqueeze(1).expand(-1, N).reshape(B_ * N)
        voxel_feat_for_points_hand, hand_indices = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, times_t_expanded)

        voxel_feat_for_points_head, head_indices = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, times_t_expanded)

        # Temporal embeddings
        temporal_emb_hand = self.temporal_emb_table_hand[step_nums]  # [B, temporal_emb_dim]
        temporal_emb_head = self.temporal_emb_table_head[step_nums]
        temporal_emb_hand = temporal_emb_hand.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        temporal_emb_head = temporal_emb_head.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # Decoder for loss (cosine similarity)
        dec_hand = self.implicit_decoder(voxel_feat_for_points_hand, hand_coords_world_flat)
        cos_sim_hand = F.cosine_similarity(dec_hand, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head = self.implicit_decoder(voxel_feat_for_points_head, head_coords_world_flat)
        cos_sim_head = F.cosine_similarity(dec_head, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        total_cos_loss = self.clip_loss_coef * (cos_loss_hand + cos_loss_head)

        # Transformer input
        voxel_feat_for_points_hand_proj = self.voxel_proj(voxel_feat_for_points_hand)
        voxel_feat_for_points_head_proj = self.voxel_proj(voxel_feat_for_points_head)

        voxel_feat_for_points_hand_batched = voxel_feat_for_points_hand_proj.view(B, N, -1)
        voxel_feat_for_points_head_batched = voxel_feat_for_points_head_proj.view(B, N, -1)

        # Fuse voxel features and reduced CLIP features
        fused_hand = self.feature_fusion(voxel_feat_for_points_hand_batched, feats_hand_reduced)
        fused_head = self.feature_fusion(voxel_feat_for_points_head_batched, feats_head_reduced)

        # Select text embedding
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]

        batch_hand_coords = hand_coords_world_flat.view(B, N, 3)
        batch_head_coords = head_coords_world_flat.view(B, N, 3)

        # Pass through Transformer
        visual_token = self.transformer(
            hand=fused_hand,
            head=fused_head,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state,
            text_embeddings=selected_text_reduced
        )

        # Final action MLP
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)  # [B, state_mlp_dim * 2]
        action_pred = self.action_mlp(inp)                   # [B, action_dim]

        return action_pred, total_cos_loss
