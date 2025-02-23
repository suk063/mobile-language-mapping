import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from scipy.optimize import linear_sum_assignment

# Local imports
from ..module import TransformerEncoder, LocalSelfAttentionFusion, ActionMLP, ImplicitDecoder
from ..mapper.mapper import VoxelHashTable
from ..utils import get_3d_coordinates, get_visual_features, chamfer_3d_weighted, chamfer_cosine_weighted, chamfer_cosine_coverage_loss, transform

import open_clip
    
class Agent_point_dynamic_flow(nn.Module):
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

        self.flow_embed = nn.Linear(2 * 3 * 256, state_mlp_dim).to(self.device)

        # Action MLP is now a separate class
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 3,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hash table
        self.hash_voxel = hash_voxel
        self.implicit_decoder = implicit_decoder
   
        self.used_voxel_idx_set = set()
        self.voxel_grid = None
        self.voxel_points_dict = None

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

    def forward_train(self, observations, object_labels, step_nums):
        """
        1. Process RGB/Depth from observations.
        2. Extract CLIP features.
        3. Look up voxel features.
        4. Compute reconstruction loss (cosine similarity) for time t AND t+1(p1).
        5. Fuse features, pass through Transformer, and then Action MLP.
        6. Compute scene flow loss.
        """
        # ------------------------------------------------------
        # Unpack inputs
        # ------------------------------------------------------
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]  # [B, state_dim]

        # Depth at t+1
        hand_depth_p1 = pixels["fetch_hand_depth_p1"] / 1000.0
        head_depth_p1 = pixels["fetch_head_depth_p1"] / 1000.0
        hand_pose_p1 = pixels["fetch_hand_pose_p1"]
        head_pose_p1 = pixels["fetch_head_pose_p1"]

        # Preprocess RGB at time t
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

        # Depth at time t
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")

            hand_depth_p1 = hand_depth_p1.view(b2, fs2 * d2, h2, w2)
            head_depth_p1 = head_depth_p1.view(b2, fs2 * d2, h2, w2)
            hand_depth_p1 = F.interpolate(hand_depth_p1, (16, 16), mode="nearest")
            head_depth_p1 = F.interpolate(head_depth_p1, (16, 16), mode="nearest")

        # Camera poses at time t
        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        # ------------------------------------------------------
        # (추가) RGB at time t+1 (p1)
        # ------------------------------------------------------
        hand_rgb_p1 = pixels["fetch_hand_rgb_p1"]
        head_rgb_p1 = pixels["fetch_head_rgb_p1"]

        if hand_rgb_p1.shape[2] != 3:
            hand_rgb_p1 = hand_rgb_p1.permute(0, 1, 4, 2, 3)
            head_rgb_p1 = head_rgb_p1.permute(0, 1, 4, 2, 3)

        B_p1, fs_p1, d_p1, H_p1, W_p1 = hand_rgb_p1.shape
        hand_rgb_p1 = hand_rgb_p1.reshape(B_p1, fs_p1 * d_p1, H_p1, W_p1)
        head_rgb_p1 = head_rgb_p1.reshape(B_p1, fs_p1 * d_p1, H_p1, W_p1)

        hand_rgb_p1 = transform(hand_rgb_p1.float() / 255.0)
        head_rgb_p1 = transform(head_rgb_p1.float() / 255.0)

        # ------------------------------------------------------
        # Extract CLIP visual features
        # ------------------------------------------------------
        with torch.no_grad():
            # time t
            hand_visfeat = get_visual_features(self.clip_model, hand_rgb)  # [B, clip_input_dim, Hf, Wf]
            head_visfeat = get_visual_features(self.clip_model, head_rgb)
            # time t+1
            hand_visfeat_p1 = get_visual_features(self.clip_model, hand_rgb_p1)
            head_visfeat_p1 = get_visual_features(self.clip_model, head_rgb_p1)

        # ------------------------------------------------------
        # 3D world coords
        # ------------------------------------------------------
        # time t
        hand_coords_world, hand_coords_cam = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, head_coords_cam = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        # time t+1
        hand_coords_world_p1, hand_coords_cam_p1 = get_3d_coordinates(
            hand_visfeat_p1, hand_depth_p1, hand_pose_p1, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_p1, head_coords_cam_p1 = get_3d_coordinates(
            head_visfeat_p1, head_depth_p1, head_pose_p1, self.fx, self.fy, self.cx, self.cy
        )

        # ------------------------------------------------------
        # Flattening
        # ------------------------------------------------------
        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        # t
        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # t+1
        hand_coords_world_flat_p1 = hand_coords_world_p1.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat_p1 = head_coords_world_p1.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # ------------------------------------------------------
        # Flatten CLIP features
        # ------------------------------------------------------
        # t
        hand_visfeat = hand_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        head_visfeat = head_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        feats_hand_flat = hand_visfeat.reshape(B_ * N, -1)
        feats_head_flat = head_visfeat.reshape(B_ * N, -1)

        # t+1
        hand_visfeat_p1 = hand_visfeat_p1.permute(0, 2, 3, 1).reshape(B_, N, -1)
        head_visfeat_p1 = head_visfeat_p1.permute(0, 2, 3, 1).reshape(B_, N, -1)
        feats_hand_flat_p1 = hand_visfeat_p1.reshape(B_ * N, -1)
        feats_head_flat_p1 = head_visfeat_p1.reshape(B_ * N, -1)

        # ------------------------------------------------------
        # Reduce feature dim
        # ------------------------------------------------------
        # t
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat)  # [B*N, voxel_feature_dim]
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat)
        feats_hand_reduced = feats_hand_flat_reduced.reshape(B_, N, -1)
        feats_head_reduced = feats_head_flat_reduced.reshape(B_, N, -1)

        # t+1
        feats_hand_flat_reduced_p1 = self.clip_dim_reducer(feats_hand_flat_p1)
        feats_head_flat_reduced_p1 = self.clip_dim_reducer(feats_head_flat_p1)
        feats_hand_reduced_p1 = feats_hand_flat_reduced_p1.reshape(B_, N, -1)
        feats_head_reduced_p1 = feats_head_flat_reduced_p1.reshape(B_, N, -1)

        # ------------------------------------------------------
        # Voxel lookup (time t)
        # ------------------------------------------------------
        times_t_expanded = step_nums.unsqueeze(1).expand(-1, N).reshape(B_ * N)
        voxel_feat_for_points_hand, hand_indices = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, times_t_expanded
        )
        voxel_feat_for_points_head, head_indices = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, times_t_expanded
        )

        # ------------------------------------------------------
        # Decoder at time t for cos_loss
        # ------------------------------------------------------
        dec_hand = self.implicit_decoder(voxel_feat_for_points_hand, hand_coords_world_flat)
        cos_sim_hand = F.cosine_similarity(dec_hand, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head = self.implicit_decoder(voxel_feat_for_points_head, head_coords_world_flat)
        cos_sim_head = F.cosine_similarity(dec_head, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        # ------------------------------------------------------
        # Voxel lookup (time t+1)
        # ------------------------------------------------------
        step_nums_p1 = step_nums + 1
        times_tp1_expanded = step_nums_p1.unsqueeze(1).expand(-1, N).reshape(B_ * N)

        voxel_feat_for_points_hand_p1, hand_indices_p1 = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat_p1, times_tp1_expanded
        )
        voxel_feat_for_points_head_p1, head_indices_p1 = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat_p1, times_tp1_expanded
        )

        # ------------------------------------------------------
        # Decoder at time t+1 for cos_loss
        # ------------------------------------------------------
        dec_hand_p1 = self.implicit_decoder(voxel_feat_for_points_hand_p1, hand_coords_world_flat_p1)
        cos_sim_hand_p1 = F.cosine_similarity(dec_hand_p1, feats_hand_flat_p1, dim=-1)
        cos_loss_hand_p1 = 1.0 - cos_sim_hand_p1.mean()

        dec_head_p1 = self.implicit_decoder(voxel_feat_for_points_head_p1, head_coords_world_flat_p1)
        cos_sim_head_p1 = F.cosine_similarity(dec_head_p1, feats_head_flat_p1, dim=-1)
        cos_loss_head_p1 = 1.0 - cos_sim_head_p1.mean()

        total_cos_loss = (cos_loss_hand + cos_loss_head +
                          cos_loss_hand_p1 + cos_loss_head_p1)

        # -------------------------------------------------
        # Scene flow prediction & Chamfer loss
        # -------------------------------------------------
        flow_hand = self.hash_voxel.query_scene_flow(hand_coords_world_flat, times_t_expanded)  # [N,3]
        flow_head = self.hash_voxel.query_scene_flow(head_coords_world_flat, times_t_expanded)

        pred_hand_next = hand_coords_world_flat + flow_hand
        pred_head_next = head_coords_world_flat + flow_head

        pred_hand_next_b = pred_hand_next.view(B_, N, 3)
        pred_head_next_b = pred_head_next.view(B_, N, 3)
        hand_coords_world_flat_p1_b = hand_coords_world_flat_p1.view(B_, N, 3)
        head_coords_world_flat_p1_b = head_coords_world_flat_p1.view(B_, N, 3)

        #    shape: (B*N,) -> then reshape to (B, N)
        hand_var_flat = self.hash_voxel.get_variance_for_points(pred_hand_next)  # (B*N,)
        head_var_flat = self.hash_voxel.get_variance_for_points(pred_head_next)  # (B*N,)

        hand_var_b = hand_var_flat.view(B_, N)
        head_var_b = head_var_flat.view(B_, N)

        hand_chamfer_loss = chamfer_3d_weighted(
            pred_points=pred_hand_next_b, 
            gt_points=hand_coords_world_flat_p1_b, 
            pred_weights=hand_var_b,
            threshold=1.0
        )
        head_chamfer_loss = chamfer_3d_weighted(
            pred_points=pred_head_next_b, 
            gt_points=head_coords_world_flat_p1_b, 
            pred_weights=head_var_b,
            threshold=1.0
        )

        scene_flow_loss = hand_chamfer_loss + head_chamfer_loss

        voxel_feat_tp1_pred_hand, _ = self.hash_voxel.query_voxel_feature(
            pred_hand_next, times_tp1_expanded
        )
        voxel_feat_tp1_pred_head, _ = self.hash_voxel.query_voxel_feature(
            pred_head_next, times_tp1_expanded
        )

        voxel_feat_for_points_hand_p1 = voxel_feat_for_points_hand_p1.detach()
        voxel_feat_for_points_head_p1 = voxel_feat_for_points_head_p1.detach()
        voxel_feat_tp1_pred_hand = voxel_feat_tp1_pred_hand.detach()
        voxel_feat_tp1_pred_head = voxel_feat_tp1_pred_head.detach()

        # Reshape to [B, N, feature_dim]
        voxel_feat_tp1_pred_hand_b = voxel_feat_tp1_pred_hand.view(B_, N, -1)
        voxel_feat_tp1_pred_head_b = voxel_feat_tp1_pred_head.view(B_, N, -1)

        # Ground truth (actual) t+1 features already exist in: voxel_feat_for_points_hand_p1, voxel_feat_for_points_head_p1
        voxel_feat_for_points_hand_p1_b = voxel_feat_for_points_hand_p1.view(B_, N, -1)
        voxel_feat_for_points_head_p1_b = voxel_feat_for_points_head_p1.view(B_, N, -1)

        # Use the same variance-based weights for predicted side or define new ones if needed
        hand_feat_var_flat = self.hash_voxel.get_variance_for_points(pred_hand_next)  # (B*N,)
        head_feat_var_flat = self.hash_voxel.get_variance_for_points(pred_head_next)  # (B*N,)
        hand_feat_var_b = hand_feat_var_flat.view(B_, N)
        head_feat_var_b = head_feat_var_flat.view(B_, N)

        # Compute Chamfer Cosine (Weighted)
        hand_feat_chamfer_loss = chamfer_cosine_weighted(
            pred_feat=voxel_feat_tp1_pred_hand_b,
            gt_feat=voxel_feat_for_points_hand_p1_b,
            pred_weights=hand_feat_var_b,
<<<<<<< HEAD
            threshold=0.1  
=======
            threshold=0.05  
>>>>>>> d186eee95cfc40c3dce985ca59f5c6953c0b4d6c
        )
        head_feat_chamfer_loss = chamfer_cosine_weighted(
            pred_feat=voxel_feat_tp1_pred_head_b,
            gt_feat=voxel_feat_for_points_head_p1_b,
            pred_weights=head_feat_var_b,
<<<<<<< HEAD
            threshold=0.1
=======
            threshold=0.05
>>>>>>> d186eee95cfc40c3dce985ca59f5c6953c0b4d6c
        )

        # dec_hand_pred_p1 = self.implicit_decoder(voxel_feat_tp1_pred_hand, pred_hand_next)
        # dec_head_pred_p1 = self.implicit_decoder(voxel_feat_tp1_pred_head, pred_head_next)

        # cos_sim_hand_pred_p1 = F.cosine_similarity(dec_hand_pred_p1, feats_hand_flat_p1, dim=-1)
        # cos_sim_head_pred_p1 = F.cosine_similarity(dec_head_pred_p1, feats_head_flat_p1, dim=-1)
        # pred_cos_loss_hand_p1 = 1.0 - cos_sim_hand_pred_p1.mean()
        # pred_cos_loss_head_p1 = 1.0 - cos_sim_head_pred_p1.mean()

        scene_flow_cos_loss = hand_feat_chamfer_loss + head_feat_chamfer_loss

        # -------------------------------------------------
        # Prepare flow_emb to feed into Action MLP
        # -------------------------------------------------
        flow_hand_batched = flow_hand.view(B_, N, 3)
        flow_head_batched = flow_head.view(B_, N, 3)
        flow_hand_flat = flow_hand_batched.view(B_, -1)  # (B, N*3)
        flow_head_flat = flow_head_batched.view(B_, -1)  # (B, N*3)
        flow_cat = torch.cat([flow_hand_flat, flow_head_flat], dim=1)  # (B, 2*N*3)

        flow_emb = self.flow_embed(flow_cat)  # [B, state_mlp_dim]

        # -------------------------------------------------
        # Transformer input
        # -------------------------------------------------
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
        inp = torch.cat([visual_token, state_token, flow_emb], dim=1)  # [B, state_mlp_dim * 3]
        action_pred = self.action_mlp(inp)  # [B, action_dim]

        return action_pred, total_cos_loss, scene_flow_loss, scene_flow_cos_loss
    
    def forward_eval(self, observations, object_labels, step_nums):
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
        times_t_expanded = step_nums.unsqueeze(1).expand(-1, N).reshape(B_ * N)
        voxel_feat_for_points_hand, hand_indices = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, times_t_expanded)

        voxel_feat_for_points_head, head_indices = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, times_t_expanded)

        # ============ 5) Compute scene flow loss ============
        # Predicted flow v at time t
        flow_hand = self.hash_voxel.query_scene_flow(hand_coords_world_flat, times_t_expanded)  # [N,3]
        flow_head = self.hash_voxel.query_scene_flow(head_coords_world_flat, times_t_expanded)

        # reshape flow_hand, flow_head each to (B, N*3), then concat -> (B, 2*N*3)
        flow_hand_batched = flow_hand.view(B_, N, 3)
        flow_head_batched = flow_head.view(B_, N, 3)
        flow_hand_flat = flow_hand_batched.view(B_, -1)  # (B, N*3)
        flow_head_flat = flow_head_batched.view(B_, -1)  # (B, N*3)
        flow_cat = torch.cat([flow_hand_flat, flow_head_flat], dim=1)  # (B, 2*N*3=1536 if N=256)

        flow_emb = self.flow_embed(flow_cat)  # [B, state_mlp_dim]

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
        inp = torch.cat([visual_token, state_token, flow_emb], dim=1)  # [B, state_mlp_dim * 2]
        action_pred = self.action_mlp(inp)                   # [B, action_dim]

        return action_pred