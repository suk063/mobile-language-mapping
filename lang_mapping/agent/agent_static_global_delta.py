import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module.transformer import TransformerEncoder, GlobalPerceiver
from ..module.mlp import ActionMLP, ImplicitDecoder, ConcatMLPFusion
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from ..utils import get_3d_coordinates, get_visual_features, transform

import open_clip
    
class Agent_static_global_delta(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        voxel_feature_dim: int = 128,
        state_mlp_dim: int = 1024,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: VoxelHashTable = None,
        delta_map: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
        global_k: int = 1024,
        num_learnable_tokens: int = 16,
        episode_num: int = 244,
    ):
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

        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            output_dim=state_mlp_dim,
            num_token=512+num_learnable_tokens
        )

        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.delta_map = delta_map
        
        self.implicit_decoder = implicit_decoder

        self.state_proj = nn.Linear(state_dim, voxel_feature_dim).to(self.device)

        # Local feature fusion
        self.fusion_image = ConcatMLPFusion(feat_dim=voxel_feature_dim, L=10)
        
        self.fusion_subtask = ConcatMLPFusion(feat_dim=voxel_feature_dim, L=10)
        self.fusion_subtask_global = ConcatMLPFusion(feat_dim=voxel_feature_dim)
        
        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        # self.register_buffer(
        #     "used_voxel_idx", 
        #     torch.from_numpy(np.load("used_voxel_idx.npy")).float().to(device).unsqueeze(0)
        #     )
        
        self.state_perceiver = GlobalPerceiver(
            hidden_dim=voxel_feature_dim,
            nhead=8,
            num_layers=4,
            out_dim=voxel_feature_dim,
            voxel_proj=self.implicit_decoder,
            num_learnable_tokens = num_learnable_tokens
        ).to(device)
         
        # Time embeddings for conditioning
        self.subtask_embedding = nn.Parameter(
            torch.randn(episode_num, voxel_feature_dim, device=device) * 0.01
        )

        self.global_k = global_k

    def forward(self, observations, object_labels, subtask_idx):
        """
        Stage 1: Learn voxel and implicit decoder only. Returns total_cos_loss.
        CLIP is frozen (with torch.no_grad).
        """
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]

        # Extract CLIP features without gradient
        with torch.no_grad():
            hand_rgb = pixels["fetch_hand_rgb"]
            head_rgb = pixels["fetch_head_rgb"]
            # Reshape to (B, C, H, W)
            if hand_rgb.shape[2] != 3:
                hand_rgb = hand_rgb.permute(0, 1, 4, 2, 3)
                head_rgb = head_rgb.permute(0, 1, 4, 2, 3)
            B, fs, d, H, W = hand_rgb.shape
            hand_rgb = hand_rgb.reshape(B, fs * d, H, W)
            head_rgb = head_rgb.reshape(B, fs * d, H, W)

            # Normalize RGB
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

            hand_visfeat = get_visual_features(self.clip_model, hand_rgb)
            head_visfeat = get_visual_features(self.clip_model, head_rgb)

        # Compute 3D world coordinates
        hand_coords_world, _ = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, _ = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        # Flatten coordinates
        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # Flatten CLIP features 
        with torch.no_grad():
            hand_visfeat = hand_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
            head_visfeat = head_visfeat.permute(0, 2, 3, 1).reshape(B_, N, -1)
        feats_hand_flat = hand_visfeat.reshape(B_ * N, -1)
        feats_head_flat = head_visfeat.reshape(B_ * N, -1)

        # Voxel features
        voxel_feat_points_hand_static, _ = self.static_map.query_voxel_feature(
            hand_coords_world_flat, return_indices=False
        )
        voxel_feat_points_head_static, _ = self.static_map.query_voxel_feature(
            head_coords_world_flat, return_indices=False
        )
        
        voxel_feat_points_hand_delta, _ = self.delta_map.query_voxel_feature(
            hand_coords_world_flat, return_indices=False
        )
        voxel_feat_points_head_delta, _ = self.delta_map.query_voxel_feature(
            head_coords_world_flat, return_indices=False
        )
        
        subtask_embedding = self.subtask_embedding[subtask_idx, :].unsqueeze(1).expand(-1, N, -1)
        subtask_embedding_flat = subtask_embedding.reshape(B_*N, -1)
        
        voxel_feat_points_hand_delta_subtask = self.fusion_subtask(voxel_feat_points_hand_delta, subtask_embedding_flat, hand_coords_world_flat)
        voxel_feat_points_head_delta_subtask = self.fusion_subtask(voxel_feat_points_head_delta, subtask_embedding_flat, head_coords_world_flat)
        
        voxel_feat_points_hand = voxel_feat_points_hand_static + voxel_feat_points_hand_delta_subtask
        voxel_feat_points_head = voxel_feat_points_head_static + voxel_feat_points_head_delta_subtask

        # Implicit decoding and cosine loss
        dec_hand_projected, dec_hand_final = self.implicit_decoder(
            voxel_feat_points_hand, hand_coords_world_flat, return_intermediate=True
        )
        cos_sim_hand = F.cosine_similarity(dec_hand_final, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head_projected, dec_head_final = self.implicit_decoder(
            voxel_feat_points_head, head_coords_world_flat, return_intermediate=True
        )
        cos_sim_head = F.cosine_similarity(dec_head_final, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        total_cos_loss = cos_loss_hand + cos_loss_head
        
        ###################################################################################

        # Find closest points
        state_projected = self.state_proj(state)
        head_translation = head_pose[:, 0, :3, 3] # [B, 3]
        
        valid_coords = self.static_map.stored_coords
        
        valid_coords_exp = valid_coords.unsqueeze(0).expand(B_, -1, 3)
        dist = torch.norm(valid_coords_exp - head_translation.unsqueeze(1), dim=-1)
        K = self.global_k
        _, topk_indices = torch.topk(dist, k=K, dim=-1, largest=False)
        coords_kv = torch.gather(valid_coords_exp, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 3))
        coords_kv_flat = coords_kv.view(B_*K, 3)
    
        # selected_dist = torch.gather(dist, 1, topk_indices)  # [B_, K]
        # max_dist_per_batch = selected_dist.max(dim=1).values # [B_]

        # print(max_dist_per_batch)
        
        feats_kv_flat_static, _ = self.static_map.query_voxel_feature(coords_kv_flat, return_indices=False)
        feats_kv_flat_delta, _ = self.delta_map.query_voxel_feature(coords_kv_flat, return_indices=False)
        
        subtask_embedding = self.subtask_embedding[subtask_idx, :].unsqueeze(1).expand(-1, K, -1)
        subtask_embedding_flat = subtask_embedding.reshape(B_* K, -1)
        
        feats_kv_delta_subtask_flat = self.fusion_subtask_global(feats_kv_flat_delta, subtask_embedding_flat)
        feats_kv_delta_subtask = feats_kv_delta_subtask_flat.view(B_, K, -1)
        
        feats_kv_static = feats_kv_flat_static.view(B_, K, -1)
        
        feats_kv = feats_kv_static + feats_kv_delta_subtask

        global_token = self.state_perceiver(
            state_projected,
            coords_kv,
            feats_kv 
        )
        
        # Reduce CLIP dimension 
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat)
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat)

        # Fuse voxel and CLIP features
        fused_hand = self.fusion_image(
            feats_hand_flat_reduced,
            dec_hand_projected,
            hand_coords_world_flat
        )
        fused_head = self.fusion_image(
            feats_head_flat_reduced,
            dec_head_projected,
            head_coords_world_flat
        )

        # Get text embeddings (projected)
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]

        # Prepare coords for transformer
        batch_hand_coords = hand_coords_world_flat.view(B_, N, 3)
        batch_head_coords = head_coords_world_flat.view(B_, N, 3)

        batch_fused_hand = fused_hand.view(B_, N, -1)
        batch_fused_head = fused_head.view(B_, N, -1)

        # Transformer
        visual_token = self.transformer(
            hand=batch_fused_hand,
            head=batch_fused_head,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state_projected,
            text_embeddings=selected_text_reduced,
            global_token=global_token
        )

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return total_cos_loss, action_pred
    