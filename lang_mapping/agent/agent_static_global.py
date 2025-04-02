import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module import *
from ..mapper.mapper import VoxelHashTable
from ..utils import get_3d_coordinates, get_visual_features, positional_encoding, transform

import open_clip

class GlobalPointsPointNet(nn.Module):
    def __init__(
        self,
        d_model=120,       
        hidden_dim=256,
        final_out_dim=1024,  
        pe_freq_L=10,         
    ):
        super().__init__()
        
        self.pe_freq_L = pe_freq_L

        self.mlp1 = nn.Sequential(
            nn.Linear(d_model + 6 * pe_freq_L, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 전역 풀링 후 후처리
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, final_out_dim),
        )

    def forward(self, voxel_feat_global_points, coords_global_points):
        """
        Args:
            state_token: [B, d_model=120] (옵션, 여기서는 직접 사용 안 함)
            voxel_feat_global_points: [B, P, 120]
            coords_global_points: [B, P, 3]

        Returns:
            out: [B, final_out_dim=1024]
        """
        B, P, D = voxel_feat_global_points.shape
        
        encoded_coords = positional_encoding(coords_global_points, L=self.pe_freq_L)

        x = torch.cat([voxel_feat_global_points, encoded_coords], dim=-1)
        
        x = self.mlp1(x)  # (B, P, hidden_dim)
        
        x = torch.max(x, dim=1)[0]  # (B, hidden_dim)
        
        out = self.post_fusion_mlp(x)  # (B, final_out_dim=1024)

        return out

class Agent_static_global(nn.Module):
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
        hash_voxel: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
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

        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            output_dim=state_mlp_dim
        )

        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 3,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.hash_voxel = hash_voxel
        self.implicit_decoder = implicit_decoder

        self.voxel_proj = VoxelProj(voxel_feature_dim=voxel_feature_dim).to(self.device)

        # Local feature fusion
        self.feature_fusion = ConcatMLPFusion(feat_dim=voxel_feature_dim)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.collected_points = []
        
        self.points = torch.from_numpy(np.load("points.npy")).float().to(device).unsqueeze(0)
        
        self.state_to_voxeldim = nn.Linear(42, voxel_feature_dim).to(self.device)
        
        self.global_pointnet = GlobalPointsPointNet(
            d_model=voxel_feature_dim,  # 120
            hidden_dim=256,
            final_out_dim=state_mlp_dim, 
            pe_freq_L=10
        ).to(self.device)


    def forward_mapping(self, observations):
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

        with torch.no_grad():
            # hand_depth: (B_, 16, 16) -> (B_*16*16)
            hand_depth_flat = hand_depth.view(B_ * N)
            head_depth_flat = head_depth.view(B_ * N)

            # depth >= 0.5
            mask_hand = (hand_depth_flat >= 0.5)
            mask_head = (head_depth_flat >= 0.5)

            valid_hand_coords = hand_coords_world_flat[mask_hand]
            valid_head_coords = head_coords_world_flat[mask_head]

            # 실제 저장
            if valid_hand_coords.shape[0] > 0:
                self.collected_points.append(valid_hand_coords.cpu().numpy())
            if valid_head_coords.shape[0] > 0:
                self.collected_points.append(valid_head_coords.cpu().numpy())

        # Voxel features
        voxel_feat_view_points_hand, _ = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, return_indices=False
        )
        voxel_feat_view_points_head, _ = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, return_indices=False
        )

        # Implicit decoding and cosine loss
        dec_hand_final = self.implicit_decoder(
            voxel_feat_view_points_hand, hand_coords_world_flat, return_intermediate=False
        )
        cos_sim_hand = F.cosine_similarity(dec_hand_final, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head_final = self.implicit_decoder(
            voxel_feat_view_points_head, head_coords_world_flat, return_intermediate=False
        )
        cos_sim_head = F.cosine_similarity(dec_head_final, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        total_cos_loss = cos_loss_hand + cos_loss_head
        return total_cos_loss

    def forward_policy(self, observations, object_labels, step_nums):
        """
        Stage 2: Use frozen voxel/implicit decoder and fine-tuned CLIP
        to predict actions (for BC loss).
        """
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

        # Query voxel features 
        with torch.no_grad():
            voxel_feat_view_points_hand, _ = self.hash_voxel.query_voxel_feature(
                hand_coords_world_flat, return_indices=False
            )
            voxel_feat_view_points_head, _ = self.hash_voxel.query_voxel_feature(
                head_coords_world_flat, return_indices=False
            )

        voxel_feat_view_points_hand = self.voxel_proj(voxel_feat_view_points_hand, hand_coords_world_flat)
        voxel_feat_view_points_head = self.voxel_proj(voxel_feat_view_points_head, head_coords_world_flat)
            
        # Fuse voxel and CLIP features
        fused_hand = self.feature_fusion(
            feats_hand_flat_reduced,
            voxel_feat_view_points_hand,
            hand_coords_world_flat
        )
        fused_head = self.feature_fusion(
            feats_head_flat_reduced,
            voxel_feat_view_points_head,
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

        state_voxel_dim = self.state_to_voxeldim(state)

        # Transformer
        visual_token = self.transformer(
            hand=batch_fused_hand,
            head=batch_fused_head,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state_voxel_dim,
            text_embeddings=selected_text_reduced
        )
        
        # Find closest points
        head_translations = head_pose[:, 0, :3, 3] # [B, 3]
        points_repeated = self.points.repeat(B_, 1, 1)  # [B, P, 3]

        diff = points_repeated - head_translations.unsqueeze(1)  # [B, P, 3]
        distances = torch.sum(diff * diff, dim=-1)  # [B, P]

        _, topk_indices = torch.topk(distances, k=4096, dim=1, largest=False)
        batch_indices = torch.arange(B_, device=distances.device).unsqueeze(-1)
        closest_points = points_repeated[batch_indices, topk_indices, :]  # [B, 1000, 3]

        closest_points_flatten = closest_points.reshape(B_ * 4096, 3)
        
        # Query voxel features 
        with torch.no_grad():
            voxel_feat_global_points, _ = self.hash_voxel.query_voxel_feature(
                closest_points_flatten, return_indices=False
            )

        voxel_feat_global_points = self.voxel_proj(voxel_feat_global_points, closest_points_flatten)
        voxel_feat_global_points = voxel_feat_global_points.view(B_, 4096, -1)

        coords_global_points = closest_points
        
        map_token_global = self.global_pointnet(
            voxel_feat_global_points,    # (B, 4096, 120)
            coords_global_points         # (B, 4096, 3)
        )
        
        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, map_token_global, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred
