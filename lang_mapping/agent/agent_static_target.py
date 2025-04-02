import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module import *
from ..mapper.mapper import VoxelHashTable
from ..utils import get_3d_coordinates, get_visual_features, positional_encoding, transform, rotary_pe_3d

import open_clip

class TransformerCrossAttentionLayer(nn.Module):
    """
    요청사항:
    1) Self-Attention (rotary PE 가능)
    2) Cross-Attention #1 (Q=src, K=V=text)
    3) Cross-Attention #2 (Q=src, K=V=target+initial)
    4) FeedForward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 1) Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 2) Cross-Attention (with text)
        self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 3) Cross-Attention (with target+initial)
        self.cross_attn_ti = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 4) FeedForward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorm + Dropout for each sub-block
        self.norm1 = nn.LayerNorm(d_model)  # after self-attn
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attn text
        self.norm3 = nn.LayerNorm(d_model)  # after cross-attn target+initial
        self.norm4 = nn.LayerNorm(d_model)  # after feedforward

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(
        self,
        src,
        text,
        target,
        initial,
        coords_src=None,
        coords_text=None,
        coords_target=None,
        coords_initial=None
    ):
        """
        Args:
            src: [B, N, d_model], query (self-attn용, 이후 cross-attn query)
            text: [B, N_text, d_model], cross-attn #1에서 key/value
            target: [B, N_t, d_model]
            initial: [B, N_i, d_model]
            coords_src: [B, N, 3], src용 3D 좌표(있으면 rotary PE)
            coords_text: [B, N_text, 3], text용 3D 좌표(있으면 rotary PE)
            coords_target: [B, N_t, 3]
            coords_initial: [B, N_i, 3]
        """
        # ------------------------------------------------------------------
        # 1) Self-Attention (+ rotary PE)
        # ------------------------------------------------------------------
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_ = src
        else:
            q_rot = k_rot = v_ = src

        src2, _ = self.self_attn(query=q_rot, key=k_rot, value=v_)
        src = self.norm1(src + self.dropout1(src2))

        # ------------------------------------------------------------------
        # 2) Cross-Attention with text (+ rotary PE in key/value)
        # ------------------------------------------------------------------
        if coords_text is not None:
            text_rot = rotary_pe_3d(text, coords_text)
        else:
            text_rot = text

        src2, _ = self.cross_attn_text(query=src, key=text_rot, value=text)
        src = self.norm2(src + self.dropout2(src2))

        # ------------------------------------------------------------------
        # 3) Cross-Attention with target + initial
        #    (+ rotary PE in key/value)
        # ------------------------------------------------------------------
        ti_cat = torch.cat([target, initial], dim=1)  # [B, N_t + N_i, d_model]
        if coords_target is not None and coords_initial is not None:
            coords_ti = torch.cat([coords_target, coords_initial], dim=1)  # [B, (N_t+N_i), 3]
            ti_cat_rot = rotary_pe_3d(ti_cat, coords_ti)
        else:
            ti_cat_rot = ti_cat

        src2, _ = self.cross_attn_ti(query=src, key=ti_cat_rot, value=ti_cat)
        src = self.norm3(src + self.dropout3(src2))

        # ------------------------------------------------------------------
        # 4) FeedForward
        # ------------------------------------------------------------------
        src2 = self.linear2(self.dropout_ff(F.gelu(self.linear1(src))))
        src = self.norm4(src + self.dropout4(src2))

        return src

class TransformerEncoder(nn.Module):
    """
    예시:
    - state, text, target, initial, hand, head 등을 모두 합쳐서 src로 둔 뒤
      self-attn을 먼저 수행.
    - 이후 cross-attn 내부에서 text, target, initial을 다시 별도 인자로 받아서
      rotary PE 적용 후 cross-attn. 
    """
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=2, num_heads=8, output_dim=1024):
        super().__init__()
        self.state_projection = nn.Linear(42, input_dim)
        self.layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])
        # 예시로 post_fusion_mlp 부분 수정 없이 유지
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * 256, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim)
        )

    def forward(
        self,
        target, 
        initial,
        target_coords,
        initial_coords,
        hand,
        head,
        coords_hand=None,
        coords_head=None,
        state=None,
        text_embeddings=None,
    ):
        """
        hand, head: [B, N, input_dim]
        coords_hand, coords_head: [B, N, 3]
        state: [B, 42]
        text_embeddings: [B, input_dim]
        """
        B, N, D = hand.shape

        # Project state into a single token
        state_token = self.state_projection(state).unsqueeze(1)  # [B,1,input_dim]
        coords_state = torch.zeros(B, 1, 3, device=state.device)

        # text_embeddings -> [B,1,input_dim]
        text_embeddings = text_embeddings.unsqueeze(1)
        coords_text = torch.zeros(B, 1, 3, device=state.device) 

        # self-attn용 src
        # [state_token, text_embeddings, target, initial, hand, head]
        src = torch.cat([state_token, text_embeddings, target, initial, hand, head], dim=1)  
        coords_src = torch.cat([coords_state, coords_text, target_coords, initial_coords, coords_hand, coords_head], dim=1)

        # cross-attn용(두 번째 블록)에서 사용할 target/initial 그대로 넘기기
        # rotary PE용 coords도 별도로 넘길 것
        for layer in self.layers:
            src = layer(
                src=src,
                text=text_embeddings,   # (B, 1, d_model) or (B, Nt, d_model)
                target=target,
                initial=initial,
                coords_src=coords_src,
                coords_text=coords_text,
                coords_target=target_coords,
                coords_initial=initial_coords
            )

        # Post-fusion MLP
        data = src[:, 4:, :].reshape(B, -1)
        
        out = self.post_fusion_mlp(data)
        return out

class Agent_static_target(nn.Module):
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
            input_dim=state_mlp_dim * 2,
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

        # Voxel features
        voxel_feat_for_points_hand, _ = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, return_indices=False
        )
        voxel_feat_for_points_head, _ = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, return_indices=False
        )

        # Implicit decoding and cosine loss
        dec_hand_final = self.implicit_decoder(
            voxel_feat_for_points_hand, hand_coords_world_flat, return_intermediate=False
        )
        cos_sim_hand = F.cosine_similarity(dec_hand_final, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head_final = self.implicit_decoder(
            voxel_feat_for_points_head, head_coords_world_flat, return_intermediate=False
        )
        cos_sim_head = F.cosine_similarity(dec_head_final, feats_head_flat, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()

        total_cos_loss = cos_loss_hand + cos_loss_head
        return total_cos_loss

    def forward_policy(self, observations, object_labels, step_nums, targets_pos, rigid_objs_pos):
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
            voxel_feat_for_points_hand, _ = self.hash_voxel.query_voxel_feature(
                hand_coords_world_flat, return_indices=False
            )
            voxel_feat_for_points_head, _ = self.hash_voxel.query_voxel_feature(
                head_coords_world_flat, return_indices=False
            )

        voxel_feat_for_points_hand = self.voxel_proj(voxel_feat_for_points_hand, hand_coords_world_flat)
        voxel_feat_for_points_head = self.voxel_proj(voxel_feat_for_points_head, head_coords_world_flat)

        # Fuse voxel and CLIP features
        fused_hand = self.feature_fusion(
            feats_hand_flat_reduced,
            voxel_feat_for_points_hand,
            hand_coords_world_flat
        )
        fused_head = self.feature_fusion(
            feats_head_flat_reduced,
            voxel_feat_for_points_head,
            head_coords_world_flat
        )
        
        with torch.no_grad():
            targets_voxel_feat, _ = self.hash_voxel.query_voxel_feature(
                targets_pos, return_indices=False
            )
            rigid_objs_voxel_feat, _ = self.hash_voxel.query_voxel_feature(
                rigid_objs_pos, return_indices=False
            )

        # (B, feat_dim) -> (B, 1, feat_dim) 
        targets_voxel_feat = targets_voxel_feat.unsqueeze(1)      # [B, 1, D]
        rigid_objs_voxel_feat = rigid_objs_voxel_feat.unsqueeze(1)# [B, 1, D]

        # (B, 3) -> (B, 1, 3) 
        coords_targets = targets_pos.unsqueeze(1)     # [B, 1, 3]
        coords_rigid_objs = rigid_objs_pos.unsqueeze(1)  # [B, 1, 3]

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
            target=targets_voxel_feat,
            initial=rigid_objs_voxel_feat,
            target_coords=coords_targets,
            initial_coords=coords_rigid_objs,
            hand=batch_fused_hand,
            head=batch_fused_head,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state,
            text_embeddings=selected_text_reduced
        )

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred
