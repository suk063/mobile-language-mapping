import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module import ImplicitDecoder, ActionMLP, ConcatMLPFusion, TransformerEncoder, init_weights_kaiming
from ..mapper.mapper import VoxelHashTable
from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d

import open_clip

class PerceiverAttentionLayer(nn.Module):
    """
    Inputs:
      - q: [B, Q, dim]
      - k: [B, N, dim]
      - v: [B, N, dim]
      - coords_q:  [B, Q, 3] (positional coords for q)
      - coords_kv: [B, N, 3] (positional coords for k and v)
    """
    def __init__(self, dim: int, nhead: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords_q: torch.Tensor = None,
        coords_kv: torch.Tensor = None
    ) -> torch.Tensor:
        # (1) Apply rotary PE to Query if coords_q is provided
        if coords_q is not None:
            q = rotary_pe_3d(q, coords_q)

        # (2) Apply rotary PE to Key if coords_kv is provided
        if coords_kv is not None:
            k = rotary_pe_3d(k, coords_kv)
            # v remains unchanged (by design), but could also apply PE if desired

        # (3) Multi-head cross-attention
        attn_out, _ = self.attn(q, k, v)  # [B, Q, dim]

        # (4) Residual + LayerNorm
        x = self.ln1(q + attn_out)        # [B, Q, dim]

        # (5) Position-wise feedforward
        ffn_out = self.ffn(x)            # [B, Q, dim]

        # (6) Residual + LayerNorm
        x = self.ln2(x + ffn_out)        # [B, Q, dim]
        return x


class GlobalPerceiver(nn.Module):
    """
    - Query: state_projected -> shape [B, 1, hidden_dim]
    - Key/Value: derived from valid_feats -> implicit_decoder -> [B, N, hidden_dim]
    - coords_q: head_translation -> [B, 1, 3]
    - coords_kv: valid_coords -> [B, N, 3]
    """
    def __init__(
        self,
        hidden_dim: int = 120,
        nhead: int = 8,
        num_layers: int = 4,
        out_dim: int = 120,
        voxel_proj: nn.Module = None,
        num_learnable_tokens: int = 16,
    ):
        super().__init__()
        
        self.modality_embed_state = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_embed_learnable = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.hidden_dim = hidden_dim
        self.voxel_proj = voxel_proj

        # Build a stack of cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverAttentionLayer(dim=hidden_dim, nhead=nhead)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        self.num_learnable_tokens = num_learnable_tokens
        self.learnable_tokens = nn.Parameter(
            torch.zeros(1, num_learnable_tokens, hidden_dim)
        )
        
        self.apply(init_weights_kaiming)

    def forward(
        self,
        state: torch.Tensor,            # [B, state_dim]
        valid_coords: torch.Tensor,     # [B, N, 3]
        valid_feats: torch.Tensor       # [B, N, voxel_feature_dim]
    ) -> torch.Tensor:
        """
        Args:
            state:           [B, state_dim]
            head_translation [B, 3]
            valid_coords:    [B, N, 3]
            valid_feats:     [B, N, voxel_feature_dim]
        """
        B, N, _ = valid_feats.shape

        # (1) Query = [ state_token + learnable_tokens ]
        state_token = state.unsqueeze(1)  # [B, 1, hidden_dim]
        state_token = state_token + self.modality_embed_state
        
        learnable_tokens = (
            self.learnable_tokens + self.modality_embed_learnable
        ).repeat(B, 1, 1)  # [B, num_learnable_tokens, hidden_dim]
        q = torch.cat([state_token, learnable_tokens], dim=1)  # [B, 1 + num_learnable_tokens, hidden_dim]
        
        coords_state = torch.zeros(B, 1, 3, device=q.device) 
        coords_learnable = torch.zeros(B, self.num_learnable_tokens, 3, device=q.device)  # [B, num_learnable_tokens, 3]
        coords_q = torch.cat([coords_state, coords_learnable], dim=1) 
        
        feats_flat = valid_feats.reshape(B * N, -1)
        coords_flat = valid_coords.reshape(B * N, 3)
        
        k_flat, _ = self.voxel_proj(
            feats_flat, coords_flat, return_intermediate=True
        )

        k = k_flat.view(B, N, self.hidden_dim)
        v = k
        
        coords_kv = valid_coords

        # (5) Pass through Perceiver cross-attention layers
        x = q
        for layer in self.layers:
            x = layer(
                q=x,
                k=k,
                v=v,
                coords_q=coords_q,
                coords_kv=coords_kv
            )

        # (6) Final projection
        out = self.out_proj(x[:, 1:, :])  # [B, out_dim]

        return out
    
class Agent_static_global(nn.Module):
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
        hash_voxel: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
        voxel_proj: ImplicitDecoder = None,
        global_k: int = 1024,
        num_learnable_tokens: int = 16
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
        self.voxel_proj = voxel_proj
        
        self.state_proj_perceiver = nn.Linear(state_dim, voxel_feature_dim).to(self.device)
        self.state_proj = nn.Linear(state_dim, voxel_feature_dim).to(self.device)

        # Local feature fusion
        self.feature_fusion = ConcatMLPFusion(feat_dim=voxel_feature_dim)

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
            voxel_proj=self.voxel_proj,
            num_learnable_tokens = num_learnable_tokens
        ).to(device)

        self.global_k = global_k

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
        voxel_feat_points_hand, _ = self.hash_voxel.query_voxel_feature(
            hand_coords_world_flat, return_indices=False
        )
        voxel_feat_points_head, _ = self.hash_voxel.query_voxel_feature(
            head_coords_world_flat, return_indices=False
        )

        # Implicit decoding and cosine loss
        dec_hand_final = self.implicit_decoder(
            voxel_feat_points_hand, hand_coords_world_flat, return_intermediate=False
        )
        cos_sim_hand = F.cosine_similarity(dec_hand_final, feats_hand_flat, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()

        dec_head_final = self.implicit_decoder(
            voxel_feat_points_head, head_coords_world_flat, return_intermediate=False
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

        # Find closest points
        state_projected_perceiver = self.state_proj_perceiver(state)
        head_translation = head_pose[:, 0, :3, 3] # [B, 3]
        
        valid_coords, valid_feats = self.hash_voxel.get_all_valid_voxel_data()
        
        valid_coords_exp = valid_coords.unsqueeze(0).expand(B_, -1, 3)
        dist = torch.norm(valid_coords_exp - head_translation.unsqueeze(1), dim=-1)
        K = self.global_k
        _, topk_indices = torch.topk(dist, k=K, dim=-1, largest=False)
        coords_kv = torch.gather(valid_coords_exp, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 3))
        coords_kv_flat = coords_kv.view(B_*K, 3)
        
        # selected_dist = torch.gather(dist, 1, topk_indices)  # [B_, K]
        # max_dist_per_batch = selected_dist.max(dim=1).values # [B_]

        # print(max_dist_per_batch)
                
        feats_kv_flat, _ = self.hash_voxel.query_voxel_feature(coords_kv_flat, return_indices=False)
        feats_kv = feats_kv_flat.view(B_, K, -1)

        global_token = self.state_perceiver(
            state_projected_perceiver,
            coords_kv,
            feats_kv
        )

        # Query voxel features 
        with torch.no_grad():
            voxel_feat_points_hand, _ = self.hash_voxel.query_voxel_feature(hand_coords_world_flat, return_indices=False)
            voxel_feat_points_head, _ = self.hash_voxel.query_voxel_feature(head_coords_world_flat, return_indices=False)

        voxel_feat_points_hand_projected, _ = self.implicit_decoder(voxel_feat_points_hand, hand_coords_world_flat, return_intermediate=True)
        voxel_feat_points_head_projected, _ = self.implicit_decoder(voxel_feat_points_head, head_coords_world_flat, return_intermediate=True)

        # Fuse voxel and CLIP features
        fused_hand = self.feature_fusion(
            feats_hand_flat_reduced,
            voxel_feat_points_hand_projected,
            hand_coords_world_flat
        )
        fused_head = self.feature_fusion(
            feats_head_flat_reduced,
            voxel_feat_points_head_projected,
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

        state_projected = self.state_proj(state)

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

        return action_pred