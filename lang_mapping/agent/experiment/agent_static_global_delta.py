import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
# from ..module.transformer import TransformerEncoder, GlobalPerceiver
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, coords_src: torch.Tensor = None) -> torch.Tensor:
        # Self-attention, use rotary encoding if coords_src is provided
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_rot = src
        else:
            q_rot = k_rot = v_rot = src

        attn_out, _ = self.self_attn(q_rot, k_rot, v_rot)
        src = self.norm1(src + self.dropout1(attn_out))

        # Feed-forward
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm_ff(src + self.dropout_ff(ff_out))
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=120,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        output_dim=1024,
        prefix_len=16,
    ):
        super().__init__()
        
        self.prefix_len = prefix_len
        self.prefix = nn.Parameter(torch.randn(1, prefix_len, input_dim))

        self.modality_embed_image = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_3d = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Post-fusion MLP
        # self.post_fusion_mlp = nn.Sequential(
        #     nn.Linear(input_dim * self.prefix_len, 4096),
        #     nn.LayerNorm(4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 2048),
        #     nn.LayerNorm(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, output_dim)
        # )
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * self.prefix_len, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim)
        )
        
        self.apply(init_weights_kaiming)
        
    def forward(
        self,
        hand_token: torch.Tensor,  # [B, N, input_dim]
        head_token: torch.Tensor,  # [B, N, input_dim] 
        coords_hand: torch.Tensor = None,
        coords_head: torch.Tensor = None,
        state: torch.Tensor = None,  # [B, input_dim] or None
        text_embeddings: torch.Tensor = None,  # [B, input_dim] or None
        voxel_token: torch.Tensor = None,       # [B, M, D] or None
        coords_voxel: torch.Tensor = None,
    ) -> torch.Tensor:
        B, N, D = hand_token.shape
        
        prefix_token = self.prefix.expand(B, self.prefix_len, D)  # [B, prefix_len, D]
        prefix_coords = torch.zeros(B, self.prefix_len, 3, device=hand_token.device)
        
        tokens = []
        coords_list = []
        
        if state is not None:
            state_token = state.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))
        
        if text_embeddings is not None:
            text_token = text_embeddings.unsqueeze(1)  # [B, 1, D]
            tokens.append(text_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))

        if voxel_token is not None:
            M = voxel_token.size(1)
            voxel_token = voxel_token + self.modality_embed_3d.expand(B, M, -1)
            tokens.append(voxel_token)    # [B, M, D]
            coords_list.append(coords_voxel)  # [B, M, 3]

        hand_token = hand_token + self.modality_embed_image.expand(B, N, -1)
        tokens.append(hand_token)     # [B, N, D]
        coords_list.append(coords_hand)
        
        head_token = head_token + self.modality_embed_image.expand(B, N, -1)
        tokens.append(head_token)     # [B, N, D]
        coords_list.append(coords_head)
        
        main_src = torch.cat(tokens, dim=1)         # [B, S+N, D]
        main_coords = torch.cat(coords_list, dim=1) # [B, S+N, 3]
        
        src = torch.cat([prefix_token, main_src], dim=1)         # [B, prefix_len + S + N, D]
        coords_src = torch.cat([prefix_coords, main_coords], dim=1)  # [B, prefix_len + S + N, 3]

        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(src=src, coords_src=coords_src)

        # start_idx = self.prefix_len
        # if state is not None:
        #     start_idx += 1
        # if text_embeddings is not None:
        #     start_idx += 1
        # if voxel_token is not None:
        #     start_idx += M

        fused_tokens = src[:, :self.prefix_len, :]  # [B, (N + ?), input_dim]
        data = fused_tokens.reshape(B, -1)
        out = self.post_fusion_mlp(data)  # [B, output_dim]
        return out

class Agent_global_multistep(nn.Module):
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
        implicit_decoder: ImplicitDecoder = None,
        global_k: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        prefix_len: int = 16
    ):
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
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        # Reduce CLIP feature dimension
        self.clip_dim_reducer = DimReducer(clip_input_dim, voxel_feature_dim, L=10)
        
        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=256,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=state_mlp_dim,
            prefix_len=prefix_len
        )
        
        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder

        self.state_proj = nn.Linear(state_dim, voxel_feature_dim).to(self.device)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
         
        # Time embeddings for conditioning
        # self.subtask_embedding = nn.Parameter(
        #     torch.randn(episode_num, voxel_feature_dim, device=device) * 0.01
        # )

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
        
        # Project state
        state_projected = self.state_proj(state)

        # For HEAD: get nearest K global coords
        head_translation = head_pose[:, 0, :3, 3]  # [B, 3]
        valid_coords = self.static_map.valid_grid_coords
        
        valid_coords_exp = valid_coords.unsqueeze(0).expand(B_, -1, 3)
        dist = torch.norm(valid_coords_exp - head_translation.unsqueeze(1), dim=-1)
        K = self.global_k
        _, topk_indices = torch.topk(dist, k=K, dim=-1, largest=False)
        
        coords_kv_head = torch.gather(
            valid_coords_exp, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        coords_kv_head_flat = coords_kv_head.view(B_ * K, 3)

        # Decode HEAD voxel
        feats_kv_flat_static_head, _ = self.static_map.query_voxel_feature(
            coords_kv_head_flat, return_indices=False
        )
        head_voxels_flat, _ = self.implicit_decoder(
            feats_kv_flat_static_head, coords_kv_head_flat, return_intermediate=True
        )
        head_voxel = head_voxels_flat.view(B_, K, -1)
        
        # Reduce CLIP dimension 
        feats_hand_flat_reduced = self.clip_dim_reducer(feats_hand_flat, hand_coords_world_flat)
        feats_head_flat_reduced = self.clip_dim_reducer(feats_head_flat, head_coords_world_flat)
        feats_hand_reduced = feats_hand_flat_reduced.view(B_, N, -1)
        feats_head_reduced = feats_head_flat_reduced.view(B_, N, -1)

        # Get text embeddings (projected)
        text_embeddings_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced = text_embeddings_reduced[object_labels, :]

        # Prepare coords for transformer
        batch_hand_coords = hand_coords_world_flat.view(B_, N, 3)
        batch_head_coords = head_coords_world_flat.view(B_, N, 3)

        # Transformer
        visual_token = self.transformer(
            hand_token=feats_hand_reduced,
            head_token=feats_head_reduced,
            coords_hand=batch_hand_coords,
            coords_head=batch_head_coords,
            state=state_projected,
            text_embeddings=selected_text_reduced,
            voxel_token=head_voxel,
            coords_voxel=coords_kv_head,
        )
        

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([state_token, visual_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred