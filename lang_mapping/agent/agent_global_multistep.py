import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
# from ..module.transformer import TransformerEncoder, GlobalPerceiver
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from ..utils import get_3d_coordinates, get_visual_features, transform
import open_clip
import math

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
def rotary_pe_3d(
    x: torch.Tensor,      # Could be [B, S, D] or [B, n_heads, S, D]
    coords: torch.Tensor, # [B, S, 3]
    base: float = 10000.0
) -> torch.Tensor:
    """
    A flexible 3D rotary positional embedding that supports both 3D and 4D inputs.
    Args:
        x: [..., S, D], either [B, S, D] or [B, n_heads, S, D].
        coords: [B, S, 3].
        base: Base for frequency calculation.
    Returns:
        Tensor of the same shape as x, with RoPE applied.
    """
    # If input x is 4D => flatten to 3D, apply RoPE, then reshape back.
    if x.dim() == 4:
        B, H, S, D = x.shape
        # Flatten the first two dims for x
        x_reshaped = x.reshape(B * H, S, D)

        # Also reshape coords to match (B*H, S, 3)
        # Here we broadcast coords over the head dimension
        coords_reshaped = coords.unsqueeze(1).expand(B, H, S, 3).reshape(B * H, S, 3)

        # Apply RoPE in 3D form
        x_rotated = _rotary_pe_3d_impl(x_reshaped, coords_reshaped, base)

        # Reshape back to [B, n_heads, S, D]
        return x_rotated.view(B, H, S, D)
    elif x.dim() == 3:
        # Directly apply the original logic
        return _rotary_pe_3d_impl(x, coords, base)
    else:
        raise ValueError(
            f"rotary_pe_3d expects x to be 3D or 4D, but got shape {x.shape}"
        )


def _rotary_pe_3d_impl(
    x: torch.Tensor,      # [B, S, D]
    coords: torch.Tensor, # [B, S, 3]
    base: float = 10000.0
) -> torch.Tensor:
    """
    Core RoPE logic for 3D input shape [B, S, D].
    """
    B, S, D = x.shape
    assert D % 6 == 0, "D must be a multiple of 6"
    num_block = D // 6

    # Compute frequency factors
    k_idx = torch.arange(num_block, device=x.device, dtype=x.dtype)
    theta_k = 1.0 / (base ** (k_idx / (D / 6)))

    # Reshape x into blocks of size 6
    x_splitted = x.view(B, S, num_block, 6)   # [B, S, num_block, 6]

    # coords: [B, S, 3] => separate x_p, y_p, z_p
    x_p, y_p, z_p = coords[..., 0], coords[..., 1], coords[..., 2]

    out_blocks = []
    for k in range(num_block):
        block = x_splitted[:, :, k, :]   # [B, S, 6]
        x_angle = x_p * theta_k[k]
        y_angle = y_p * theta_k[k]
        z_angle = z_p * theta_k[k]

        b0, b1, b2, b3, b4, b5 = (block[..., i] for i in range(6))
        cos_x, sin_x = torch.cos(x_angle), torch.sin(x_angle)
        cos_y, sin_y = torch.cos(y_angle), torch.sin(y_angle)
        cos_z, sin_z = torch.cos(z_angle), torch.sin(z_angle)

        # Rotate pairs around X, Y, Z
        b0_ = b0 * cos_x - b1 * sin_x
        b1_ = b0 * sin_x + b1 * cos_x

        b2_ = b2 * cos_y - b3 * sin_y
        b3_ = b2 * sin_y + b3 * cos_y

        b4_ = b4 * cos_z - b5 * sin_z
        b5_ = b4 * sin_z + b5 * cos_z

        out_blocks.append(torch.stack([b0_, b1_, b2_, b3_, b4_, b5_], dim=-1))

    # Stack all blocks along num_block dim, then reshape back
    x_out = torch.stack(out_blocks, dim=2).view(B, S, D)
    return x_out

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, coords_src: torch.Tensor = None) -> torch.Tensor:
        # src shape: (B, S, d_model)
        B, S, _ = src.shape
        
        # 1) Q, K, V projections
        q = self.W_q(src)  # (B, S, d_model)
        k = self.W_k(src)
        v = self.W_v(src)
        
        # 2) Reshape and transpose for multi-head
        # => (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3) Apply RoPE if coords_src is provided
        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
            # v is often unchanged in RoPE
        
        # 4) Scaled dot-product attention
        # scores shape: (B, n_heads, S, S)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, n_heads, S, head_dim)
        
        # 5) Reshape back to (B, S, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        
        # 6) Output projection + residual + norm
        out = self.out_proj(attn_output)
        src2 = self.norm1(src + self.dropout_attn(out))
        
        # 7) Feed-forward
        ff_out = self.linear2(self.activation(self.linear1(src2)))
        out2 = self.norm2(src2 + self.dropout_ff(ff_out))
        
        return out2
    
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
                n_heads=num_heads,
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
            input_dim=state_mlp_dim * 4,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder

        self.state_proj = nn.Linear(state_dim, voxel_feature_dim).to(self.device)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
         
        self.global_k = global_k

    def forward(self, observations, object_labels, subtask_idx):
        """
        Forward pass that processes time t and t-1 in a single batch:
         1) Merge data of t and t-1 to form batch=2B
         2) Do feature extraction, depth, coordinates, voxel
         3) Split back into t and t-1
         4) Compute action_t = MLP(state_t, out_t, out_t-1)
        """
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
       
        # 2) Concatenate t and t-1 (2B)
        hand_rgb_all  = torch.cat([hand_rgb_t,  hand_rgb_m1],  dim=0)
        head_rgb_all  = torch.cat([head_rgb_t,  head_rgb_m1],  dim=0)
        hand_depth_all = torch.cat([hand_depth_t, hand_depth_m1], dim=0)
        head_depth_all = torch.cat([head_depth_t, head_depth_m1], dim=0)
        hand_pose_all = torch.cat([hand_pose_t, hand_pose_m1], dim=0)
        head_pose_all = torch.cat([head_pose_t, head_pose_m1], dim=0)
        state_all = torch.cat([state_t, state_m1], dim=0)

        # If needed, permute hand_rgb_all so channel=3
        if hand_rgb_all.shape[2] != 3:
            hand_rgb_all = hand_rgb_all.permute(0, 1, 4, 2, 3)
            head_rgb_all = head_rgb_all.permute(0, 1, 4, 2, 3)
        
        # Flatten frames
        _, fs, d, H, W = hand_rgb_all.shape
        hand_rgb_all = hand_rgb_all.reshape(2*B, fs * d, H, W)
        head_rgb_all = head_rgb_all.reshape(2*B, fs * d, H, W)

        # Transform to [0,1], apply normalization
        hand_rgb_all = transform(hand_rgb_all.float() / 255.0)
        head_rgb_all = transform(head_rgb_all.float() / 255.0)

        with torch.no_grad():
            # Combine hand + head => single pass in CLIP
            rgb_combined_all = torch.cat([hand_rgb_all, head_rgb_all], dim=0) 
            visfeat_combined_all = get_visual_features(self.clip_model, rgb_combined_all)
            # Split back
            hand_visfeat_all, head_visfeat_all = torch.split(visfeat_combined_all, 2*B, dim=0)
        
        # Handle depth (reshape, interpolate)
        
        hand_depth_all = hand_depth_all / 1000.0
        head_depth_all = head_depth_all / 1000.0
        
        if hand_depth_all.dim() == 5:
            _, fs, d2, H, W = hand_depth_all.shape
            hand_depth_all = hand_depth_all.view(2*B, fs * d2, H, W)
            head_depth_all = head_depth_all.view(2*B, fs * d2, H, W)
            hand_depth_all = F.interpolate(hand_depth_all, (16, 16), mode="nearest")
            head_depth_all = F.interpolate(head_depth_all, (16, 16), mode="nearest")

        # 3D world coords
        hand_coords_world_all, _ = get_3d_coordinates(
            hand_visfeat_all, hand_depth_all, hand_pose_all, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_all, _ = get_3d_coordinates(
            head_visfeat_all, head_depth_all, head_pose_all,
            self.fx, self.fy, self.cx, self.cy
        )

        # Global voxel query using head position
        head_translation_all = head_pose_all[:, 0, :3, 3]
        valid_coords = self.static_map.valid_grid_coords
        valid_coords_exp = valid_coords.unsqueeze(0).expand(2*B, -1, 3)
        dist = torch.norm(valid_coords_exp - head_translation_all.unsqueeze(1), dim=-1)
        K = self.global_k
        _, topk_indices = torch.topk(dist, k=K, dim=-1, largest=False)
        coords_kv_head_all = torch.gather(
            valid_coords_exp, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        coords_kv_head_flat_all = coords_kv_head_all.view(2*B*K, 3)
        feats_kv_flat_static_head_all, _ = self.static_map.query_voxel_feature(
            coords_kv_head_flat_all, return_indices=False
        )
        head_voxels_flat_all, _ = self.implicit_decoder(
            feats_kv_flat_static_head_all, coords_kv_head_flat_all, return_intermediate=True
        )
        head_voxel_all = head_voxels_flat_all.view(2*B, K, -1)

        # Reduce CLIP dimension for hand/head
        _, C_, Hf, Wf = hand_coords_world_all.shape
        N = Hf * Wf

        feats_hand_all = hand_visfeat_all.permute(0, 2, 3, 1).reshape(2*B, N, -1)
        feats_head_all = head_visfeat_all.permute(0, 2, 3, 1).reshape(2*B, N, -1)

        hand_coords_world_flat_all = hand_coords_world_all.permute(0, 2, 3, 1).reshape(2*B*N, 3)
        feats_hand_flat_all = feats_hand_all.reshape(2*B*N, -1)
        feats_hand_reduced_flat = self.clip_dim_reducer(feats_hand_flat_all, hand_coords_world_flat_all)
        feats_hand_reduced_all = feats_hand_reduced_flat.view(2*B, N, -1)

        head_coords_world_flat_all = head_coords_world_all.permute(0, 2, 3, 1).reshape(2*B*N, 3)
        feats_head_flat_all = feats_head_all.reshape(2*B*N, -1)
        feats_head_reduced_flat = self.clip_dim_reducer(feats_head_flat_all, head_coords_world_flat_all)
        feats_head_reduced_all = feats_head_reduced_flat.view(2*B, N, -1)

        # Text embeddings
        object_labels_all = torch.cat([object_labels, object_labels], dim=0)
        text_emb_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced_all = text_emb_reduced[object_labels_all, :]

        # Project state
        state_proj_all = self.state_proj(state_all)

        # Transformer forward
        out_transformer_all = self.transformer(
            hand_token=feats_hand_reduced_all,
            head_token=feats_head_reduced_all,
            coords_hand=hand_coords_world_all.permute(0, 2, 3, 1).reshape(2*B, N, 3),
            coords_head=head_coords_world_all.permute(0, 2, 3, 1).reshape(2*B, N, 3),
            state=state_proj_all, 
            text_embeddings=selected_text_reduced_all,
            voxel_token=head_voxel_all,
            coords_voxel=coords_kv_head_all
        )

        # 8) Split results back to t and t-1
        out_transformer_t, out_transformer_m1 = torch.split(out_transformer_all, B, dim=0)

        state_projected_t, state_projected_m1 = torch.split(self.state_mlp(state_all), B, dim=0)

        # 9) Build final action_t using (state_t, out_t, out_t-1)
        action_input_t = torch.cat([state_projected_t, state_projected_m1, out_transformer_t, out_transformer_m1], dim=-1)
        action_t = self.action_mlp(action_input_t)

        return action_t