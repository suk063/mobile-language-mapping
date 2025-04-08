import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
# from ..module.transformer import TransformerEncoder, GlobalPerceiver
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, ConcatMLPFusion,VoxelProj
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip
import math

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
 
class PerceiverAttentionLayer(nn.Module):
    def __init__(
        self, 
        dim: int = 256, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        assert dim % nhead == 0, "dim must be divisible by nhead"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Output projection after attention
        self.out_proj = nn.Linear(dim, dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim),
        )

        # LayerNorm layers
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords_q: torch.Tensor = None,
        coords_kv: torch.Tensor = None
    ) -> torch.Tensor:
        B, Q_len, _ = q.shape
        _, KV_len, _ = k.shape

        # Linear projection
        q_proj = self.W_q(q).view(B, Q_len, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, Q_len, head_dim]
        k_proj = self.W_k(k).view(B, KV_len, self.nhead, self.head_dim).transpose(1, 2) # [B, nhead, KV_len, head_dim]
        v_proj = self.W_v(v).view(B, KV_len, self.nhead, self.head_dim).transpose(1, 2) # [B, nhead, KV_len, head_dim]

        # Apply Rotary Positional Embedding if provided
        if coords_q is not None:
            q_proj = rotary_pe_3d(q_proj, coords_q)
        if coords_kv is not None:
            k_proj = rotary_pe_3d(k_proj, coords_kv)

        # 3) Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_proj)

        # Combine heads back into single tensor
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q_len, self.dim)

        # 5) Residual + Norm
        out = self.out_proj(attn_output)
        x = self.ln1(q + self.dropout_attn(out))

        # 6) Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout_ff(ffn_out))

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
        input_dim: int = 240,
        nhead: int = 8,
        num_layers: int = 2,
        hidden_dim: int = 1024,
        out_dim: int = 240,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nhead = nhead

        # Learnable tokens (2 tokens = hand, head)
        self.learnable_tokens = nn.Parameter(torch.zeros(1, 2, input_dim))
        nn.init.xavier_uniform_(self.learnable_tokens)

        # Optional modality embedding for the tokens
        self.modality_embed_learnable = nn.Parameter(torch.randn(1, 1, input_dim))

        # Perceiver cross-attn layers
        self.layers = nn.ModuleList([
            PerceiverAttentionLayer(dim=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
            for _ in range(num_layers)
        ])

        # projection
        self.out_proj = nn.Linear(input_dim, out_dim)
        self.apply(init_weights_kaiming)

    def forward(
        self,
        state,                     # [B, hidden_dim]
        hand_translation_all: torch.Tensor,  # [B, 3]
        head_translation_all: torch.Tensor,  # [B, 3]
        valid_coords: torch.Tensor,          # [B, N, 3]
        valid_feats_projected: torch.Tensor            # [B, N, feat_dim]
    ) -> torch.Tensor:
        """
        Args:
            hand_translation_all: [B, 3]
            head_translation_all: [B, 3]
            valid_coords:         [B, N, 3]
            valid_feats:          [B, N, feat_dim]
        Returns:
            out: [B, 2, out_dim]
        """
        B2, N, _ = valid_feats_projected.shape

        # (1) state token
        state_token = state.unsqueeze(1)  # [B,1,hidden_dim]
        coords_state = torch.zeros(B2, 1, 3, device=state.device)

        # (2) learnable tokens
        learned_tokens = (
            self.learnable_tokens + self.modality_embed_learnable
        ).repeat(B2, 1, 1)  # [B,2,input_dim]
        coords_learned = torch.stack([hand_translation_all, head_translation_all], dim=1)  # [B,2,3]

        # Combine them: total Q_len=3
        q = torch.cat([state_token, learned_tokens], dim=1)      
        coords_q = torch.cat([coords_state, coords_learned], dim=1)

        # K, V
        k = valid_feats_projected
        v = valid_feats_projected
        coords_kv = valid_coords
        
        # (4) Pass through cross-attention layers
        x = q
        for layer in self.layers:
            x = layer(
                q=x,
                k=k,
                v=v,
                coords_q=coords_q,
                coords_kv=coords_kv
            )
        # x shape: [B,3,hidden_dim]
        # The first token (index=0) is the state token; we only want the 2 learned tokens
        # => [B,2,hidden_dim], then apply out_proj
        out_tokens = x[:, 1:, :]
        out = self.out_proj(out_tokens)  # [B,2,out_dim]
        return out   
            
class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1
    ):
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
        input_dim=240,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        output_dim=128,
        proj_dim=16,
    ):
        super().__init__()

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
        
        self.output_proj= nn.Linear(input_dim, proj_dim)
        
        # Post-fusion MLP
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * (256 * 2 + 4), 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
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
        perceiver_out_all: torch.Tensor = None,
        hand_translation_all: torch.Tensor = None,
        head_translation_all: torch.Tensor = None,
    ) -> torch.Tensor:
        B2, N, D = hand_token.shape
        
        tokens = []
        coords_list = []
        
        if state is not None:
            state_token = state.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token)
            coords_list.append(torch.zeros(B2, 1, 3, device=state.device))
        
        if text_embeddings is not None:
            text_token = text_embeddings.unsqueeze(1)  # [B, 1, D]
            tokens.append(text_token)
            coords_list.append(torch.zeros(B2, 1, 3, device=state.device))

        if perceiver_out_all is not None:
            M = 2
            perceiver_out_all = perceiver_out_all + self.modality_embed_3d.expand(B2, M, -1) # [B, 2, D]
            tokens.append(perceiver_out_all)    # [B, M, D]
            
            coords_cam = torch.stack([hand_translation_all, head_translation_all], dim=1)         # [B,2,3]
            coords_list.append(coords_cam)  # [B, 2, 3]

        hand_token = hand_token + self.modality_embed_image.expand(B2, N, -1)
        tokens.append(hand_token)     # [B, N, D]
        coords_list.append(coords_hand)
        
        head_token = head_token + self.modality_embed_image.expand(B2, N, -1)
        tokens.append(head_token)     # [B, N, D]
        coords_list.append(coords_head)
        
        src = torch.cat(tokens, dim=1)         # [B, S+N, D]
        coords_src = torch.cat(coords_list, dim=1) # [B, S+N, 3]
        
        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(src=src, coords_src=coords_src)

        start_idx = 0
        # if state is not None:
        #     start_idx += 1
        # if text_embeddings is not None:
        #     start_idx += 1
        # if perceiver_out_all is not None:
        #     start_idx += M

        # fused_tokens = src[:, start_idx:, :]  # [B, (N + ?), input_dim]
        
        fused_tokens = self.output_proj(src)
        
        data = fused_tokens.reshape(B2, -1)
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
        state_mlp_dim: int = 128,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: VoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers: int = 2,
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
            hidden_dim=1024,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=state_mlp_dim,
        )
        
        self.global_perceiver = GlobalPerceiver(
            input_dim=voxel_feature_dim,
            nhead=num_heads,
            num_layers=num_layers,
            out_dim=voxel_feature_dim
        )
        
        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 6,
            action_dim=action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder

        self.state_proj = nn.Linear(state_dim, voxel_feature_dim).to(self.device)
        self.voxel_proj = VoxelProj(voxel_feature_dim=voxel_feature_dim).to(self.device)

        # Local feature fusion
        self.feature_fusion = ConcatMLPFusion(feat_dim=voxel_feature_dim, clip_embedding_dim=clip_input_dim)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

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
            hand_visfeat_all = get_visual_features(self.clip_model, hand_rgb_all)
            head_visfeat_all = get_visual_features(self.clip_model, head_rgb_all)
        
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
        hand_translation_all = hand_pose_all[:, 0, :3, 3]  # [2B, 3]
        head_translation_all = head_pose_all[:, 0, :3, 3]  # [2B, 3]
        
        valid_coords, valid_feats = self.static_map.get_all_valid_voxel_data() # [N, 3], [N, D]
        valid_feats_projected, _ = self.implicit_decoder(valid_feats, valid_coords, return_intermediate=True)
        
        valid_coords_expanded = valid_coords.unsqueeze(0).expand(2*B, -1, -1)  # [2B, N, 3]
        valid_feats_projected_expanded = valid_feats_projected.unsqueeze(0).expand(2*B, -1, -1)
        
        # Project state
        state_proj_all = self.state_proj(state_all)
        
        # 3) Run GlobalPerceiver
        perceiver_out_all = self.global_perceiver(
            state=state_proj_all,
            hand_translation_all=hand_translation_all,
            head_translation_all=head_translation_all,
            valid_coords=valid_coords_expanded,  
            valid_feats_projected=valid_feats_projected_expanded 
        )
        
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

        # # Query voxel features 
        # with torch.no_grad():
        #     voxel_feat_points_hand_flat_all, _ = self.static_map.query_voxel_feature(
        #         hand_coords_world_flat_all, return_indices=False
        #     )
        #     voxel_feat_points_head_flat_all, _ = self.static_map.query_voxel_feature(
        #         head_coords_world_flat_all, return_indices=False
        #     )
        
        # voxel_feat_points_hand_flat_all, _ = self.implicit_decoder(voxel_feat_points_hand_flat_all, hand_coords_world_flat_all, return_intermediate=True)
        # voxel_feat_points_head_flat_all, _ = self.implicit_decoder(voxel_feat_points_head_flat_all, head_coords_world_flat_all, return_intermediate=True)
        
        # Fuse voxel and CLIP features
        # fused_hand_all = self.feature_fusion(
        #     voxel_feat_points_hand_flat_all,
        #     feats_hand_flat_all,
        #     hand_coords_world_flat_all
        # ).view(2*B, N, -1)
        
        # fused_head_all = self.feature_fusion(
        #     voxel_feat_points_head_flat_all,
        #     feats_head_flat_all,
        #     head_coords_world_flat_all
        # ).view(2*B, N, -1)
            
        # Text embeddings
        object_labels_all = torch.cat([object_labels, object_labels], dim=0)
        text_emb_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced_all = text_emb_reduced[object_labels_all, :]

        # Transformer forward
        out_transformer_all = self.transformer(
            hand_token=feats_hand_reduced_all,
            head_token=feats_head_reduced_all,
            coords_hand=hand_coords_world_flat_all.reshape(2*B, N, 3),
            coords_head=head_coords_world_flat_all.reshape(2*B, N, 3),
            state=state_proj_all, 
            text_embeddings=selected_text_reduced_all,
            perceiver_out_all=perceiver_out_all,
            hand_translation_all=hand_translation_all,
            head_translation_all=head_translation_all,
        )

        # 8) Split results back to t and t-1
        out_transformer_t, out_transformer_m1 = torch.split(out_transformer_all, B, dim=0)
        state_projected_t, state_projected_m1 = torch.split(self.state_mlp(state_all), B, dim=0)

        # 9) Build final action_t using (state_t, out_t, out_t-1)
        state_projected_delta = state_projected_t - state_projected_m1
        out_transformer_delta = out_transformer_t - out_transformer_m1
        
        action_input_t = torch.cat([state_projected_t, state_projected_m1, state_projected_delta, out_transformer_t, out_transformer_m1, out_transformer_delta], dim=-1)
        action_t = self.action_mlp(action_input_t)

        return action_t