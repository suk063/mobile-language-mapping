import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ..module.transformer import TransformerEncoder, GlobalPerceiver, LocalSelfAttentionFusion
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj, ConcatMLPFusion, VoxelProj
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from lang_mapping.grid_net import GridNet

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip
import math


def generate_subsequent_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) 
    mask = mask.bool()  # True/False
    mask = mask.masked_fill(mask, float('-inf')) 
    return mask

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_causal_mask(seq_len, len_m1, device):
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    causal_mask[:len_m1, len_m1:] = True

    return causal_mask  # (S, S) 크기

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

    def forward(
        self, 
        src: torch.Tensor,             # (B, S, d_model)
        coords_src: torch.Tensor = None,  # (B, S, 3) or None
        len_m1: int = 513
    ) -> torch.Tensor:
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
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if len_m1 > 0:
            attn_mask = make_causal_mask(S, len_m1, src.device)  # (S, S)
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))   
    
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn_output)))
        
        # Feed froward network
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
        proj_dim=16,
    ):
        super().__init__()
        
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
        
        self.apply(init_weights_kaiming)
               
    def forward(
        self,
        hand_token_t: torch.Tensor,  # [B, N, input_dim]
        head_token_t: torch.Tensor,  # [B, N, input_dim] 
        hand_token_m1: torch.Tensor,  # [B, N, input_dim]
        head_token_m1: torch.Tensor,  # [B, N, input_dim]
        coords_hand_t: torch.Tensor = None,
        coords_head_t: torch.Tensor = None, 
        coords_hand_m1: torch.Tensor = None,
        coords_head_m1: torch.Tensor = None,
        state_t: torch.Tensor = None,  # [B, input_dim] or None
        state_m1: torch.Tensor = None,  # [B, input_dim] or None
    ) -> torch.Tensor:
        B, N, D = hand_token_t.shape
        
        tokens = []
        coords_list = []
        coords_src = None
        
        if state_m1 is not None:
            state_token_m1 = state_m1.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token_m1)
            coords_list.append(torch.zeros(B, 1, 3, device=state_t.device))
        
        tokens.append(hand_token_m1)     # [B, N, D]
        tokens.append(head_token_m1) 
        
        if coords_hand_m1 is not None:
            coords_list.append(coords_hand_m1)
            coords_list.append(coords_head_m1)
        
        if state_t is not None:
            state_token_t = state_t.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token_t)
            coords_list.append(torch.zeros(B, 1, 3, device=state_t.device))

        tokens.append(hand_token_t)     # [B, N, D]
        tokens.append(head_token_t) 
        
        if coords_hand_t is not None:
            coords_list.append(coords_hand_t)
            coords_list.append(coords_head_t)
        
        src = torch.cat(tokens, dim=1)
        if len(coords_list) > 0:
            coords_src = torch.cat(coords_list, dim=1)  # (B, S, 3)
        
        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
            )

        start_idx = 1 + 512 # 1 for state, 512 for m1, 1 for state

        return src[:, start_idx:, :]

class ActionTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        action_dim: int,
        action_horizon: int = 16,
    ):
        super().__init__()
        
        self.query_embed = nn.Embedding(action_horizon, d_model)  # [3, d_model]
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.action_head = nn.Linear(d_model, action_dim)
        self.action_horizon = action_horizon
        
    def forward(self, memory, state) -> torch.Tensor:

        # state # [B, 1, d_model]

        B, N, d_model = memory.shape
  
        # memory = memory.view(B, fs*N, d_model)             # [B, 2*N, d_model]
        memory = memory.permute(1, 0, 2).contiguous()     # [N, B, d_model]
        
        query_pos = self.query_embed.weight                # [3, d_model]
        query_pos = query_pos.unsqueeze(1).repeat(1, B, 1) # [3, B, d_model]
        
        state = state.permute(1, 0, 2).contiguous()
        tgt = torch.cat([state, query_pos], dim=0)
        
        causal_mask = generate_subsequent_mask(self.action_horizon+1).to(memory.device)
        
        decoder_out = self.decoder(
            tgt=tgt,    # [T, B, d_model]
            memory=memory,      # [N, B, d_model]
            tgt_mask=causal_mask
        ) 
        
        decoder_out = decoder_out.permute(1, 0, 2)         # [B, 4, d_model]
        action_out = self.action_head(decoder_out)         # [B, 4, action_dim]
        return action_out[:,1:, :]

class Agent_global_multistep_gridnet(nn.Module):
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
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers_transformer: int = 4,
        num_layers_perceiver: int = 2,
        num_learnable_tokens: int = 16,
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
        if text_input:
            text_input += [""]
        
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            self.text_embeddings = text_embeddings - redundant_emb


        # Reduce CLIP feature dimension
        self.dim_reducer_hand = DimReducer(clip_input_dim, voxel_feature_dim, L=10)
        self.dim_reducer_head = DimReducer(clip_input_dim, voxel_feature_dim, L=10)
        # self.dim_reducer = DimReducer(clip_input_dim, voxel_feature_dim, L=0)
        self.voxel_proj = DimReducer(clip_input_dim, voxel_feature_dim, L=10).to(self.device)
        
        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=1024,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )
        
        self.global_perceiver = GlobalPerceiver(
            input_dim=voxel_feature_dim,
            nhead=num_heads,
            num_layers=num_layers_perceiver,
            out_dim=voxel_feature_dim,
            num_learnable_tokens=num_learnable_tokens
        )
        
        # Action MLP
        self.action_dim = np.prod(single_act_shape)
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=240,             # out_transformer 최종 dim=240이라 가정
            nhead=8,
            num_decoder_layers=6,    # 필요에 따라 조정
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=self.action_dim
        ).to(self.device)

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder
        
        # self.feature_fusion = ConcatMLPFusion(feat_dim=voxel_feature_dim, clip_embedding_dim=clip_input_dim)
        self.feature_fusion_attn = LocalSelfAttentionFusion(feat_dim=clip_input_dim)

        self.state_proj_perceiver =  StateProj(state_dim=state_dim, output_dim=voxel_feature_dim).to(self.device)   
        self.state_proj_transformer =  StateProj(state_dim=state_dim, output_dim=voxel_feature_dim).to(self.device)   

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.valid_coords = self.static_map.features[0].vertex_positions().to(self.device)
        self.state_mlp_for_action = nn.Linear(state_dim, voxel_feature_dim).to(self.device)
    
    def forward(self, observations, object_labels):
        """
        Forward pass that processes time t and t-1 in a single batch:
         1) Merge data of t and t-1 to form batch=2B
         2) Do feature extraction, depth, coordinates, voxel
         3) Split back into t and t-1
         4) Compute action_t = MLP(state_t, out_t, out_t-1)
        """
        # 1) Extract data
        hand_rgb_t   = observations["pixels"]["fetch_hand_rgb"]
        head_rgb_t   = observations["pixels"]["fetch_head_rgb"]

        hand_depth_t  = observations["pixels"]["fetch_hand_depth"]
        head_depth_t  = observations["pixels"]["fetch_head_depth"]

        hand_pose_t   = observations["pixels"]["fetch_hand_pose"]
        head_pose_t   = observations["pixels"]["fetch_head_pose"]

        hand_rgb_m1  = observations["pixels"]["fetch_hand_rgb_m1"]
        head_rgb_m1  = observations["pixels"]["fetch_head_rgb_m1"]

        hand_depth_m1 = observations["pixels"]["fetch_hand_depth_m1"]
        head_depth_m1 = observations["pixels"]["fetch_head_depth_m1"]

        hand_pose_m1  = observations["pixels"]["fetch_hand_pose_m1"]
        head_pose_m1  = observations["pixels"]["fetch_head_pose_m1"]

        state_t  = observations["state"]
        state_m1 = observations["state_m1"]

        B = hand_rgb_t.shape[0]
       
        # If needed, permute hand_rgb_t so channel=3
        if hand_rgb_t.shape[2] != 3:
            hand_rgb_t = hand_rgb_t.permute(0, 1, 4, 2, 3)
            head_rgb_t = head_rgb_t.permute(0, 1, 4, 2, 3)
            hand_rgb_m1 = hand_rgb_m1.permute(0, 1, 4, 2, 3)
            head_rgb_m1 = head_rgb_m1.permute(0, 1, 4, 2, 3)
        
        # Flatten frames
        _, fs, d, H, W = hand_rgb_t.shape
        hand_rgb_t = hand_rgb_t.reshape(B, fs * d, H, W)
        head_rgb_t = head_rgb_t.reshape(B, fs * d, H, W)
        hand_rgb_m1 = hand_rgb_m1.reshape(B, fs * d, H, W)
        head_rgb_m1 = head_rgb_m1.reshape(B, fs * d, H, W)

        # Transform to [0,1], apply normalization
        hand_rgb_t = transform(hand_rgb_t.float() / 255.0)
        head_rgb_t = transform(head_rgb_t.float() / 255.0)
        hand_rgb_m1 = transform(hand_rgb_m1.float() / 255.0)
        head_rgb_m1 = transform(head_rgb_m1.float() / 255.0)

        with torch.no_grad():
            hand_visfeat_t = get_visual_features(self.clip_model, hand_rgb_t)
            head_visfeat_t = get_visual_features(self.clip_model, head_rgb_t)
            hand_visfeat_m1 = get_visual_features(self.clip_model, hand_rgb_m1)
            head_visfeat_m1 = get_visual_features(self.clip_model, head_rgb_m1)
        
        # Handle depth (reshape, interpolate)
        
        hand_depth_t = hand_depth_t / 1000.0
        head_depth_t = head_depth_t / 1000.0
        hand_depth_m1 = hand_depth_m1 / 1000.0
        head_depth_m1 = head_depth_m1 / 1000.0
        
        if hand_depth_t.dim() == 5:
            _, fs, d2, H, W = hand_depth_t.shape
            hand_depth_t = hand_depth_t.view(B, fs * d2, H, W)
            head_depth_t = head_depth_t.view(B, fs * d2, H, W)
            hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest")
            head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest")
            
            hand_depth_m1 = hand_depth_m1.view(B, fs * d2, H, W)
            head_depth_m1 = head_depth_m1.view(B, fs * d2, H, W)
            hand_depth_m1 = F.interpolate(hand_depth_m1, (16, 16), mode="nearest")
            head_depth_m1 = F.interpolate(head_depth_m1, (16, 16), mode="nearest")

        # 3D world coords
        hand_coords_world_t, _ = get_3d_coordinates(
            hand_visfeat_t, hand_depth_t, hand_pose_t, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_t, _ = get_3d_coordinates(
            head_visfeat_t, head_depth_t, head_pose_t,
            self.fx, self.fy, self.cx, self.cy
        )
        
        hand_coords_world_m1, _ = get_3d_coordinates(
            hand_visfeat_m1, hand_depth_m1, hand_pose_m1, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_m1, _ = get_3d_coordinates(
            head_visfeat_m1, head_depth_m1, head_pose_m1,
            self.fx, self.fy, self.cx, self.cy
        )
    
    
        # Global voxel query using each pose
        # hand_translation_t = hand_pose_t[:, 0, :3, 3]  # [2B, 3]
        # head_translation_t = head_pose_t[:, 0, :3, 3]  # [2B, 3]
                
        # valid_feats = self.static_map.query_feature(self.valid_coords)
        
        # valid_feats_projected = self.implicit_decoder(valid_feats, self.valid_coords, return_intermediate=False)

        # valid_feats_projected = self.voxel_proj(valid_feats_projected, self.valid_coords)

        # valid_coords_expanded = self.valid_coords.unsqueeze(0).expand(B, -1, -1)  # [2B, N, 3]
        # valid_feats_projected_expanded = valid_feats_projected.unsqueeze(0).expand(B, -1, -1)
        
        # # Project state
        # state_proj_perceiver_t = self.state_proj_perceiver(state_t)
        
        # 3) Run GlobalPerceiver
        # perceiver_out_t = self.global_perceiver(
        #     state=state_proj_perceiver_t,
        #     hand_translation_t=hand_translation_t,
        #     head_translation_t=head_translation_t,
        #     valid_coords=valid_coords_expanded,  
        #     valid_feats_projected=valid_feats_projected_expanded 
        # )
        
        # Reduce CLIP dimension for hand/head
        _, C_, Hf, Wf = hand_coords_world_t.shape
        N = Hf * Wf

        feats_hand_t = hand_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_t = head_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_hand_m1 = hand_visfeat_m1.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_m1 = head_visfeat_m1.permute(0, 2, 3, 1).reshape(B, N, -1)

        feats_hand_t_norm = F.normalize(feats_hand_t, dim=-1, p=2)
        feats_head_t_norm = F.normalize(feats_head_t, dim=-1, p=2)
        feats_hand_m1_norm = F.normalize(feats_hand_m1, dim=-1, p=2)
        feats_head_m1_norm = F.normalize(feats_head_m1, dim=-1, p=2)
        
        text_embed_norm = F.normalize(self.text_embeddings, dim=-1, p=2)
        text_embed_batch = text_embed_norm[object_labels, :].unsqueeze(1)
        
        gating_score_hand_t = (feats_hand_t_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_head_t = (feats_head_t_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_hand_m1 = (feats_hand_m1_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
        gating_score_head_m1 = (feats_head_m1_norm * text_embed_batch).sum(dim=-1, keepdim=True)  # [B, N, 1]
            
        feats_hand_t_gated = feats_hand_t + feats_hand_t * gating_score_hand_t
        feats_head_t_gated = feats_head_t + feats_head_t * gating_score_head_t
        feats_hand_m1_gated = feats_hand_m1 + feats_hand_m1 * gating_score_hand_m1
        feats_head_m1_gated = feats_head_m1 + feats_head_m1 * gating_score_head_m1
                
        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_hand_flat_t = feats_hand_t_gated.reshape(B*N, -1)
        feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_t, hand_coords_world_flat_t)
        feats_hand_reduced_t = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_t = feats_head_t_gated.reshape(B*N, -1)
        feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_t, head_coords_world_flat_t)
        feats_head_reduced_t = feats_head_reduced_flat.view(B, N, -1)
        
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_hand_flat_m1 = feats_hand_m1_gated.reshape(B*N, -1)
        feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_m1, hand_coords_world_flat_m1)
        feats_hand_reduced_m1 = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_m1 = feats_head_m1_gated.reshape(B*N, -1)
        feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_m1, head_coords_world_flat_m1)
        feats_head_reduced_m1 = feats_head_reduced_flat.view(B, N, -1)
        
        # Query voxel features and cos simeilarity
        # with torch.no_grad():
        #     voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t)
        #     voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t)
    
        # voxel_feat_points_hand_flat_final_t = self.implicit_decoder(
        #     voxel_feat_points_hand_flat_t, hand_coords_world_flat_t, return_intermediate=False)
        # voxel_feat_points_head_flat_final_t = self.implicit_decoder(
        #     voxel_feat_points_head_flat_t, head_coords_world_flat_t, return_intermediate=False)
        
        # cos_sim_hand = F.cosine_similarity(voxel_feat_points_hand_flat_final_t, feats_hand_flat_t, dim=-1)
        # cos_loss_hand = 1.0 - cos_sim_hand.mean()  
        
        # cos_sim_head = F.cosine_similarity(voxel_feat_points_head_flat_final_t, feats_head_flat_t, dim=-1)
        # cos_loss_head = 1.0 - cos_sim_head.mean()      
        
        # total_cos_loss = cos_loss_hand + cos_loss_head 
        
        # Fuse voxel and CLIP features
        # fused_hand_t = self.feature_fusion_attn(
        #     voxel_feat_points_hand_flat_final_t.view(B, N, -1),
        #     feats_hand_flat_t.view(B, N, -1),
        # ).reshape(B*N, -1)
        
        # fused_head_t = self.feature_fusion_attn(
        #     voxel_feat_points_head_flat_final_t.view(B, N, -1),
        #     feats_head_flat_t.view(B, N, -1),
        # ).reshape(B*N, -1)
        
        # fused_hand_reduced_t = self.dim_reducer(fused_hand_t, hand_coords_world_flat_t).view(B, N, -1)
        # fused_head_reduced_t = self.dim_reducer(fused_head_t, head_coords_world_flat_t).view(B, N, -1)        

        state_proj_transformer_t = self.state_proj_transformer(state_t)
        state_proj_transformer_m1 = self.state_proj_transformer(state_m1)

        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=feats_hand_reduced_t,
            head_token_t=feats_head_reduced_t,
            hand_token_m1=feats_hand_reduced_m1,
            head_token_m1=feats_head_reduced_m1,
            coords_hand_t=hand_coords_world_flat_t.reshape(B, N, 3),
            coords_head_t=head_coords_world_flat_t.reshape(B, N, 3),
            coords_hand_m1=hand_coords_world_flat_m1.reshape(B, N, 3),
            coords_head_m1=head_coords_world_flat_m1.reshape(B, N, 3),
            state_t=state_proj_transformer_t,
            state_m1=state_proj_transformer_m1,  
        ) # [B, N, 240]
        
        
        state_t_proj  = self.state_mlp_for_action(state_t).unsqueeze(1)   # [B, 240]
        action_out = self.action_transformer(out_transformer, state_t_proj)
        
        return action_out, None