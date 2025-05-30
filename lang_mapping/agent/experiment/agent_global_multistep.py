import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ..module.transformer import TransformerEncoder, GlobalPerceiver, LocalSelfAttentionFusion
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj, ConcatMLPFusion, VoxelProj
from lang_mapping.mapper.mapper_delta import VoxelHashTable

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip

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
        text_tokens = self.tokenizer(text_input).to(self.device)
        self.text_proj = nn.Linear(clip_input_dim, voxel_feature_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)

        # Reduce CLIP feature dimension
        self.dim_reducer = DimReducer(clip_input_dim, voxel_feature_dim, L=10)
        self.voxel_proj = DimReducer(clip_input_dim, voxel_feature_dim, L=10).to(self.device)
        
        # Transformer for feature fusion
        self.transformer = TransformerEncoder(
            input_dim=voxel_feature_dim,
            hidden_dim=1024,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
            output_dim=state_mlp_dim,
        )
        
        self.global_perceiver = GlobalPerceiver(
            input_dim=voxel_feature_dim,
            nhead=num_heads,
            num_layers=num_layers_perceiver,
            out_dim=voxel_feature_dim,
            num_learnable_tokens=num_learnable_tokens
        )
        
        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp_multi = ActionMLP(
            input_dim=state_mlp_dim * 4,
            action_dim=action_dim
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

        # Global voxel query using each pose
        hand_translation_all = hand_pose_all[:, 0, :3, 3]  # [2B, 3]
        head_translation_all = head_pose_all[:, 0, :3, 3]  # [2B, 3]
        
        valid_coords, valid_feats = self.static_map.get_all_valid_voxel_data() # [N, 3], [N, D]
        valid_feats = self.implicit_decoder(valid_feats, valid_coords, return_intermediate=False)
        
        valid_feats_projected = self.voxel_proj(valid_feats, valid_coords)
        
        valid_coords_expanded = valid_coords.unsqueeze(0).expand(2*B, -1, -1)  # [2B, N, 3]
        valid_feats_projected_expanded = valid_feats_projected.unsqueeze(0).expand(2*B, -1, -1)
        
        # Project state
        state_proj_perceiver_all = self.state_proj_perceiver(state_all)
        
        # 3) Run GlobalPerceiver
        perceiver_out_all = self.global_perceiver(
            state=state_proj_perceiver_all,
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
        # feats_hand_reduced_flat = self.clip_dim_reducer(feats_hand_flat_all, hand_coords_world_flat_all)
        # feats_hand_reduced_all = feats_hand_reduced_flat.view(2*B, N, -1)

        head_coords_world_flat_all = head_coords_world_all.permute(0, 2, 3, 1).reshape(2*B*N, 3)
        feats_head_flat_all = feats_head_all.reshape(2*B*N, -1)
        # feats_head_reduced_flat = self.clip_dim_reducer(feats_head_flat_all, head_coords_world_flat_all)
        # feats_head_reduced_all = feats_head_reduced_flat.view(2*B, N, -1)
        
        # Query voxel features and cos simeilarity
        with torch.no_grad():
            voxel_feat_points_hand_flat_all, _ = self.static_map.query_voxel_feature(
                hand_coords_world_flat_all, return_indices=False
            )
            voxel_feat_points_head_flat_all, _ = self.static_map.query_voxel_feature(
                head_coords_world_flat_all, return_indices=False
            )
        
        voxel_feat_points_hand_flat_final_all = self.implicit_decoder(
            voxel_feat_points_hand_flat_all, hand_coords_world_flat_all, return_intermediate=False)
        voxel_feat_points_head_flat_final_all = self.implicit_decoder(
            voxel_feat_points_head_flat_all, head_coords_world_flat_all, return_intermediate=False)
        
        cos_sim_hand = F.cosine_similarity(voxel_feat_points_hand_flat_final_all, feats_hand_flat_all, dim=-1)
        cos_loss_hand = 1.0 - cos_sim_hand.mean()  
        
        cos_sim_head = F.cosine_similarity(voxel_feat_points_head_flat_final_all, feats_head_flat_all, dim=-1)
        cos_loss_head = 1.0 - cos_sim_head.mean()      
        
        total_cos_loss = cos_loss_hand + cos_loss_head 
        
        # Fuse voxel and CLIP features
        fused_hand_all = self.feature_fusion_attn(
            voxel_feat_points_hand_flat_final_all.view(2*B, N, -1),
            feats_hand_flat_all.view(2*B, N, -1),
        ).reshape(2*B*N, -1)
        
        fused_head_all = self.feature_fusion_attn(
            voxel_feat_points_head_flat_final_all.view(2*B, N, -1),
            feats_head_flat_all.view(2*B, N, -1),
        ).reshape(2*B*N, -1)
        
        fused_hand_reduced_all = self.dim_reducer(fused_hand_all, hand_coords_world_flat_all).view(2*B, N, -1)
        fused_head_reduced_all = self.dim_reducer(fused_head_all, head_coords_world_flat_all).view(2*B, N, -1)        
            
        # Text embeddings
        object_labels_all = torch.cat([object_labels, object_labels], dim=0)
        text_emb_reduced = self.text_proj(self.text_embeddings)
        selected_text_reduced_all = text_emb_reduced[object_labels_all, :]

        state_proj_transformer_all = self.state_proj_transformer(state_all)

        # Transformer forward
        out_transformer_all = self.transformer(
            hand_token=fused_hand_reduced_all,
            head_token=fused_head_reduced_all,
            coords_hand=hand_coords_world_flat_all.reshape(2*B, N, 3),
            coords_head=head_coords_world_flat_all.reshape(2*B, N, 3),
            state=state_proj_transformer_all, 
            text_embeddings=selected_text_reduced_all,
            perceiver_out_all=perceiver_out_all,
            hand_translation_all=hand_translation_all,
            head_translation_all=head_translation_all,
        )

        # 8) Split results back to t and t-1
        out_transformer_t, out_transformer_m1 = torch.split(out_transformer_all, B, dim=0)
        state_projected_t, state_projected_m1 = torch.split(self.state_mlp(state_all), B, dim=0)

        # 9) Build final action_t using (state_t, out_t, out_t-1)
        # state_projected_delta = state_projected_t - state_projected_m1
        # out_transformer_delta = out_transformer_t - out_transformer_m1
        
        action_input_t = torch.cat([state_projected_t, state_projected_m1, out_transformer_t, out_transformer_m1], dim=-1)
        action_t = self.action_mlp_multi(action_input_t)

        return action_t, total_cos_loss