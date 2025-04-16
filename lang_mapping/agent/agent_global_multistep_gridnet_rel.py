import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from lang_mapping.module.mlp import ImplicitDecoder
from lang_mapping.grid_net import GridNet
from lang_mapping.utils import get_3d_coordinates, get_visual_features, transform

import open_clip

class Agent_global_multistep_gridnet_rel(nn.Module):
    def __init__(
        self,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
    ):
        super().__init__()

        self.device = device

        # Load CLIP model
        clip_model, _, _ = open_clip.create_model_and_transforms(
            open_clip_model[0], pretrained=open_clip_model[1]
        )
        self.clip_model = clip_model.to(self.device)
        
        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

    def forward_mapping(self, observations, is_grasp):
        
        bool_mask = (is_grasp < 0.5)  
        if bool_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 1) Extract data
        hand_rgb_t   = observations["pixels"]["fetch_hand_rgb"][bool_mask]
        head_rgb_t   = observations["pixels"]["fetch_head_rgb"][bool_mask]

        hand_depth_t  = observations["pixels"]["fetch_hand_depth"][bool_mask]
        head_depth_t  = observations["pixels"]["fetch_head_depth"][bool_mask]

        hand_pose_t   = observations["pixels"]["fetch_hand_pose"][bool_mask]
        head_pose_t   = observations["pixels"]["fetch_head_pose"][bool_mask]

        B = hand_rgb_t.shape[0]
        
        # If needed, permute hand_rgb_t so channel=3
        if hand_rgb_t.shape[2] != 3:
            hand_rgb_t = hand_rgb_t.permute(0, 1, 4, 2, 3)
            head_rgb_t = head_rgb_t.permute(0, 1, 4, 2, 3)
        
        # Flatten frames
        _, fs, d, H, W = hand_rgb_t.shape
        hand_rgb_t = hand_rgb_t.reshape(B, fs * d, H, W)
        head_rgb_t = head_rgb_t.reshape(B, fs * d, H, W)
        
        # Transform to [0,1], apply normalization
        hand_rgb_t = transform(hand_rgb_t.float() / 255.0)
        head_rgb_t = transform(head_rgb_t.float() / 255.0)

        with torch.no_grad():
            hand_visfeat_t = get_visual_features(self.clip_model, hand_rgb_t)
            head_visfeat_t = get_visual_features(self.clip_model, head_rgb_t)
        
        # Handle depth (reshape, interpolate)
        hand_depth_t = hand_depth_t / 1000.0
        head_depth_t = head_depth_t / 1000.0
        
        if hand_depth_t.dim() == 5:
            _, fs, d2, H, W = hand_depth_t.shape
            hand_depth_t = hand_depth_t.view(B, fs * d2, H, W)
            head_depth_t = head_depth_t.view(B, fs * d2, H, W)
            # hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest")
            # head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest")
            hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest-exact")
            head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest-exact")

        # 3D world coords
        hand_coords_world_t, hand_coords_camera_t = get_3d_coordinates(
            hand_depth_t, hand_pose_t, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_t, head_coords_camera_t = get_3d_coordinates(
            head_depth_t, head_pose_t,
            self.fx, self.fy, self.cx, self.cy
        )

        # Reduce CLIP dimension for hand/head
        _, C_, Hf, Wf = hand_coords_world_t.shape
        N = Hf * Wf

        feats_hand_t = hand_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
        feats_head_t = head_visfeat_t.permute(0, 2, 3, 1).reshape(B, N, -1)
                
        feats_hand_flat_t = feats_hand_t.reshape(B*N, -1)
        feats_head_flat_t = feats_head_t.reshape(B*N, -1)                
                
        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        hand_coords_camera_flat_t = hand_coords_camera_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_camera_flat_t = head_coords_camera_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        # filtering out points
        hand_depth_flat_t = hand_depth_t.reshape(B*N)
        head_depth_flat_t = head_depth_t.reshape(B*N)

        depth_mask_hand = hand_depth_flat_t > 0.3  
        depth_mask_head = head_depth_flat_t > 0.3

        # Query voxel features and cos simeilarity
        voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t)
        voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t)

        # Implicit decoder
        # hand
        voxel_feat_points_hand_masked = voxel_feat_points_hand_flat_t[depth_mask_hand]
        coords_camera_hand_masked     = hand_coords_camera_flat_t[depth_mask_hand]
        feats_hand_masked            = feats_hand_flat_t[depth_mask_hand]

        if voxel_feat_points_hand_masked.shape[0] > 0:
            voxel_feat_points_hand_final = self.implicit_decoder(
                voxel_feat_points_hand_masked, coords_camera_hand_masked
            )
            cos_sim_hand = F.cosine_similarity(voxel_feat_points_hand_final, feats_hand_masked, dim=-1)
            cos_loss_hand = 1.0 - cos_sim_hand.mean()
        else:
            cos_loss_hand = 0.0

        # head
        voxel_feat_points_head_masked = voxel_feat_points_head_flat_t[depth_mask_head]
        coords_camera_head_masked     = head_coords_camera_flat_t[depth_mask_head]
        feats_head_masked            = feats_head_flat_t[depth_mask_head]

        if voxel_feat_points_head_masked.shape[0] > 0:
            voxel_feat_points_head_final = self.implicit_decoder(
                voxel_feat_points_head_masked, coords_camera_head_masked
            )
            cos_sim_head = F.cosine_similarity(voxel_feat_points_head_final, feats_head_masked, dim=-1)
            cos_loss_head = 1.0 - cos_sim_head.mean()
        else:
            cos_loss_head = 0.0
            
        total_cos_loss = cos_loss_hand + cos_loss_head
        return total_cos_loss
    
    def forward_policy(self, observations, object_labels):
        
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
            hand_depth_t, hand_pose_t, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_t, _ = get_3d_coordinates(
            head_depth_t, head_pose_t,
            self.fx, self.fy, self.cx, self.cy
        )
        
        hand_coords_world_m1, _ = get_3d_coordinates(
            hand_depth_m1, hand_pose_m1, 
            self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world_m1, _ = get_3d_coordinates(
            head_depth_m1, head_pose_m1,
            self.fx, self.fy, self.cx, self.cy
        )

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
        # feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_t, hand_coords_world_flat_t)
        # feats_hand_reduced_t = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_t = feats_head_t_gated.reshape(B*N, -1)
        # feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_t, head_coords_world_flat_t)
        # feats_head_reduced_t = feats_head_reduced_flat.view(B, N, -1)
        
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_hand_flat_m1 = feats_hand_m1_gated.reshape(B*N, -1)
        # feats_hand_reduced_flat = self.dim_reducer_hand(feats_hand_flat_m1, hand_coords_world_flat_m1)
        # feats_hand_reduced_m1 = feats_hand_reduced_flat.view(B, N, -1)

        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        feats_head_flat_m1 = feats_head_m1_gated.reshape(B*N, -1)
        # feats_head_reduced_flat = self.dim_reducer_head(feats_head_flat_m1, head_coords_world_flat_m1)
        # feats_head_reduced_m1 = feats_head_reduced_flat.view(B, N, -1)
        
        # Query voxel features and cos simeilarity
        with torch.no_grad():
            voxel_feat_points_hand_flat_t, _ = self.static_map.query_voxel_feature(hand_coords_world_flat_t)
            voxel_feat_points_head_flat_t, _ = self.static_map.query_voxel_feature(head_coords_world_flat_t)
            
            voxel_feat_points_hand_flat_m1, _ = self.static_map.query_voxel_feature(hand_coords_world_flat_m1)
            voxel_feat_points_head_flat_m1, _ = self.static_map.query_voxel_feature(head_coords_world_flat_m1)
    
        voxel_feat_points_hand_flat_final_t = self.implicit_decoder(
            voxel_feat_points_hand_flat_t, hand_coords_world_flat_t, return_intermediate=False)
        voxel_feat_points_head_flat_final_t = self.implicit_decoder(
            voxel_feat_points_head_flat_t, head_coords_world_flat_t, return_intermediate=False)
        
        voxel_feat_points_hand_flat_final_m1 = self.implicit_decoder(
            voxel_feat_points_hand_flat_m1, hand_coords_world_flat_m1, return_intermediate=False)
        voxel_feat_points_head_flat_final_m1 = self.implicit_decoder(
            voxel_feat_points_head_flat_m1, head_coords_world_flat_m1, return_intermediate=False)
        
        # Fuse voxel and CLIP features
        
        voxel_feat_points_hand_flat_final_t = voxel_feat_points_hand_flat_final_t.detach()
        voxel_feat_points_head_flat_final_t = voxel_feat_points_head_flat_final_t.detach()
        voxel_feat_points_hand_flat_final_m1 = voxel_feat_points_hand_flat_final_m1.detach()
        voxel_feat_points_head_flat_final_m1 = voxel_feat_points_head_flat_final_m1.detach()
        
        fused_hand_t = self.feature_fusion_attn_hand(
            voxel_feat_points_hand_flat_final_t.view(B, N, -1),
            feats_hand_flat_t.view(B, N, -1),
        ).reshape(B*N, -1)
        
        fused_head_t = self.feature_fusion_attn_head(
            voxel_feat_points_head_flat_final_t.view(B, N, -1),
            feats_head_flat_t.view(B, N, -1),
        ).reshape(B*N, -1)
        
        fused_hand_m1 = self.feature_fusion_attn_hand(
            voxel_feat_points_hand_flat_final_m1.view(B, N, -1),
            feats_hand_flat_m1.view(B, N, -1),
        ).reshape(B*N, -1)
        
        fused_head_m1 = self.feature_fusion_attn_head(
            voxel_feat_points_head_flat_final_m1.view(B, N, -1),
            feats_head_flat_m1.view(B, N, -1),
        ).reshape(B*N, -1)
        
        fused_hand_reduced_t = self.dim_reducer_hand(fused_hand_t, hand_coords_world_flat_t).view(B, N, -1)
        fused_head_reduced_t = self.dim_reducer_head(fused_head_t, head_coords_world_flat_t).view(B, N, -1)
        
        fused_hand_reduced_m1 = self.dim_reducer_hand(fused_hand_m1, hand_coords_world_flat_m1).view(B, N, -1)
        fused_head_reduced_m1 = self.dim_reducer_head(fused_head_m1, head_coords_world_flat_m1).view(B, N, -1)                

        state_proj_transformer_t = self.state_proj_transformer(state_t)
        state_proj_transformer_m1 = self.state_proj_transformer(state_m1)

        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=fused_hand_reduced_t,
            head_token_t=fused_head_reduced_t,
            hand_token_m1=fused_hand_reduced_m1,
            
            head_token_m1=fused_head_reduced_m1,
            coords_hand_t=hand_coords_world_flat_t.reshape(B, N, 3),
            coords_head_t=head_coords_world_flat_t.reshape(B, N, 3),
            coords_hand_m1=hand_coords_world_flat_m1.reshape(B, N, 3),
            coords_head_m1=head_coords_world_flat_m1.reshape(B, N, 3),
            state_t=state_proj_transformer_t,
            state_m1=state_proj_transformer_m1, 
        ) # [B, N, 240]
        
        
        state_t_proj  = self.state_mlp_for_action(state_t).unsqueeze(1)   # [B, 240]
        action_out = self.action_transformer(out_transformer, state_t_proj)
        
        return action_out