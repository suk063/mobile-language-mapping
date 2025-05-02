import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Local imports
from ..module.transformer import ActionTransformerDecoder, TransformerEncoder
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj
from ..module.global_module import HierarchicalSceneTransformer, LocalFeatureFusion

from lang_mapping.grid_net import GridNet

from ..utils import get_3d_coordinates, get_visual_features, transform, gate_with_text
import open_clip

class Agent_global_gridnet_multiepisode_pointtrans(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        transf_input_dim: int = 768,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers_transformer: int = 4,
        num_action_layer: int = 6,
        action_pred_horizon: int = 16,
    ):
        super().__init__()

        self.device = device

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        pose_flat_dim      = int(np.prod(sample_obs["pixels"]["fetch_hand_pose"].shape[2:]))
        raw_state_dim      = sample_obs["state"].shape[1]        # 42
        
        # state_dim = raw_state_dim + pose_flat_dim
        state_dim = raw_state_dim
        
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
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            self.text_embeddings  = F.normalize(text_embeddings, dim=-1, p=2)
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            self.text_embeddings = text_embeddings - redundant_emb
        
        # Transformer for feature fusion
        self.dim_reducer_hand = DimReducer(clip_input_dim, transf_input_dim, L=0)
        self.dim_reducer_head = DimReducer(clip_input_dim, transf_input_dim, L=0)
        
        self.transformer = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=transf_input_dim * 4,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )
        
        # Action MLP
        self.action_dim = np.prod(single_act_shape)
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=transf_input_dim,         
            nhead=8,
            num_decoder_layers=num_action_layer,   
            dim_feedforward=transf_input_dim * 4,
            dropout=0.1,
            action_dim=self.action_dim,
            action_pred_horizon=action_pred_horizon
        )

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder
        
        self.state_proj =  StateProj(state_dim, transf_input_dim)
        # self.voxel_proj = VoxelProj(voxel_feature_dim=transf_input_dim) 

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.state_mlp_action = StateProj(state_dim, transf_input_dim)
        
        self.transf_input_dim   = transf_input_dim
        
        self.scene_encoder = HierarchicalSceneTransformer(
            in_dim  = transf_input_dim,
            out_dim = transf_input_dim,
        )
        
        # self.bica = BiCrossAttnRoPE(dim=transf_input_dim, heads=8)
        self.local_fuser = LocalFeatureFusion(
            dim       = transf_input_dim,
            n_heads   = num_heads,
            ff_mult   = 4,
            radius    = 0.3, 
            k         = 1,
            dropout   = 0.1
        )
    
    @staticmethod
    def _flatten_pose(p):            # p: [B, 1, 3, 4]
        return p.squeeze(1).reshape(p.size(0), -1)      # → [B, 12]

    def _gather_scene_kv(
        self,
        batch_episode_ids: torch.Tensor,         # [B]
        text_emb:          torch.Tensor,         # [B,768]
        level_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        kv_coords     : (B,  L_max, 3)   padded with 0
        kv_feats      : (B,  L_max, C)   padded with 0
        kv_pad_mask   : (B,  L_max)      True ⟹ pad
        """
        B                              = batch_episode_ids.size(0)
        scene_ids                       = batch_episode_ids.tolist()
        per_scene_coords, per_scene_len = [], []

        # ── 1) collect coordinates per scene ──────────────────────────────
        for sid in scene_ids:
            c = self.valid_coords[level_idx][int(sid)].to(self.device)     # (L_i,3)
            per_scene_coords.append(c)
            per_scene_len.append(c.size(0))

        L_max = max(per_scene_len)

        kv_coords   = torch.zeros(B, L_max, 3,                   device=self.device)
        kv_feats    = torch.zeros(B, L_max, self.transf_input_dim, device=self.device)
        kv_pad_mask = torch.ones( B, L_max, dtype=torch.bool,    device=self.device)

        # ── 2) query voxel features scene-wise ────────────────────────────
        for b, (sid, coords) in enumerate(zip(scene_ids, per_scene_coords)):
            L                     = coords.size(0)
            kv_coords  [b, :L]    = coords
            kv_pad_mask[b, :L]    = False                         # not pad

            scene_ids_tensor      = torch.full((L, 1), sid, device=self.device)
            with torch.no_grad():
                vox_raw           = self.static_map.query_feature(coords, scene_ids_tensor)
                vox_feat          = self.implicit_decoder(vox_raw)          # (L,F_dec)

            vox_feat              = gate_with_text(vox_feat.unsqueeze(0), text_emb[b:b+1]).squeeze(0)
            kv_feats   [b, :L]    = vox_feat

        return kv_coords, kv_feats, kv_pad_mask
    
    def forward_policy(self, observations, object_labels, batch_episode_ids):

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
        
        # state_t_cat  = torch.cat([state_t,  head_pose_flat_t],  dim=1)
        # state_m1_cat = torch.cat([state_m1, head_pose_flat_m1], dim=1)

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
            hand_depth_t = F.interpolate(hand_depth_t, (16, 16), mode="nearest-exact")
            head_depth_t = F.interpolate(head_depth_t, (16, 16), mode="nearest-exact")
            
            hand_depth_m1 = hand_depth_m1.view(B, fs * d2, H, W)
            head_depth_m1 = head_depth_m1.view(B, fs * d2, H, W)
            hand_depth_m1 = F.interpolate(hand_depth_m1, (16, 16), mode="nearest-exact")
            head_depth_m1 = F.interpolate(head_depth_m1, (16, 16), mode="nearest-exact")

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

        hand_coords_world_flat_t = hand_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_t = head_coords_world_t.permute(0, 2, 3, 1).reshape(B*N, 3)        
        hand_coords_world_flat_m1 = hand_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        head_coords_world_flat_m1 = head_coords_world_m1.permute(0, 2, 3, 1).reshape(B*N, 3)
        
        # --------------------------------------------------------------------- #
        # 1)  text embeddings for this batch
        # --------------------------------------------------------------------- #
        text_emb = self.text_embeddings[object_labels]        # (B,768)
        
        feats_hand_t  = gate_with_text(feats_hand_t,  text_emb)        # (B*N,768)
        feats_head_t  = gate_with_text(feats_head_t,  text_emb)
        feats_hand_m1 = gate_with_text(feats_hand_m1, text_emb)
        feats_head_m1 = gate_with_text(feats_head_m1, text_emb)
        
        feats_hand_t  = self.dim_reducer_hand(feats_hand_t.reshape(B*N, -1)).reshape(B, N, -1) 
        feats_head_t  = self.dim_reducer_head(feats_head_t.reshape(B*N, -1)).reshape(B, N, -1)
        feats_hand_m1 = self.dim_reducer_hand(feats_hand_m1.reshape(B*N, -1)).reshape(B, N, -1)
        feats_head_m1 = self.dim_reducer_head(feats_head_m1.reshape(B*N, -1)).reshape(B, N, -1)

        state_proj_t = self.state_proj(state_t)
        state_proj_m1 = self.state_proj(state_m1)      
        
        coords_hand_t = hand_coords_world_flat_t.view(B, N, 3)
        coords_head_t = head_coords_world_flat_t.view(B, N, 3)
        
        coords_hand_m1 = hand_coords_world_flat_m1.view(B, N, 3)
        coords_head_m1 = head_coords_world_flat_m1.view(B, N, 3)
        
        kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0 = self._gather_scene_kv(batch_episode_ids, text_emb, 0)
        
        feats_hand_t  = self.local_fuser(
            coords_hand_t, feats_hand_t,
            kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0
        )
        feats_head_t  = self.local_fuser(
            coords_head_t, feats_head_t,
            kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0
        )
        feats_hand_m1 = self.local_fuser(
            coords_hand_m1, feats_hand_m1,
            kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0
        )
        feats_head_m1 = self.local_fuser(
            coords_head_m1, feats_head_m1,
            kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0
        )
        
        # kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0 = self._gather_scene_kv(batch_episode_ids, text_emb, 0)
        
        pts_kv   = torch.cat([kv_coords_lvl0, kv_feats_lvl0], dim=-1)            # [B,L, 3+768]
        global_coords, global_tok = self.scene_encoder(pts_kv, kv_pad_lvl0)     
           
        # Transformer forward
        visual_tok = self.transformer(
            hand_token_t=feats_hand_t,
            head_token_t=feats_head_t,
            hand_token_m1=feats_hand_m1,
            head_token_m1=feats_head_m1,
            coords_hand_t=coords_hand_t,
            coords_head_t=coords_head_t,
            coords_hand_m1=coords_hand_m1,
            coords_head_m1=coords_head_m1,
            state_t=state_proj_t.unsqueeze(1),
            state_m1=state_proj_m1.unsqueeze(1)
        ) 
        

        state_tok  = self.state_mlp_action(state_t).unsqueeze(1) # [B,1,128]
        
        cond_tok   = torch.cat([visual_tok, global_tok], dim=1) 
        action_out = self.action_transformer(cond_tok, state_tok)
        
        return action_out