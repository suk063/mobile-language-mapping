import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
# Local imports
from lang_mapping.module.transformer import ActionTransformerDecoder, TransformerEncoder, LocalFeatureFusion
from lang_mapping.module.mlp import StateProj, DimReducer
from lang_mapping.utils.utils import get_3d_coordinates, get_visual_features_dino, transform
from lang_mapping.mapper.mapper import MultiVoxelHashTable
from lang_mapping.module.mlp import ImplicitDecoder
from lang_mapping.module.scene_encoder import GlobalSceneEncoder

class Agent_map_bc(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        text_input: list,
        transf_input_dim: int,
        num_heads: int,
        num_layers_transformer: int,
        clip_input_dim: int,
        num_action_layer: int,
        action_pred_horizon: int,
        camera_intrinsics: list[float],
        static_maps: MultiVoxelHashTable = None,
        implicit_decoder: ImplicitDecoder = None,
    ):
        super().__init__()

        # --- Feature and Action Dimensions ---
        state_dim = sample_obs["state"].shape[1]
        self.action_dim = np.prod(single_act_shape)

        # --- Camera Intrinsics ---
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

        # --- Vision and Language Pre-trained Models ---
        # DINOv2 for visual features
        self.vision_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # CLIP for text embeddings
        clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-L-14", "merged2b_s4b_b131k")
        tokenizer = open_clip.get_tokenizer("EVA02-L-14")
        
        if text_input:
            text_input = ['pick up the' + s.replace('_', ' ') for s in text_input]
        
        text_tokens = tokenizer(text_input)
        with torch.no_grad():
            text_embeddings = clip_model.encode_text(text_tokens)
            self.register_buffer("text_embeddings", F.normalize(text_embeddings, dim=-1, p=2))

        del clip_model, tokenizer

        # --- Agent Modules ---
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim)
        
        self.transformer_encoder = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=transf_input_dim * 4,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=transf_input_dim,
            nhead=num_heads,
            num_decoder_layers=num_action_layer,
            dim_feedforward=transf_input_dim * 4,
            dropout=0.1,
            action_dim=self.action_dim,
            action_pred_horizon=action_pred_horizon
        )
        
        self.local_feature_fusion = LocalFeatureFusion(
            dim=transf_input_dim,
            n_heads=num_heads,
            ff_mult=4,
            radius=0.4,
            k=8,
        )
        
        self.dim_reducer = DimReducer(clip_input_dim, transf_input_dim)
        self.state_mlp_action = StateProj(state_dim, transf_input_dim)
        
        self.scene_encoder = GlobalSceneEncoder(
            out_dim=transf_input_dim,
            heads=num_heads,
            dropout=0.1
        )

        if static_maps is not None:
            object.__setattr__(self, "static_maps", static_maps)

        if implicit_decoder is not None:
            object.__setattr__(self, "implicit_decoder", implicit_decoder)
        
    def _process_sensor_data(self, rgb, depth, pose):   
        if rgb.shape[2] != 3:
            rgb = rgb.permute(0, 1, 4, 2, 3)
            depth = depth.permute(0, 1, 4, 2, 3)

        B, fs, d, H, W = rgb.shape
        rgb = rgb.reshape(B * fs, d, H, W)

        rgb = transform(rgb.float() / 255.0)

        visfeat = get_visual_features_dino(self.vision_model, rgb)

        depth = depth / 1000.0
        
        _, fs, d2, H, W = depth.shape
        depth = depth.view(B * fs, d2, H, W)
        depth = F.interpolate(depth, (16, 16), mode="nearest-exact")

        coords_world, _ = get_3d_coordinates(
            depth, pose, 
            self.fx, self.fy, self.cx, self.cy
        )

        _, _, Hf, Wf = coords_world.shape
        N = Hf * Wf

        feats = visfeat.permute(0, 2, 3, 1).reshape(B, N, -1)
        coords_world_flat = coords_world.permute(0, 2, 3, 1).reshape(B, N, 3)
        
        return feats, coords_world_flat

    def _gather_static_map_features(self, scene_ids, level_index):
        """
        Parameters
        ----------
        scene_ids : (B,) LongTensor
            The scene-id for every sample in the batch.
        level_index : int
            Which pyramid level of the voxel hash table to query.

        Returns
        -------
        kv_coords : Tensor (B, max_len, 3)
            World coordinates of the queried voxels (zero padded).
        kv_feats : Tensor (B, max_len, D)
            Corresponding voxel features where D = num_levels * feature_dim.
        kv_pad_mask : BoolTensor (B, max_len)
            Padding mask where **True** indicates padded (invalid) positions.
        """
        device = scene_ids.device
        B = scene_ids.shape[0]

        # Collect per-scene coordinate tensors
        per_scene_coords = [
            self.static_maps.voxel_hash_tables[sid].levels[level_index].coords
            for sid in scene_ids.tolist()
        ]
        max_len = max(coords.shape[0] for coords in per_scene_coords)

        # Output dimensions
        feat_dim = self.static_maps.num_levels * self.static_maps.feature_dim

        kv_coords = torch.zeros(B, max_len, 3, device=device, dtype=per_scene_coords[0].dtype)
        kv_feats = torch.zeros(B, max_len, feat_dim, device=device, dtype=torch.float32)
        kv_pad_mask = torch.ones(B, max_len, device=device, dtype=torch.bool)

        # Fill tensors scene-by-scene
        for i, (sid, coords) in enumerate(zip(scene_ids.tolist(), per_scene_coords)):
            n = coords.shape[0]
            kv_coords[i, :n] = coords
            kv_pad_mask[i, :n] = False
            sid_tensor = torch.full((n,), sid, device=device, dtype=torch.long)
            kv_feats[i, :n] = self.static_maps.query_feature(coords, sid_tensor)

        kv_feats = self.dim_reducer(self.implicit_decoder(kv_feats.reshape(B * max_len, -1))).reshape(B, max_len, -1)

        return kv_coords, kv_feats, kv_pad_mask
        
    def forward(self, observations, object_labels, scene_ids):
        
        state  = observations["state"].squeeze(1)
        
        feats_hand, coords_hand = self._process_sensor_data(
            observations["fetch_hand_rgb"],
            observations["fetch_hand_depth"],
            observations["fetch_hand_pose"],
        )
        
        feats_head, coords_head = self._process_sensor_data(
            observations["fetch_head_rgb"],
            observations["fetch_head_depth"],
            observations["fetch_head_pose"],
        )
        
        feats = torch.cat([feats_hand, feats_head], dim=1)
        coords = torch.cat([coords_hand, coords_head], dim=1)  # (B, N, 3)

        kv_coords, kv_feats, kv_pad_mask = self._gather_static_map_features(scene_ids, 1)
        
        # Global scene encoding
        pts_kv   = torch.cat([kv_coords, kv_feats], dim=-1)            # [B,L, 3+768]
        global_coords, global_tok = self.scene_encoder(pts_kv, kv_pad_mask)          
        
        # Local feature fusion
        feats = self.local_feature_fusion(coords, feats, kv_coords, kv_feats, kv_pad_mask)
        
        text_emb = self.text_proj(self.text_embeddings[object_labels]).unsqueeze(1)        # (B,768)
        state_tok = self.state_mlp_action(state).unsqueeze(1) # [B,1,128]
    
        visual_tok = self.transformer_encoder(
            visual_token=feats,
            coords=coords,  
        ) 
        
        action_out = self.action_transformer(visual_tok, global_tok, text_emb, state_tok)
        
        return action_out
