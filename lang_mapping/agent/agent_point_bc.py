import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
# Local imports
from lang_mapping.module.transformer import ActionTransformerDecoder, TransformerEncoder
from lang_mapping.module.mlp import StateProj, DimReducer
from lang_mapping.utils.utils import get_3d_coordinates, get_visual_features_dino, transform

class DP3Encoder(nn.Module):
    """
    DP3 Encoder: 3-layer MLP + LayerNorm + Max-Pooling + Projection
    - Input  : (B, N, in_dim) 
    - Output : (B, out_dim) 
    """
    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        
        self.proj = nn.Linear(256, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, in_dim)
        return: (B, out_dim)
        """
        feats = self.mlp(pts)                 # (B, N, out_dim)
        feats_out = self.ln(self.proj(feats))    # (B, out_dim)
        
        return feats_out

class Agent_point_bc(nn.Module):
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
    ):
        super().__init__()

        # --- Feature and Action Dimensions ---
        state_dim = sample_obs["state"].shape[1]
        self.action_dim = np.prod(single_act_shape)

        # --- Camera Intrinsics ---
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

        # --- Vision and Language Pre-trained Models ---
        # DINOv2 for visual features
        # self.vision_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # CLIP for text embeddings
        clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-L-14", "merged2b_s4b_b131k")
        tokenizer = open_clip.get_tokenizer("EVA02-L-14")
        
        if text_input:
            text_input = ['pick up the ' + s.replace('_', ' ') for s in text_input]
        
        text_tokens = tokenizer(text_input)
        with torch.no_grad():
            text_embeddings = clip_model.encode_text(text_tokens)
            self.register_buffer("text_embeddings", F.normalize(text_embeddings, dim=-1, p=2))

        del clip_model, tokenizer

        # --- Agent Modules ---
        self.dp3_encoder = DP3Encoder(in_dim=3, out_dim=transf_input_dim)
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim)
        
        self.transformer_encoder = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=transf_input_dim * 4,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=512,
            transf_input_dim=transf_input_dim,
            nhead=num_heads,
            num_decoder_layers=num_action_layer,
            dim_feedforward=512 * 4,
            dropout=0.1,
            action_dim=self.action_dim,
            action_pred_horizon=action_pred_horizon
        )
        
        self.dim_reducer = DimReducer(clip_input_dim, transf_input_dim)
        self.state_mlp_action = StateProj(state_dim, transf_input_dim)
        
    def _process_sensor_data(self, rgb, depth, pose):   
        if rgb.shape[2] != 3:
            rgb = rgb.permute(0, 1, 4, 2, 3)
            depth = depth.permute(0, 1, 4, 2, 3)

        B, fs, d, H, W = rgb.shape
        rgb = rgb.reshape(B * fs, d, H, W)

        rgb = transform(rgb.float() / 255.0)

        # visfeat = get_visual_features_dino(self.vision_model, rgb)

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

        coords_world_flat = coords_world.permute(0, 2, 3, 1).reshape(B, N, 3)
        feats = self.dp3_encoder(coords_world_flat)
        
        return feats, coords_world_flat
        
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
        coords = torch.cat([coords_hand, coords_head], dim=1)  # (B, N, 3
        
        text_emb = self.text_proj(self.text_embeddings[object_labels]).unsqueeze(1)        # (B,768)
        state_tok = self.state_mlp_action(state).unsqueeze(1) # [B,1,128]
    
        visual_tok = self.transformer_encoder(
            visual_token=feats,
            coords=coords,  
        ) 

        # Max pooling over the sequence dimension to get a single visual token per batch
        visual_tok = torch.amax(visual_tok, dim=1, keepdim=True)
        action_out = self.action_transformer(visual_tok, state_tok, text_emb)
        
        return action_out
