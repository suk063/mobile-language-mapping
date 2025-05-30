import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local imports
from ..module.transformer import ActionTransformerDecoder, TransformerEncoder, LocalSelfAttentionFusion
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj, VoxelProj
from ..module.global_module import HierarchicalSceneTransformer, LocalFeatureFusion

from lang_mapping.grid_net import GridNet

from ..utils import get_3d_coordinates, get_visual_features, transform, gate_with_text
from torch.nn.utils.rnn import pad_sequence
import open_clip

import os, glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF          # for tensor→PIL
from PIL import Image
from matplotlib.cm import get_cmap

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _maybe_unnormalise(img: torch.Tensor) -> torch.Tensor:
    if img.min() < 0 or img.max() > 1:          # heuristic for ImageNet-style norm
        img = img * _IMAGENET_STD.to(img.device) + _IMAGENET_MEAN.to(img.device)
    return img.clamp(0, 1)

def exp_minmax(x: torch.Tensor, gamma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential min–max normalization.
    Step 1 : linear min–max                       (0‒1)
    Step 2 : exponential re-weighting  y = exp(γ·(x-1))  → (0‒1/e^γ, 1]
             (γ > 0 boosts high values; γ < 0 boosts low values)
    Step 3 : linear min–max again to restore exact 0‒1 range.
    """
    # ① linear min-max
    x0 = (x - x.min()) / (x.max() - x.min() + eps)

    # ② exponential shaping
    y  = torch.exp(gamma * (x0 - 1.0))        # shift so that x0=1 → y=1
    # ③ rescale to [0,1]
    y  = (y - y.min()) / (y.max() - y.min() + eps)
    return y

def visualise_hand_and_feats(
    hand_rgb_t:      torch.Tensor,   # (B,3,H,W)
    feats_hand_gate: torch.Tensor,   # (B,N,C)
    feats_hand_fused:torch.Tensor,   # (B,N,C)
    save_dir: str = "vis_mag",
    out_res: int = 512,              # NEW – target resolution (square)
):
    """
    저장 파일
      rgb_<k>.png          : 원본 색상
      gate_mag_<k>.png     : 512×512 Viridis heat-map (gate output)
      fused_mag_<k>.png    : 512×512 Viridis heat-map (local fuser output)
    """
    os.makedirs(save_dir, exist_ok=True)

    existing = glob.glob(os.path.join(save_dir, "rgb_*.png"))
    start_k  = (
        max(int(os.path.splitext(os.path.basename(p))[0].split('_')[1]) for p in existing) + 1
        if existing else 0
    )

    B, N, C = feats_hand_gate.shape
    H = W = int(N ** 0.5)                              # patch grid (e.g., 16×16)
    viridis = get_cmap("viridis")

    for b in range(B):
        k = start_k + b

        # 1) RGB ------------------------------------------------------------------
        rgb_tensor = _maybe_unnormalise(hand_rgb_t[b].cpu())
        TF.to_pil_image(rgb_tensor).save(f"{save_dir}/rgb_{k}.png")

        # 2) magnitude heat-maps --------------------------------------------------
        for tag, feats in [("gate_mag", feats_hand_gate), ("fused_mag", feats_hand_fused)]:
            mag = feats[b].norm(p=2, dim=-1)                 # (N,)  ‖v‖₂
            mag_map = mag.view(H, W).cpu()

            # min–max → [0,1]
            # mag_norm = mag_norm = exp_minmax(mag_map, gamma=2.0)

            # apply Viridis → RGB uint8
            color_img = (viridis(mag_map.numpy())[:, :, :3] * 255).astype("uint8")

            # upscale to (out_res, out_res)
            upscaled = Image.fromarray(color_img).resize(
                (out_res, out_res), resample=Image.NEAREST
            )
            upscaled.save(f"{save_dir}/{tag}_{k}.png")

def visualise_hand_and_textsim(
    hand_rgb_t:      torch.Tensor,   # (B,3,H,W)
    feats_hand_gate: torch.Tensor,   # (B,N,C)
    feats_hand_fused:torch.Tensor,   # (B,N,C)
    text_emb:        torch.Tensor,   # (B,C)
    view:            str,            # "hand" or "head" ⇒ 파일 prefix
    save_dir: str = "vis_sim",
    out_res: int = 512,
):
    """
    저장 파일
      <view>_rgb_<k>.png
      <view>_gate_sim_<k>.png
      <view>_fused_sim_<k>.png
    """
    os.makedirs(save_dir, exist_ok=True)

    existing = glob.glob(os.path.join(save_dir, f"{view}_rgb_*.png"))
    start_k  = (
        max(int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in existing) + 1
        if existing else 0
    )

    B, N, C = feats_hand_gate.shape
    H = W = int(N ** 0.5)
    viridis = get_cmap("viridis")

    for b in range(B):
        k = start_k + b

        # 1) RGB ------------------------------------------------------------------
        rgb_tensor = _maybe_unnormalise(hand_rgb_t[b].cpu())
        TF.to_pil_image(rgb_tensor).save(f"{save_dir}/{view}_rgb_{k}.png")

        # 2) text-similarity heat-maps -------------------------------------------
        for tag, feats in [("gate_sim", feats_hand_gate), ("fused_sim", feats_hand_fused)]:
            f_norm = F.normalize(feats[b], dim=-1)
            t_norm = F.normalize(text_emb[b], dim=-1)
            sim    = (f_norm * t_norm).sum(dim=-1).view(H, W).cpu()


            sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            color_img = (viridis(sim_norm.numpy())[:, :, :3] * 255).astype("uint8")

            Image.fromarray(color_img).resize((out_res, out_res), Image.NEAREST) \
                .save(f"{save_dir}/{view}_{tag}_{k}.png")
            
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
            # text_input += [""]
            text_input = [s.replace('_', ' ') for s in text_input] + [""]
        
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
        self.voxel_proj = DimReducer(clip_input_dim, transf_input_dim, L=0)

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
            radius    = 0.2, 
            k         = 8,
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

                vox_raw           = self.static_map.query_feature(coords, scene_ids_tensor)
                vox_feat          = self.implicit_decoder(vox_raw)          # (L,F_dec)

                vox_feat              = gate_with_text(vox_feat.unsqueeze(0), text_emb[b:b+1]).squeeze(0)
                kv_feats   [b, :L]    = self.voxel_proj(vox_feat)

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
        
        hand_rgb_save= hand_rgb_t.float() / 255
        head_rgb_save= head_rgb_t.float() / 255
        
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
        
        feats_hand_gate = feats_hand_t.clone()
        feats_head_gate = feats_head_t.clone()

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
        
        feats_hand_gate = feats_hand_t.clone()
        
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
        
        
        # Local feature fusion
        kv_coords_lvl1, kv_feats_lvl1, kv_pad_lvl1 = self._gather_scene_kv(batch_episode_ids, text_emb, 1)
        
        feats_hand_t  = self.local_fuser(
            coords_hand_t, feats_hand_t,
            kv_coords_lvl1, kv_feats_lvl1, kv_pad_lvl1
        )
        feats_head_t  = self.local_fuser(
            coords_head_t, feats_head_t,
            kv_coords_lvl1, kv_feats_lvl1, kv_pad_lvl1
        )
        feats_hand_m1 = self.local_fuser(
            coords_hand_m1, feats_hand_m1,
            kv_coords_lvl1, kv_feats_lvl1, kv_pad_lvl1
        )
        feats_head_m1 = self.local_fuser(
            coords_head_m1, feats_head_m1,
            kv_coords_lvl1, kv_feats_lvl1, kv_pad_lvl1
        )
        
        visualise_hand_and_textsim(
            hand_rgb_t       = hand_rgb_save.clamp(0, 1),
            feats_hand_gate  = feats_hand_gate,
            feats_hand_fused = feats_hand_t,
            text_emb         = text_emb,
            view             = "hand",
        )

        visualise_hand_and_textsim(
            hand_rgb_t       = head_rgb_save.clamp(0, 1),
            feats_hand_gate  = feats_head_gate,
            feats_hand_fused = feats_head_t,
            text_emb         = text_emb,
            view             = "head",
        )
        
        kv_coords_lvl0, kv_feats_lvl0, kv_pad_lvl0 = self._gather_scene_kv(batch_episode_ids, text_emb, 0)
        
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
