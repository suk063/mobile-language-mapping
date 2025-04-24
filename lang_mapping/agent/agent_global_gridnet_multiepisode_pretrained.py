import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ..module.transformer import TransformerEncoder, GlobalPerceiver, ActionTransformerDecoder, LocalSelfAttentionFusion
from ..module.mlp import ActionMLP, ImplicitDecoder, DimReducer, StateProj, ConcatMLPFusion
from lang_mapping.grid_net import GridNet

from ..utils import get_3d_coordinates, get_visual_features, transform, rotary_pe_3d
import open_clip
import math
from typing import Optional
import timm

class RoPEViTBlock(nn.Module):
    """Wrap a timm ViT block to add 3‑D RoPE + optional causal mask."""

    def __init__(self, block: nn.Module, embed_dim: int =384, new_heads: int = 8):
        super().__init__()
        self.block = block
        self.h     = new_heads
        self.d_h   = embed_dim // new_heads
        
        self.register_buffer(
            "scale",
            torch.tensor(1.0 / math.sqrt(self.d_h), dtype=torch.float32)
        )

    def forward(
        self,
        src: torch.Tensor,                    # [B,S,D]
        coords_src: Optional[torch.Tensor] = None,  # [B,S,3]
        causal_mask: Optional[torch.Tensor] = None, # [S,S] bool
    ) -> torch.Tensor:
        B, S, D = src.shape

        # 1) LN + QKV ---------------------------------------------------
        x = self.block.norm1(src)
        qkv = self.block.attn.qkv(x)                # (B,S,3D)
        qkv = qkv.reshape(B, S, 3, self.h, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv                               # (B,h,S,d_h)

        # 2) 3‑D RoPE ---------------------------------------------------
        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)

        # 3) Attention ---------------------------------------------------
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.block.attn.attn_drop(attn)

        attn_out = torch.matmul(attn, v)            # (B,h,S,d_h)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.block.attn.proj_drop(self.block.attn.proj(attn_out))

        # 4) Residual + MLP --------------------------------------------
        x = src + self.block.drop_path1(attn_out)
        x = x + self.block.drop_path2(self.block.mlp(self.block.norm2(x)))
        return x


class ViTRoPEEncoder(nn.Module):
    def __init__(self, model="vit_small_patch16_224", pretrained=True, num_transf_layers: int = 4):
        super().__init__()

        vit = timm.create_model(model, pretrained=pretrained)

        vit.patch_embed = vit.cls_token = vit.pos_embed = vit.pre_logits = None
        if hasattr(vit, "patch_drop"):
            vit.patch_drop = None

        keep = vit.blocks[-num_transf_layers:]
        self.blocks = nn.ModuleList([RoPEViTBlock(b, new_heads=8) for b in keep])
        self.norm   = vit.norm
        self.embed_dim = vit.embed_dim  # 384

    # ------------------------------------------------------------------
    # internal helper = single ViT pass (was previous forward)
    # ------------------------------------------------------------------
    def _encode(self, src: torch.Tensor, coords_src: Optional[torch.Tensor] = None, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        for blk in self.blocks:
            x = blk(x, coords_src=coords_src, causal_mask=causal_mask)
        x = self.norm(x)
        return x

    # ------------------------------------------------------------------
    # public forward = TransformerEncoder‑style signature
    # ------------------------------------------------------------------
    def forward(
        self,
        hand_token_t, head_token_t, hand_token_m1, head_token_m1,
        coords_hand_t=None, coords_head_t=None,
        coords_hand_m1=None, coords_head_m1=None,
        state_t=None, state_m1=None,
        hand_pose_m1=None, head_pose_m1=None,
        hand_pose_t=None, head_pose_t=None,
        perceiver_token=None,
    ):
        """Build the big token sequence exactly like the original encoder and
        send it through the ViT blocks. Returns the *current‑time* slice
        (after m‑1 + pose tokens), identical to the old behaviour.
        """
        B, N, D = hand_token_t.shape
        tokens, coords = [], []
        zeros1 = lambda: torch.zeros(B, 1, 3, device=hand_token_t.device)

        # 1) m‑1 state + poses + visual
        if state_m1 is not None:
            tokens.append(state_m1.unsqueeze(1)); coords.append(zeros1())
        # tokens += [hand_pose_m1.unsqueeze(1), head_pose_m1.unsqueeze(1)]
        # coords += [zeros1(), zeros1()]
        tokens += [hand_token_m1, head_token_m1]
        coords += [coords_hand_m1, coords_head_m1]

        # 2) current state + poses + visual
        if state_t is not None:
            tokens.append(state_t.unsqueeze(1)); coords.append(zeros1())
        # tokens += [hand_pose_t.unsqueeze(1), head_pose_t.unsqueeze(1)]
        # coords += [zeros1(), zeros1()]
        tokens += [hand_token_t, head_token_t]
        coords += [coords_hand_t, coords_head_t]

        # 3) optional perceiver tokens
        if perceiver_token is not None:
            tokens.append(perceiver_token)
            K = perceiver_token.size(1)
            coords.append(torch.zeros(B, K, 3, device=perceiver_token.device))
        else:
            K = 0

        src = torch.cat(tokens, 1)                    # (B,S,D)
        coords_src = torch.cat(coords, 1) if coords_hand_t is not None else None

        # causal mask ---------------------------------------------------
        len_m1 = 1 + hand_token_m1.size(1) * 2
        len_t  = 1 + hand_token_t.size(1) * 2
        causal_mask = build_causal_mask(len_m1, len_t, K, src.device)

        out = self._encode(src, coords_src, causal_mask)
        return out[:, len_m1 + 1 :, :]


def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def build_causal_mask(n_m1, n_t, n_global, device):
    S    = n_m1 + n_t + n_global
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)

    # m‑1 Query / t Key
    mask[0:n_m1, n_m1:n_m1 + n_t] = True
    
    # m‑1 · t Query / global Key
    # mask[0 : n_m1 + n_t, n_m1 + n_t : ] = True
    
    # global Query / m-1, t global Key
    # mask[n_m1 + n_t :, : n_m1 + n_t] = True

    return mask

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
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_h, device=q.device, dtype=q.dtype))
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
        num_layers: int = 4,
        hidden_dim: int = 1024,
        out_dim: int = 240,
        num_learnable_tokens: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nhead = nhead

        self.num_learnable_tokens = num_learnable_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, self.num_learnable_tokens, input_dim))
        nn.init.xavier_uniform_(self.global_tokens)

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
            out: [B, 16, out_dim]
        """
        B2, N, _ = valid_feats_projected.shape

        # (1) state token
        state_token        = state                         # rename
        coords_state = torch.zeros_like(state[..., :3])         # [B,2,3]

        # (2) learnable tokens
        global_tokens = self.global_tokens.expand(B2, -1, -1)   # [B,num_learnable_tokens,input_dim]
        coords_global = torch.zeros(B2, self.num_learnable_tokens, 3, device=state.device)

        # Combine them: total Q_len=3
        q = torch.cat([state_token, global_tokens], dim=1)      
        coords_q = torch.cat([coords_state, coords_global], dim=1)

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

        out_tokens = x[:, 2:, :]
        out = self.out_proj(out_tokens)  # [B, 16,out_dim]
        return out 

# --------------------------------------------------------------------- #
#                       feature–text gating helper                      #
# --------------------------------------------------------------------- #
def gate_with_text(feats: torch.Tensor,
                   text_embed: torch.Tensor) -> torch.Tensor:
    """
    Residual gating: feats ← feats + feats ✕ cos_sim(feats,text)

    feats       : (B, N, C) 
    text_embed  : (B, 768)
    proj        : nn.Linear that projects 768 → C if dims differ
    """

    # -- match dimensions ----------------------------------------------
    txt = F.normalize(text_embed, dim=-1).unsqueeze(1)                      # (B,1,C)

    # -- cosine‑similarity gating --------------------------------------
    score = (F.normalize(feats, dim=-1) * txt).sum(-1, keepdim=True)  # (B,N,1)
    gated = feats + feats* score                                   # residual

    return gated    

class Agent_global_gridnet_multiepisode_pretrained(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        open_clip_model: tuple = ("EVA02-L-14", "merged2b_s4b_b131k"),
        text_input: list = ["bowl", "apple"],
        clip_input_dim: int = 768,
        transf_input_dim: int = 240,
        state_mlp_dim: int = 128,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
        static_map: GridNet = None,
        implicit_decoder: ImplicitDecoder = None,
        num_heads: int = 8,
        num_layers_transformer: int = 4,
        num_layers_perceiver: int = 4,
        num_learnable_tokens: int = 16,
        num_action_layer: int = 6,
        neighbor_k: int = 512,
        action_horizon: int = 16
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
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            
            text_embeddings, redundant_emb = text_embeddings[:-1, :], text_embeddings[-1:, :]
            self.text_embeddings = text_embeddings - redundant_emb


        # Reduce CLIP feature dimension
        self.dim_reducer_global = nn.Linear(clip_input_dim, transf_input_dim)
        
        # Transformer for feature fusion
        self.transformer = ViTRoPEEncoder(
            model="vit_small_patch16_224",
            pretrained=True,
            num_transf_layers=num_layers_transformer
        )
        
        # Action MLP
        self.action_dim = np.prod(single_act_shape)
        
        self.action_transformer = ActionTransformerDecoder(
            d_model=transf_input_dim,         
            nhead=8,
            num_decoder_layers=num_action_layer,   
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=self.action_dim,
            action_horizon=action_horizon
        )

        # Voxel hashing and implicit decoder
        self.static_map = static_map
        self.implicit_decoder = implicit_decoder
        
        # pose_flat_dim = int(np.prod(sample_obs["pixels"]["fetch_hand_pose"].shape[2:]))  # 3*4 = 12
        # self.pose_proj_hand = nn.Linear(pose_flat_dim, transf_input_dim)
        # self.pose_proj_head = nn.Linear(pose_flat_dim, transf_input_dim)
        
        # self.pose_proj_head_percei = nn.Linear(pose_flat_dim, transf_input_dim) 
        # self.pose_proj_head_action = nn.Linear(pose_flat_dim, transf_input_dim)
        # self.feature_fusion = ConcatMLPFusion(feat_dim=clip_input_dim, clip_embedding_dim=clip_input_dim, output_dim=transf_input_dim, L=10)
        self.feature_fusion_hand = LocalSelfAttentionFusion(feat_dim=clip_input_dim, output_dim=transf_input_dim)
        self.feature_fusion_head = LocalSelfAttentionFusion(feat_dim=clip_input_dim, output_dim=transf_input_dim)
        
        self.state_proj_transf =  nn.Linear(state_dim, transf_input_dim)
        self.state_proj_percei =  nn.Linear(state_dim, transf_input_dim) 

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        self.state_mlp_for_action = nn.Linear(state_dim, transf_input_dim).to(self.device)
        
        # self.perceiver = GlobalPerceiver(input_dim = 240,
        #                                     nhead = 8,
        #                                     num_layers =num_layers_perceiver,
        #                                     hidden_dim = 1024,
        #                                     out_dim = 240,
        #                                     num_learnable_tokens=num_learnable_tokens)
        self.neighbor_k = neighbor_k
    
    @staticmethod
    def _flatten_pose(p):            # p: [B, 1, 3, 4]
        return p.squeeze(1).reshape(p.size(0), -1)      # → [B, 12]
    
    def forward_mapping(self, observations, is_grasp, batch_episode_ids):
        
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

        batch_valid_episode_ids = batch_episode_ids[bool_mask]

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
        depth_mask_head = head_depth_flat_t > 0.6
        
        scene_ids_flat = batch_valid_episode_ids.view(-1, 1).expand(-1, N).reshape(-1, 1)       # (B*N,1)


        # Query voxel features and cos simeilarity
        voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t, scene_ids_flat)
        voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t, scene_ids_flat)

        # Implicit decoder
        # hand
        voxel_feat_points_hand_masked = voxel_feat_points_hand_flat_t[depth_mask_hand]
        coords_camera_hand_masked     = hand_coords_camera_flat_t[depth_mask_hand]
        feats_hand_masked            = feats_hand_flat_t[depth_mask_hand]

        if voxel_feat_points_hand_masked.shape[0] > 0:
            voxel_feat_points_hand_final = self.implicit_decoder(voxel_feat_points_hand_masked)
            cos_sim_hand = F.cosine_similarity(voxel_feat_points_hand_final, feats_hand_masked, dim=-1)
            cos_loss_hand = 1.0 - cos_sim_hand.mean()
        else:
            cos_loss_hand = 0.0

        # head
        voxel_feat_points_head_masked = voxel_feat_points_head_flat_t[depth_mask_head]
        coords_camera_head_masked     = head_coords_camera_flat_t[depth_mask_head]
        feats_head_masked            = feats_head_flat_t[depth_mask_head]

        if voxel_feat_points_head_masked.shape[0] > 0:
            voxel_feat_points_head_final = self.implicit_decoder(voxel_feat_points_head_masked)
            cos_sim_head = F.cosine_similarity(voxel_feat_points_head_final, feats_head_masked, dim=-1)
            cos_loss_head = 1.0 - cos_sim_head.mean()
        else:
            cos_loss_head = 0.0
    
        if cos_loss_hand == 0 and cos_loss_head == 0:
            return 0
        elif cos_loss_hand == 0:
            return cos_loss_head
        elif cos_loss_head == 0:
            return cos_loss_hand
        else:
            return (cos_loss_hand + cos_loss_head) /2
        
    
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
        
        # Query voxel features and cos simeilarity
        scene_ids_flat = batch_episode_ids.view(-1, 1).expand(-1, N).reshape(-1, 1)
        
        with torch.no_grad():
            voxel_feat_points_hand_flat_t = self.static_map.query_feature(hand_coords_world_flat_t, scene_ids_flat)
            voxel_feat_points_head_flat_t = self.static_map.query_feature(head_coords_world_flat_t, scene_ids_flat)
            
            voxel_feat_points_hand_flat_m1 = self.static_map.query_feature(hand_coords_world_flat_m1, scene_ids_flat)
            voxel_feat_points_head_flat_m1 = self.static_map.query_feature(head_coords_world_flat_m1, scene_ids_flat)
    
        voxel_feat_points_hand_flat_final_t = self.implicit_decoder(voxel_feat_points_hand_flat_t)
        voxel_feat_points_head_flat_final_t = self.implicit_decoder(voxel_feat_points_head_flat_t)
        
        voxel_feat_points_hand_flat_final_m1 = self.implicit_decoder(voxel_feat_points_hand_flat_m1)
        voxel_feat_points_head_flat_final_m1 = self.implicit_decoder(voxel_feat_points_head_flat_m1)
        
        # --------------------------------------------------------------------- #
        # 1)  text embeddings for this batch
        # --------------------------------------------------------------------- #
        text_emb = self.text_embeddings[object_labels]        # (B,768)

        # --------------------------------------------------------------------- #
        # 2)  gate BOTH voxel and visual features (t and m‑1, hand & head)
        # --------------------------------------------------------------------- #
        # -- visual CLIP feats (dim = 768) ------------------------------------
        feats_hand_t  = gate_with_text(feats_hand_t,  text_emb)        # (B*N,768)
        feats_head_t  = gate_with_text(feats_head_t,  text_emb)
        feats_hand_m1 = gate_with_text(feats_hand_m1, text_emb)
        feats_head_m1 = gate_with_text(feats_head_m1, text_emb)

        # -- voxel features from implicit decoder (dim = voxel_feature_dim) ---
        voxel_feat_points_hand_final_t  = gate_with_text(
            voxel_feat_points_hand_flat_final_t.view(B, N, -1),  text_emb)
        voxel_feat_points_head_final_t  = gate_with_text(
            voxel_feat_points_head_flat_final_t.view(B, N, -1),  text_emb)
        voxel_feat_points_hand_final_m1 = gate_with_text(
            voxel_feat_points_hand_flat_final_m1.view(B, N, -1), text_emb)
        voxel_feat_points_head_final_m1 = gate_with_text(
            voxel_feat_points_head_flat_final_m1.view(B, N, -1), text_emb)

        fused_hand_t = self.feature_fusion_hand(feats_hand_t, voxel_feat_points_hand_final_t)
        fused_head_t = self.feature_fusion_head(feats_head_t, voxel_feat_points_head_final_t)
        fused_hand_m1 = self.feature_fusion_hand(feats_hand_m1, voxel_feat_points_hand_final_m1)
        fused_head_m1 = self.feature_fusion_head(feats_head_m1, voxel_feat_points_head_final_m1)  

        state_proj_transf_t = self.state_proj_transf(state_t)
        state_proj_transf_m1 = self.state_proj_transf(state_m1)

        # hand_pose_m1_proj = self.pose_proj_hand(self._flatten_pose(
        #     observations["pixels"]["fetch_hand_pose_m1"]
        # ))
        # head_pose_m1_proj = self.pose_proj_head(self._flatten_pose(
        #     observations["pixels"]["fetch_head_pose_m1"]
        # ))
        # hand_pose_t_proj = self.pose_proj_hand(self._flatten_pose(
        #     observations["pixels"]["fetch_hand_pose"]
        # ))
        # head_pose_t_proj = self.pose_proj_head(self._flatten_pose(
        #     observations["pixels"]["fetch_head_pose"]
        # )) 
        # observations["pixels"]["fetch_head_pose"]: shape [B, 1, 3, 4]
        
        # text_token_proj = self.text_proj(text_emb)          
        
        # global voxels
    
        # valid_coords_batch = self.valid_coords[batch_episode_ids.item()].unsqueeze(0).expand(B, -1, -1)
        # B, M, _ = valid_coords_batch.shape
        
        # head_xyz           = head_pose_t[:, 0, :3, 3]                        # (B,3)
        # dist2              = ((valid_coords_batch - head_xyz.unsqueeze(1))**2).sum(-1)
        # _, topk_idx        = torch.topk(dist2, K, dim=1, largest=False)
        # # max_dist = torch.sqrt(dist2.gather(1, topk_idx).max(dim=1).values)

        # batch_idx          = torch.arange(B, device=topk_idx.device).unsqueeze(1)
        # valid_coords_batch = valid_coords_batch[batch_idx, topk_idx]         # (B,K,3)
        

        # scene_ids_valid    = batch_episode_ids.unsqueeze(1).expand(-1, M).reshape(-1,1)
        # valid_coords_flat  = valid_coords_batch.reshape(-1, 3)               # (B*K,3)

        # with torch.no_grad():
        #     voxel_feat_valid_flat = self.static_map.query_feature(valid_coords_flat, scene_ids_valid)

        # voxel_feat_valid = self.implicit_decoder(voxel_feat_valid_flat)      # (B*K,F_dec)
        # voxel_feat_valid = voxel_feat_valid.view(B, K, -1)                   # (B,K,F_dec)

        # voxel_feat_valid = gate_with_text(voxel_feat_valid, text_emb)        # (B,K,F_dec)
        # voxel_feat_valid = self.dim_reducer_global(voxel_feat_valid)           # (B,K,240)

        # coords_voxel_valid = _to_head_frame(valid_coords_batch, head_pose_t) # (B,K,3)
    
        # state_proj_percei_t = self.state_proj_percei(state_t)
        # head_pose_t_proj_percei = self.pose_proj_head_percei(
        #     self._flatten_pose(observations["pixels"]["fetch_head_pose"])  # (B,12)
        # )                                                                  # → (B,240)
        
        # state_token = torch.stack([state_proj_percei_t, head_pose_t_proj_percei], dim=1)

        # perceiver_out = self.perceiver(state_token, coords_voxel_valid, voxel_feat_valid)
        
        # Transformer forward
        out_transformer = self.transformer(
            hand_token_t=fused_hand_t,
            head_token_t=fused_head_t,
            hand_token_m1=fused_hand_m1,
            head_token_m1=fused_head_m1,
            coords_hand_t=hand_coords_world_flat_t.view(B, N, 3),
            coords_head_t=head_coords_world_flat_t.view(B, N, 3),
            coords_hand_m1=hand_coords_world_flat_m1.view(B, N, 3),
            coords_head_m1=head_coords_world_flat_m1.view(B, N, 3),
            state_t=state_proj_transf_t,
            state_m1=state_proj_transf_m1,
            # hand_pose_m1=hand_pose_m1_proj,
            # head_pose_m1=head_pose_m1_proj,
            # hand_pose_t=hand_pose_t_proj,
            # head_pose_t=head_pose_t_proj, 
            # perceiver_token=perceiver_out
        ) # [B, N, 240]
        
        state_t_proj  = self.state_mlp_for_action(state_t).unsqueeze(1)    # [B, 240]
        
        action_out = self.action_transformer(out_transformer, state_t_proj)
        
        return action_out