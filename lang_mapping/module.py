import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import rotary_pe_3d  
from .utils import positional_encoding

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes a layer's weights orthogonally.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TransformerCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Define multi-head attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,        # [B, S, d_model]
        coords_src: torch.Tensor = None,
        text: torch.Tensor = None,# [B, T, d_model] or None
    ) -> torch.Tensor:
        
        # Self-Attention (use rotary encoding if coords_src is given)
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_rot = src
        else:
            q_rot = k_rot = v_rot = src

        attn_out, _ = self.self_attn(q_rot, k_rot, v_rot)
        src = self.norm1(src + self.dropout1(attn_out))

        # Cross-Attention with text
        if text is not None:
            attn_out_text, _ = self.cross_attn_text(query=src, key=text, value=text)
            src = self.norm2(src + self.dropout2(attn_out_text))

        # FeedForward
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm_ff(src + self.dropout_ff(ff_out))
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=2, num_heads=8, output_dim=1024):
        super().__init__()
        
        self.modality_embed_state = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_text = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_learnable = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_3d = nn.Parameter(torch.randn(1, 1, input_dim))
        
        self.layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * 256, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim)
        )
        
        self.apply(init_weights_kaiming)
        
    def forward(
        self,
        hand: torch.Tensor,        # [B, N, input_dim]
        head: torch.Tensor,        # [B, N, input_dim]
        coords_hand: torch.Tensor = None,
        coords_head: torch.Tensor = None,
        state: torch.Tensor = None,          # [B, input_dim] or None
        text_embeddings: torch.Tensor = None,# [B, input_dim] or None
        global_token: torch.Tensor = None  # [B, input_dim] or None
    ) -> torch.Tensor:
        B, N, D = hand.shape
        # -------------------------------------------------------------------
        # 1) Construct the initial src tokens (self-attention input)
        # -------------------------------------------------------------------
        tokens = []
        coords_list = []
        # If we have state token
        if state is not None:
            state_token = state.unsqueeze(1) + self.modality_embed_state  # [B, 1, input_dim]
            tokens.append(state_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))
        # If we have text token
        if text_embeddings is not None:
            text_token = text_embeddings.unsqueeze(1) + self.modality_embed_text # [B, 1, input_dim]
            tokens.append(text_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))
        # If we have global_token token
        if global_token is not None:     
            Bg, Mg, Dg = global_token.shape
            global_token = global_token + self.modality_embed_learnable
            tokens.append(global_token)
            coords_list.append(torch.zeros(Bg, Mg, 3, device=state.device))
        # Now add hand + head tokens
        hand = hand + self.modality_embed_3d
        head = head + self.modality_embed_3d
        
        tokens.append(hand)  # [B, N, input_dim]
        tokens.append(head)  # [B, N, input_dim]
        # Build coords if provided
        if coords_hand is not None and coords_head is not None:
            coords_list.append(coords_hand)
            coords_list.append(coords_head)
            coords_src = torch.cat(coords_list, dim=1)  # [B, S+2N, 3]
        else:
            coords_src = None
        # Concatenate all tokens along the sequence dimension
        src = torch.cat(tokens, dim=1)  # shape: [B, S+2N, input_dim]
        # -------------------------------------------------------------------
        # 2) Pass through stacked TransformerCrossAttentionLayers
        # -------------------------------------------------------------------
        text_ = text_embeddings.unsqueeze(1) if text_embeddings is not None else None

        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
                text=text_
            )
        # -------------------------------------------------------------------
        # 3) Post-fusion MLP
        # -------------------------------------------------------------------
        num_special = 0
        if state is not None:
            num_special += 1
        if text_embeddings is not None:
            num_special += 1
        if global_token is not None:
            num_special += Mg
        # Example: skip those special tokens
        fused_tokens = src[:, num_special:, :]   # shape: [B, 2N, input_dim]
        data = fused_tokens.reshape(B, -1)       # flatten the remaining
        out = self.post_fusion_mlp(data)         # [B, output_dim]
        return out
    
class ConcatMLPFusion(nn.Module):
    """
    (feat1, feat2, coords_3d) -> [concat + sinusoidal PE] -> MLP -> fused
    """
    def __init__(self, feat_dim=120, L=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.L = L
        
        self.pos_enc_dim = 2 * self.L * 3

        in_dim = feat_dim * 2 + self.pos_enc_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, feat1, feat2, coords_3d):
        """
        Args:
            feat1, feat2: (B * N, feat_dim)
            coords_3d:    (B * N, 3)  
        
        Returns:
            fused: (B * N, feat_dim) 
        """

        x = torch.cat([feat1, feat2], dim=-1)

        pe = positional_encoding(coords_3d, L=self.L)  # (B*N, 2*L*3)

        # feature + positional_encoding을 concat
        x = torch.cat([x, pe], dim=-1)  # (B, N, feat_dim*2 + pos_enc_dim)

        # MLP 
        fused = self.mlp(x)  # (B, N, feat_dim)
        return fused
    
class ActionMLP(nn.Module):
    """
    A feed-forward MLP for producing action outputs from a latent state vector.
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [2048, 1024, 512],
        final_std: float = 0.01 * np.sqrt(2)
    ):
        """
        Args:
            input_dim (int): Dimension of the input (e.g., token + state embedding).
            action_dim (int): Number of action outputs.
            hidden_dims (list): Dimensions of the hidden layers.
            final_std (float): Std for the final layer initialization.
        """
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hdim)))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hdim

        # Final layer
        final_linear = layer_init(
            nn.Linear(prev_dim, action_dim),
            std=final_std
        )
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the action outputs."""
        return self.net(x)

class ImplicitDecoder(nn.Module):
    """
    A simple MLP to decode a 3D coordinate (with positional encoding) and its corresponding
    voxel feature into another feature vector (default 768-D).
    """
    def __init__(self, voxel_feature_dim=768, hidden_dim=768, output_dim=768, L=10):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.pe_dim = 2 * self.L * 3  # 2*L for sine/cosine, times 3 for x, y, z

        self.input_dim = self.voxel_feature_dim + self.pe_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)

        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.apply(init_weights_kaiming)

    def forward(self, voxel_features, coords_3d, return_intermediate=False):
        """
        Args:
            voxel_features (Tensor): [N, voxel_feature_dim=768].
            coords_3d (Tensor): [N, 3].
            return_intermediate (bool)
        Returns:
            Tensor: [N, output_dim=768]
        """
        pe = positional_encoding(coords_3d, L=self.L)  # [N, pe_dim]

        # 1) fc1
        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)

        # 3) fc3 
        x = torch.cat([x, pe], dim=-1)
        x1 = self.fc3(x)
        x = F.relu(self.ln3(x1), inplace=True)
        
        # 4) fc4
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)

        # 5) fc5 
        out = self.fc5(x)

        if return_intermediate:
            return x1, out
        else:
            return out

class StateProj(nn.Module):
    def __init__(self, state_dim=42, output_dim=120):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, output_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, state):
        out = self.mlp(state)
        return out

class VoxelProj(nn.Module):
    def __init__(self, voxel_feature_dim=120, L=6):
        super().__init__()
        self.L = L
        self.pos_enc_dim = 2 * self.L * 3

        self.mlp = nn.Sequential(
            nn.Linear(voxel_feature_dim + self.pos_enc_dim, voxel_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(voxel_feature_dim),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, voxel_feat, coords_3d):
        """
        Args:
            voxel_feat: (N, voxel_feature_dim)
            coords_3d:  (N, 3)
        Returns:
            projected:  (N, voxel_feature_dim)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # (N, 2*L*3)
        x = torch.cat([voxel_feat, pe], dim=-1)
        out = self.mlp(x)
        return out

class LoRALinear(nn.Module):
    def __init__(self, 
                 linear_layer: nn.Linear,
                 rank: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.original_weight = linear_layer.weight
        self.original_bias = linear_layer.bias
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.normal_(self.lora_B, std=0.02)

        self.original_weight.requires_grad_(False)
        if self.original_bias is not None:
            self.original_bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        result = torch.matmul(x, self.original_weight.T)
        if self.original_bias is not None:
            result = result + self.original_bias

        lora_out = torch.matmul(self.dropout(x), self.lora_A.T)  # (batch, rank)
        lora_out = torch.matmul(lora_out, self.lora_B.T)         # (batch, out_features)
        result += self.scaling * lora_out

        return result

class ImplicitDecoderDINO(nn.Module):
    def __init__(self, voxel_feature_dim=768, hidden_dim=768, output_dim_clip=768, output_dim_dino=1024, L=10):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.pe_dim = 2 * self.L * 3  # positional encoding dimension

        # voxel_feat + positional_encoding
        self.input_dim = self.voxel_feature_dim + self.pe_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        # ----------- CLIP BRANCH -----------
        self.fc4_clip = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4_clip = nn.LayerNorm(self.hidden_dim)
        self.fc5_clip = nn.Linear(self.hidden_dim, output_dim_clip)

        # ----------- DINO BRANCH -----------
        self.fc4_dino = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4_dino = nn.LayerNorm(self.hidden_dim)
        self.fc5_dino = nn.Linear(self.hidden_dim, output_dim_dino)

    def forward(self, voxel_features, coords_3d):
        """
        Args:
            voxel_features (Tensor): [N, voxel_feature_dim].
            coords_3d (Tensor): [N, 3].
        Returns:
            clip_out: (N, output_dim_clip=768)
            dino_out: (N, output_dim_dino=1024)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # [N, pe_dim]

        # 1) fc1
        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)

        # 3) fc3
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)

        # CLIP branch
        x_clip = F.relu(self.ln4_clip(self.fc4_clip(x)), inplace=True)
        clip_out = self.fc5_clip(x_clip)

        # DINO branch
        x_dino = F.relu(self.ln4_dino(self.fc4_dino(x)), inplace=True)
        dino_out = self.fc5_dino(x_dino)

        return clip_out, dino_out
    
class ConcatMLPFusionDINO(nn.Module):
    """
    (feat_clip, feat_voxel, feat_dino, coords_3d) -> [concat + sinusoidal PE] -> MLP -> fused
    """
    def __init__(self, feat_dim=120, L=10, num_feats=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.L = L
        self.pos_enc_dim = 2 * self.L * 3

        in_dim = feat_dim * num_feats + self.pos_enc_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, feat_voxel, feat_clip, feat_dino, coords_3d):
        """
        Args:
            feat_clip:  (B*N, feat_dim)
            feat_voxel: (B*N, feat_dim)
            feat_dino:  (B*N, feat_dim)
            coords_3d:  (B*N, 3)
        Returns:
            fused: (B*N, feat_dim)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # (B*N, 2*L*3)
        x = torch.cat([feat_voxel, feat_clip, feat_dino, pe], dim=-1)
        fused = self.mlp(x)
        return fused