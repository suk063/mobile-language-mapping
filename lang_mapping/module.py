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

class TransformerCrossAttentionLayer(nn.Module):
    """
    Consists of:
    1) Self-Attention (with optional rotary PE)
    2) Cross-Attention (Q=src, K=V=text)
    3) FeedForward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, text, coords_src=None):
        # Self-Attention (rotary PE applied if coords_src is given)
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_ = src
        else:
            q_rot = k_rot = v_ = src

        src2, _ = self.self_attn(query=q_rot, key=k_rot, value=v_)
        src = self.norm1(src + self.dropout1(src2))

        # Cross-Attention (Q=src, K=V=text)
        src2, _ = self.cross_attn(query=src, key=text, value=text)
        src = self.norm2(src + self.dropout2(src2))

        # FeedForward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm3(src + self.dropout3(src2))
        return src

class TransformerLayer(nn.Module):
    """
    A single Transformer layer with self-attention (optionally uses rotary PE).
    There is NO cross-attention here. We assume text embeddings are appended 
    as tokens within `src`.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, coords_src=None):
        """
        Args:
            src: [B, T, d_model] tokens for (state, text, hand, head, etc.)
            coords_src: [B, T, 3] if we want to apply rotary PE; else None.
        """
        # Optionally apply rotary position encoding
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_ = src
        else:
            q_rot = k_rot = v_ = src

        # Self-attention
        src2, _ = self.self_attn(query=q_rot, key=k_rot, value=v_)
        src = self.norm1(src + self.dropout1(src2))

        # Feed-forward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))

        return src

class TransformerEncoder(nn.Module):
    """
    Stacks multiple TransformerCrossAttentionLayers to fuse
    state, hand, head, and text embeddings. Produces a final output.
    """
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=2, num_heads=8, output_dim=1024):
        super().__init__()
        self.state_projection = nn.Linear(42, input_dim)
        self.layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
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

    def forward(self, hand, head, coords_hand=None, coords_head=None, state=None, text_embeddings=None):
        """
        hand, head: [B, N, input_dim]
        coords_hand, coords_head: [B, N, 3]
        state: [B, 42]
        text_embeddings: [B, input_dim]
        """
        B, N, D = hand.shape

        # Project state into a single token
        state_token = self.state_projection(state).unsqueeze(1)
        coords_state = torch.zeros(B, 1, 3, device=state.device)

        text_embeddings = text_embeddings.unsqueeze(1)
        coords_text = torch.zeros(B, 1, 3, device=state.device) 

        # Concatenate state, hand, head
        src = torch.cat([state_token, text_embeddings, hand, head], dim=1)
        if coords_hand is not None:
            coords_src = torch.cat([coords_state, coords_text, coords_hand, coords_head], dim=1)
        else:
            coords_src = None

        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(src=src, text=text_embeddings, coords_src=coords_src)

        # Post-fusion MLP
        data = src[:, 2:, :].reshape(B, -1)
        
        out = self.post_fusion_mlp(data)
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

        x1 = x

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x1)), inplace=True)

        # 3) fc3 
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)
        
        # 4) fc4
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)

        # 5) fc5 
        out = self.fc5(x)

        if return_intermediate:
            return x1, out
        else:
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
            nn.ReLU(),
        )

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