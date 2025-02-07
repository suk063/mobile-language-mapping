import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import rotary_pe_3d  # Ensure this is correctly imported from your utils

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
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm3(src + self.dropout3(src2))
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
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, hand, head, coords_hand, coords_head, state, text_embeddings):
        """
        hand, head: [B, N, input_dim]
        coords_hand, coords_head: [B, N, 3]
        state: [B, 42]
        text_embeddings: [B, T, input_dim]
        """
        B, N, D = hand.shape

        # Project state into a single token
        state_token = self.state_projection(state).unsqueeze(1)
        coords_state = torch.zeros(B, 1, 3, device=state.device)

        # For simplicity, assume text_embeddings is [B, 1, input_dim]
        # If not, adjust accordingly
        text_embeddings = text_embeddings.unsqueeze(1)

        # Concatenate state, hand, head
        src = torch.cat([state_token, hand, head], dim=1)
        coords_src = torch.cat([coords_state, coords_hand, coords_head], dim=1)

        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(src=src, text=text_embeddings, coords_src=coords_src)

        # Post-fusion MLP
        data = src[:, 1:, :].reshape(B, -1)
        out = self.post_fusion_mlp(data)
        return out

class LocalSelfAttentionFusion(nn.Module):
    """
    Merges two features (voxel_feat, cnn_feat) per spatial location
    via a self-attention layer. Each location is treated independently
    as a 2-token sequence.
    """
    def __init__(self, feat_dim=120, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(feat_dim)

    def forward(self, voxel_feat, cnn_feat):
        """
        voxel_feat, cnn_feat: (B, N, feat_dim)
        Returns fused_feat: (B, N, feat_dim)
        """
        B, N, D = voxel_feat.shape

        # Combine voxel_feat and cnn_feat into a 2-token sequence per spatial location
        x = torch.stack([voxel_feat, cnn_feat], dim=2)  # (B, N, 2, D)
        x = x.view(B * N, 2, D)                        # (B*N, 2, D)

        # Self-attention on the 2-token sequence
        y, _ = self.mha(x, x, x)
        y = self.layernorm(y)

        # Average the 2 tokens to get the final fused representation
        fused = y.mean(dim=1).view(B, N, D)
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
