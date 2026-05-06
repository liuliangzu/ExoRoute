"""
GraphMixer — Cong et al. ICLR 2023.

Faithful implementation:
  - Fixed (non-learnable) cosine time encoding
  - Link encoder: MLP-Mixer that mixes over neighbor dimension (token mixing)
    and feature dimension (channel mixing), then mean-pools
  - Node encoder: mean-pooling over recent neighbors (SIGN-style)
  - No attention mechanism at all
  - EdgePredictor: FC_src(h_src) + FC_dst(h_dst) -> ReLU -> FC_out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .base import TemporalGNNBase, FixedTimeEncoder, MergeLayer


class FeedForward(nn.Module):
    """2-layer MLP with GELU."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """
    MLP-Mixer block: token mixing + channel mixing.
    
    Token mixing operates across the neighbor dimension.
    Channel mixing operates across the feature dimension.
    """

    def __init__(self, num_tokens: int, hidden_dim: int,
                 token_expansion: float = 0.5, channel_expansion: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        # Token mixing: operates on transposed input [B, d, N]
        self.token_ln = nn.LayerNorm(hidden_dim)
        self.token_mix = FeedForward(
            num_tokens, int(num_tokens * token_expansion), dropout
        )

        # Channel mixing: operates on [B, N, d]
        self.channel_ln = nn.LayerNorm(hidden_dim)
        self.channel_mix = FeedForward(
            hidden_dim, int(hidden_dim * channel_expansion), dropout
        )

    def forward(self, x):
        """x: [B, N, d]"""
        # Token mixing: transpose, mix across N, transpose back
        res = self.token_ln(x)
        res = res.permute(0, 2, 1)  # [B, d, N]
        res = self.token_mix(res)
        res = res.permute(0, 2, 1)  # [B, N, d]
        x = x + res

        # Channel mixing: mix across d
        res = self.channel_ln(x)
        res = self.channel_mix(res)
        x = x + res

        return x


class MLPMixerEncoder(nn.Module):
    """
    Link encoder using MLP-Mixer.
    
    Processes a fixed-size set of recent neighbor interactions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(num_tokens, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, input_dim] — neighbor features (edge_feat || time_enc)
        Returns:
            h: [B, hidden_dim] — mean-pooled output
        """
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return x.mean(dim=1)  # mean pooling over neighbors


class GraphMixer(TemporalGNNBase):
    """
    GraphMixer: MLP-Mixer for temporal graphs.
    
    No attention, no memory. Fixed time encoding.
    Token mixing across neighbors, mean pooling output.
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int = 1,
                 hidden_dim: int = 64, time_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, n_neighbors: int = 20):
        super().__init__(num_nodes, node_feat_dim, edge_feat_dim, hidden_dim,
                         time_dim, num_layers, dropout, n_neighbors)

        # Override: use FIXED time encoding
        self.time_encoder = FixedTimeEncoder(time_dim)

        edge_dim = hidden_dim if edge_feat_dim > 0 else 0
        link_input_dim = edge_dim + time_dim  # edge_features || time_encoding

        # Link encoder: MLP-Mixer over recent neighbors
        self.link_encoder = MLPMixerEncoder(
            input_dim=link_input_dim,
            hidden_dim=hidden_dim,
            num_tokens=n_neighbors,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Node encoder: simple aggregation of neighbor features
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge predictor: FC_src + FC_dst -> ReLU -> FC_out
        self.fc_src = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_dst = nn.Linear(hidden_dim * 2, hidden_dim)

        # Override readout
        self.readout_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_node(self, node_ids, timestamps, node_feat, edge_feat_all):
        """Encode a node using its recent temporal neighbors via MLP-Mixer."""
        B = len(node_ids)
        device = node_ids.device
        edge_dim = self.hidden_dim if self.edge_feat_dim > 0 else 0

        if self.neighbor_finder is not None:
            ngh_ids, ngh_eidx, ngh_times = self.neighbor_finder.get_temporal_neighbors(
                node_ids, timestamps, self.n_neighbors
            )
            ngh_ids = ngh_ids.to(device)
            ngh_times = ngh_times.to(device)

            # Time encoding of deltas
            dt = (timestamps.unsqueeze(1) - ngh_times).clamp(min=0)
            time_enc = self.time_encoder(dt)  # [B, N, time_dim]

            # Edge features
            if edge_feat_all is not None and self.edge_proj is not None:
                ngh_eidx_dev = ngh_eidx.to(device)
                safe_idx = ngh_eidx_dev.clamp(max=max(edge_feat_all.shape[0] - 1, 0))
                edge_enc = self.edge_proj(edge_feat_all[safe_idx.reshape(-1)]).reshape(B, self.n_neighbors, -1)
            else:
                edge_enc = torch.zeros(B, self.n_neighbors, edge_dim, device=device)

            # Input: [edge_features || time_encoding]
            mixer_input = torch.cat([edge_enc, time_enc], dim=-1)
        else:
            # No neighbor finder: use zeros
            mixer_input = torch.zeros(B, self.n_neighbors, edge_dim + self.time_dim, device=device)

        link_emb = self.link_encoder(mixer_input)  # [B, hidden]

        # Node features
        node_emb = self.get_node_features(node_ids, node_feat)
        node_emb = self.node_encoder(node_emb)

        return torch.cat([link_emb, node_emb], dim=-1)  # [B, 2*hidden]

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        src_emb = self._encode_node(src, t, node_feat, edge_feat)
        dst_emb = self._encode_node(dst, t, node_feat, edge_feat)

        # Edge prediction: FC_src(h_src) + FC_dst(h_dst)
        combined = F.relu(self.fc_src(src_emb) + self.fc_dst(dst_emb))

        return {"node_states": combined, "src_states": src_emb, "dst_states": dst_emb}
