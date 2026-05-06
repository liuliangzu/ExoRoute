"""
TGAT (Temporal Graph Attention) — Xu et al. ICLR 2020.

Faithful implementation:
  - Functional time encoding: cos(t * omega + phi), learnable omega & phi
  - Q/K/V = concat(node_feat, edge_feat, time_enc)
  - Multi-head scaled dot-product attention
  - Recursive L-layer aggregation over temporal neighborhoods
  - MergeLayer skip connection per layer
  - Temporal neighbor sampling (most recent N before cut_time)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from .base import TemporalGNNBase, TimeEncoder, MergeLayer


class TGATAttentionLayer(nn.Module):
    """
    TGAT temporal attention layer.
    
    d_model = feat_dim + edge_dim + time_dim
    Q = [src_feat || zero_edge || time_enc(0)]  -- query node, no self-edge
    K = V = [ngh_feat || edge_feat || time_enc(dt)]  -- neighbor info
    """

    def __init__(self, feat_dim: int, edge_dim: int, time_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = feat_dim + edge_dim + time_dim
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        # MergeLayer: combines attn output with original src feature
        self.merge = MergeLayer(self.d_model, feat_dim, feat_dim, feat_dim)

    def forward(self, src_feat, ngh_feat, ngh_edge_feat, src_time_enc,
                ngh_time_enc, ngh_mask):
        """
        Args:
            src_feat: [B, feat_dim]
            ngh_feat: [B, N, feat_dim]
            ngh_edge_feat: [B, N, edge_dim]
            src_time_enc: [B, time_dim]
            ngh_time_enc: [B, N, time_dim]
            ngh_mask: [B, N] bool, True = valid
        Returns:
            output: [B, feat_dim]
        """
        B, N, _ = ngh_feat.shape
        edge_dim = ngh_edge_feat.shape[-1]

        # Query: [src_feat || zeros_for_edge || time_enc(0)]
        zero_edge = torch.zeros(B, edge_dim, device=src_feat.device)
        q = torch.cat([src_feat, zero_edge, src_time_enc], dim=-1)  # [B, d_model]
        q = q.unsqueeze(1)  # [B, 1, d_model]

        # Key/Value: [ngh_feat || edge_feat || time_enc(dt)]
        kv = torch.cat([ngh_feat, ngh_edge_feat, ngh_time_enc], dim=-1)  # [B, N, d_model]

        # Multi-head attention
        Q = self.W_q(q).reshape(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [B, heads, 1, N]

        # Mask invalid neighbors
        mask_exp = (~ngh_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        scores = scores.masked_fill(mask_exp, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, heads, 1, d_k]
        out = out.transpose(1, 2).reshape(B, 1, self.d_model)

        # Residual + layer norm
        out = self.layer_norm(self.dropout(self.fc_out(out)) + q)
        out = out.squeeze(1)  # [B, d_model]

        # Merge with original source feature (skip connection)
        output = self.merge(out, src_feat)
        return output


class TGAT(TemporalGNNBase):
    """
    Temporal Graph Attention Network.
    
    L-layer recursive temporal attention over sampled neighbors.
    Default: 3 layers, 4 heads, 20 neighbors.
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int = 1,
                 hidden_dim: int = 64, time_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, n_neighbors: int = 20, num_heads: int = 4):
        super().__init__(num_nodes, node_feat_dim, edge_feat_dim, hidden_dim,
                         time_dim, num_layers, dropout, n_neighbors)

        edge_dim = hidden_dim if edge_feat_dim > 0 else 0

        # Per-layer attention modules
        self.attn_layers = nn.ModuleList([
            TGATAttentionLayer(hidden_dim, edge_dim, time_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def _temporal_conv(self, node_ids, timestamps, node_feat, edge_feat_all, layer):
        """
        Recursive temporal convolution.
        Layer 0: return raw node features.
        Layer L: attend over layer-(L-1) neighbor embeddings.
        """
        if layer == 0:
            return self.get_node_features(node_ids, node_feat)

        # Get temporal neighbors
        if self.neighbor_finder is not None:
            ngh_ids, ngh_eidx, ngh_times = self.neighbor_finder.get_temporal_neighbors(
                node_ids, timestamps, self.n_neighbors
            )
            ngh_ids = ngh_ids.to(node_ids.device)
            ngh_times = ngh_times.to(node_ids.device)
        else:
            return self.get_node_features(node_ids, node_feat)

        B, N = ngh_ids.shape
        ngh_mask = ngh_ids > 0

        # Recurse: neighbor embeddings at layer-1
        flat_ngh = ngh_ids.reshape(-1)
        flat_times = ngh_times.reshape(-1)
        ngh_embs = self._temporal_conv(flat_ngh, flat_times, node_feat, edge_feat_all, layer - 1)
        ngh_embs = ngh_embs.reshape(B, N, -1)

        # Edge features
        if edge_feat_all is not None and self.edge_proj is not None:
            ngh_eidx_dev = ngh_eidx.to(node_ids.device)
            safe_idx = ngh_eidx_dev.clamp(max=max(edge_feat_all.shape[0] - 1, 0))
            ngh_edge = self.edge_proj(edge_feat_all[safe_idx.reshape(-1)]).reshape(B, N, -1)
            valid = (ngh_eidx_dev < edge_feat_all.shape[0]).reshape(B, N, 1).float()
            ngh_edge = ngh_edge * valid
        else:
            edge_dim = self.hidden_dim if self.edge_feat_dim > 0 else 0
            ngh_edge = torch.zeros(B, N, edge_dim, device=node_ids.device)

        # Time encodings
        dt = (timestamps.unsqueeze(1) - ngh_times.to(timestamps.device)).clamp(min=0)
        ngh_time_enc = self.time_encoder(dt)
        src_time_enc = self.time_encoder(torch.zeros(B, device=node_ids.device))

        # Self features at layer-1
        src_feat = self._temporal_conv(node_ids, timestamps, node_feat, edge_feat_all, layer - 1)

        # Attention
        output = self.attn_layers[layer - 1](
            src_feat, ngh_embs, ngh_edge, src_time_enc, ngh_time_enc,
            ngh_mask.to(node_ids.device)
        )
        return output

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        src_emb = self._temporal_conv(src, t, node_feat, edge_feat, self.num_layers)
        dst_emb = self._temporal_conv(dst, t, node_feat, edge_feat, self.num_layers)

        node_states = (src_emb + dst_emb) / 2
        return {"node_states": node_states, "src_states": src_emb, "dst_states": dst_emb}
