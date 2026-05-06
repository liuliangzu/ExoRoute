"""
TGN (Temporal Graph Network) — Rossi et al. 2020.

Faithful implementation:
  - Per-node memory with GRU updater
  - Raw message = [src_mem || dst_mem || edge_feat || time_enc(delta_t)]
  - Message function: identity (default) or MLP
  - Message aggregation: last (most recent) per node
  - Embedding module: multi-head temporal graph attention (recursive L layers)
  - MergeLayer skip connection in attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from .base import TemporalGNNBase, TimeEncoder, MergeLayer


class TGNMemory(nn.Module):
    """
    Per-node memory updated via GRU.
    
    Pipeline: raw_message -> message_fn -> aggregation -> GRU update.
    """

    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.gru = nn.GRUCell(message_dim, memory_dim)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

    def reset(self):
        self.memory.zero_()
        self.last_update.zero_()

    def get(self, node_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.memory[node_ids].detach()

    def update_memory(self, node_ids: torch.LongTensor, messages: torch.FloatTensor,
                      timestamps: torch.FloatTensor):
        """
        Last-message aggregation + GRU update.
        For nodes with multiple messages, keep only the most recent.
        """
        unique_ids, inverse = node_ids.unique(return_inverse=True)
        for idx, nid in enumerate(unique_ids):
            mask = inverse == idx
            # Last aggregation: take message with largest timestamp
            node_msgs = messages[mask]
            node_times = timestamps[mask]
            last_idx = node_times.argmax()
            msg = node_msgs[last_idx].unsqueeze(0)

            mem = self.memory[nid].unsqueeze(0)
            new_mem = self.gru(msg, mem)
            self.memory[nid] = new_mem.squeeze(0)
            self.last_update[nid] = node_times[last_idx]


class TemporalGraphAttention(nn.Module):
    """
    Multi-head temporal attention for TGN's embedding module.
    
    Query = [src_feat || time_enc(0)]
    Key = Value = [ngh_feat || edge_feat || time_enc(dt)]
    Output merged with source via MergeLayer (skip connection).
    """

    def __init__(self, feat_dim: int, edge_dim: int, time_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = feat_dim + time_dim
        self.key_dim = feat_dim + edge_dim + time_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=num_heads,
            kdim=self.key_dim,
            vdim=self.key_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Project key/value to query dimension if they differ
        if self.key_dim != self.query_dim:
            self.kv_proj = nn.Linear(self.key_dim, self.query_dim)
        else:
            self.kv_proj = None

        # MergeLayer: combines attention output with source features
        self.merge = MergeLayer(self.query_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src_feat, src_time_enc, ngh_feat, ngh_edge_feat,
                ngh_time_enc, ngh_mask):
        """
        Args:
            src_feat: [B, feat_dim]
            src_time_enc: [B, time_dim] (time_enc(0))
            ngh_feat: [B, N, feat_dim]
            ngh_edge_feat: [B, N, edge_dim]
            ngh_time_enc: [B, N, time_dim]
            ngh_mask: [B, N] bool, True = valid neighbor
        Returns:
            output: [B, feat_dim]
        """
        B, N, _ = ngh_feat.shape

        # Query: [src_feat || time_enc(0)]
        q = torch.cat([src_feat, src_time_enc], dim=-1).unsqueeze(1)  # [B, 1, q_dim]

        # Key/Value: [ngh_feat || edge_feat || time_enc(dt)]
        kv = torch.cat([ngh_feat, ngh_edge_feat, ngh_time_enc], dim=-1)  # [B, N, k_dim]

        if self.kv_proj is not None:
            kv = self.kv_proj(kv)

        # Attention mask: True means IGNORE in PyTorch
        key_padding_mask = ~ngh_mask  # [B, N]

        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1)  # [B, q_dim]

        # Merge with source (skip connection)
        output = self.merge(attn_out, src_feat)
        return output


class TGN(TemporalGNNBase):
    """
    Temporal Graph Network.
    
    Default config: GRU memory, identity message fn, last aggregation,
    graph_attention embedding, 2 attention heads.
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int = 1,
                 hidden_dim: int = 64, time_dim: int = 64, num_layers: int = 1,
                 dropout: float = 0.1, n_neighbors: int = 10, num_heads: int = 2):
        super().__init__(num_nodes, node_feat_dim, edge_feat_dim, hidden_dim,
                         time_dim, num_layers, dropout, n_neighbors)

        self.memory_dim = hidden_dim
        # Raw message dim: src_mem + dst_mem + edge_feat + time_enc
        edge_dim = hidden_dim if edge_feat_dim > 0 else 0
        raw_msg_dim = hidden_dim * 2 + edge_dim + time_dim

        # Memory module
        self.tgn_memory = TGNMemory(num_nodes, hidden_dim, raw_msg_dim)

        # Embedding: temporal graph attention (L layers, each with own attn)
        self.attn_layers = nn.ModuleList([
            TemporalGraphAttention(hidden_dim, edge_dim, time_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def _compute_raw_messages(self, src, dst, t, edge_feat=None):
        """
        Raw message = [src_memory || dst_memory || edge_features || time_enc(dt)].
        """
        src_mem = self.tgn_memory.get(src)
        dst_mem = self.tgn_memory.get(dst)

        # Time delta since last update
        dt_src = t - self.tgn_memory.last_update[src]
        time_enc = self.time_encoder(dt_src)

        parts = [src_mem, dst_mem, time_enc]
        if edge_feat is not None and self.edge_proj is not None:
            parts.insert(2, self.edge_proj(edge_feat))

        return torch.cat(parts, dim=-1)

    def _get_embedding(self, node_ids, timestamps, node_feat, edge_feat_all, layer):
        """
        Recursive temporal graph attention embedding.
        Layer 0: return memory + node features.
        Layer L: attend over L-1 neighbor embeddings.
        """
        if layer == 0:
            mem = self.tgn_memory.get(node_ids)
            feat = self.get_node_features(node_ids, node_feat)
            return mem + feat

        # Get temporal neighbors
        if self.neighbor_finder is not None:
            ngh_ids, ngh_eidx, ngh_times = self.neighbor_finder.get_temporal_neighbors(
                node_ids, timestamps, self.n_neighbors
            )
            ngh_ids = ngh_ids.to(node_ids.device)
            ngh_times = ngh_times.to(node_ids.device)
        else:
            # Fallback: no neighbor info
            B = len(node_ids)
            return self.tgn_memory.get(node_ids) + self.get_node_features(node_ids, node_feat)

        B, N = ngh_ids.shape
        ngh_mask = ngh_ids > 0  # padding = 0

        # Recurse: get neighbor embeddings at layer-1
        flat_ngh = ngh_ids.reshape(-1)
        flat_times = ngh_times.reshape(-1)
        ngh_embs = self._get_embedding(flat_ngh, flat_times, node_feat, edge_feat_all, layer - 1)
        ngh_embs = ngh_embs.reshape(B, N, -1)

        # Edge features for these neighbors
        if edge_feat_all is not None and self.edge_proj is not None:
            ngh_eidx_dev = ngh_eidx.to(node_ids.device)
            valid_mask = ngh_eidx_dev < edge_feat_all.shape[0]
            safe_idx = ngh_eidx_dev.clamp(max=edge_feat_all.shape[0] - 1)
            ngh_edge = self.edge_proj(edge_feat_all[safe_idx.reshape(-1)]).reshape(B, N, -1)
            ngh_edge = ngh_edge * valid_mask.reshape(B, N, 1).float()
        else:
            ngh_edge = torch.zeros(B, N, 0, device=node_ids.device)

        # Time encoding of delta
        dt = timestamps.unsqueeze(1) - ngh_times.to(timestamps.device)
        dt = dt.clamp(min=0)
        ngh_time_enc = self.time_encoder(dt)

        # Source features
        src_feat = self.tgn_memory.get(node_ids) + self.get_node_features(node_ids, node_feat)
        src_time_enc = self.time_encoder(torch.zeros(B, device=node_ids.device))

        # Temporal graph attention at this layer
        output = self.attn_layers[layer - 1](
            src_feat, src_time_enc, ngh_embs, ngh_edge, ngh_time_enc,
            ngh_mask.to(node_ids.device)
        )
        return output

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        device = src.device

        # Compute embeddings via recursive temporal attention
        src_emb = self._get_embedding(src, t, node_feat, edge_feat, self.num_layers)
        dst_emb = self._get_embedding(dst, t, node_feat, edge_feat, self.num_layers)

        # Compute and store messages (identity message function)
        raw_msgs = self._compute_raw_messages(src, dst, t, edge_feat)
        # Update memory for both src and dst
        self.tgn_memory.update_memory(src, raw_msgs, t)
        self.tgn_memory.update_memory(dst, raw_msgs, t)

        node_states = (src_emb + dst_emb) / 2
        return {"node_states": node_states, "src_states": src_emb, "dst_states": dst_emb}
