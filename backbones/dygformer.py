"""
DyGFormer — Yu et al. NeurIPS 2023.

Faithful implementation:
  - Patching: groups consecutive historical interactions into patches
  - Neighbor co-occurrence encoding: captures overlap between src/dst neighborhoods
  - 4-channel input: node features, edge features, time encoding, co-occurrence
  - Joint src+dst Transformer processing
  - Mean-pool over patches for final embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict
from .base import TemporalGNNBase, TimeEncoder


class NeighborCoOccurrenceEncoder(nn.Module):
    """
    Encodes how often each neighbor of src also appears in dst's history (and vice versa).
    Input: [count_in_src, count_in_dst] per position -> encoded via MLP -> sum.
    """

    def __init__(self, feat_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.feat_dim = feat_dim

    def forward(self, co_occurrence_counts: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            co_occurrence_counts: [B, seq_len, 2] (count_in_self, count_in_other)
        Returns:
            encoding: [B, seq_len, feat_dim]
        """
        # Encode each count dimension independently then sum
        # [B, seq_len, 2, 1] -> MLP -> [B, seq_len, 2, feat_dim] -> sum over dim 2
        x = co_occurrence_counts.unsqueeze(-1)  # [B, seq_len, 2, 1]
        encoded = self.encoder(x)  # [B, seq_len, 2, feat_dim]
        return encoded.sum(dim=2)  # [B, seq_len, feat_dim]


class DyGFormerBlock(nn.Module):
    """Standard Transformer encoder block with GELU."""

    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-norm Transformer
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DyGFormer(TemporalGNNBase):
    """
    DyGFormer: Transformer for dynamic graphs with patching and co-occurrence.
    
    Processes src and dst interaction sequences jointly through Transformer.
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int = 1,
                 hidden_dim: int = 64, time_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, n_neighbors: int = 20, num_heads: int = 4,
                 patch_size: int = 4, max_seq_len: int = 64,
                 channel_emb_dim: int = 50, co_occurrence_dim: int = 16):
        super().__init__(num_nodes, node_feat_dim, edge_feat_dim, hidden_dim,
                         time_dim, num_layers, dropout, n_neighbors)

        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.channel_emb_dim = channel_emb_dim
        self.num_patches = max_seq_len // patch_size

        edge_dim = hidden_dim if edge_feat_dim > 0 else 0

        # Co-occurrence encoder
        self.co_occurrence_encoder = NeighborCoOccurrenceEncoder(co_occurrence_dim)

        # Per-channel patch projections
        self.proj_node = nn.Linear(patch_size * hidden_dim, channel_emb_dim)
        self.proj_edge = nn.Linear(patch_size * edge_dim, channel_emb_dim) if edge_dim > 0 else None
        self.proj_time = nn.Linear(patch_size * time_dim, channel_emb_dim)
        self.proj_co_occ = nn.Linear(patch_size * co_occurrence_dim, channel_emb_dim)

        # Count active channels
        n_channels = 3 if edge_dim == 0 else 4
        d_model = n_channels * channel_emb_dim

        # Transformer encoder (processes joint src+dst patches)
        self.transformer = nn.ModuleList([
            DyGFormerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, hidden_dim)

    def _get_interaction_sequence(self, node_ids, timestamps, node_feat, edge_feat_all):
        """Get padded sequence of recent interactions for each node."""
        B = len(node_ids)
        device = node_ids.device
        edge_dim = self.hidden_dim if self.edge_feat_dim > 0 else 0

        # Initialize sequences
        node_seqs = torch.zeros(B, self.max_seq_len, self.hidden_dim, device=device)
        edge_seqs = torch.zeros(B, self.max_seq_len, edge_dim, device=device)
        time_seqs = torch.zeros(B, self.max_seq_len, self.time_dim, device=device)
        seq_mask = torch.zeros(B, self.max_seq_len, dtype=torch.bool, device=device)
        ngh_id_lists = [[] for _ in range(B)]

        if self.neighbor_finder is not None:
            ngh_ids, ngh_eidx, ngh_times = self.neighbor_finder.get_temporal_neighbors(
                node_ids, timestamps, self.max_seq_len
            )
            ngh_ids = ngh_ids.to(device)
            ngh_times = ngh_times.to(device)

            valid = ngh_ids > 0
            # Node features of neighbors
            flat_ngh = ngh_ids.reshape(-1)
            ngh_feats = self.get_node_features(flat_ngh, node_feat).reshape(B, self.max_seq_len, -1)
            node_seqs = ngh_feats * valid.unsqueeze(-1).float()

            # Time encodings
            dt = (timestamps.unsqueeze(1) - ngh_times).clamp(min=0)
            time_seqs = self.time_encoder(dt) * valid.unsqueeze(-1).float()

            # Edge features
            if edge_feat_all is not None and self.edge_proj is not None:
                ngh_eidx_dev = ngh_eidx.to(device)
                safe_idx = ngh_eidx_dev.clamp(max=max(edge_feat_all.shape[0] - 1, 0))
                edge_seqs = self.edge_proj(edge_feat_all[safe_idx.reshape(-1)]).reshape(B, self.max_seq_len, -1)
                edge_seqs = edge_seqs * valid.unsqueeze(-1).float()

            seq_mask = valid

            for i in range(B):
                ngh_id_lists[i] = ngh_ids[i][valid[i]].cpu().tolist()

        return node_seqs, edge_seqs, time_seqs, seq_mask, ngh_id_lists

    def _compute_co_occurrence(self, src_ngh_ids, dst_ngh_ids, seq_len, device):
        """Compute co-occurrence counts between src and dst neighbor sets."""
        B = len(src_ngh_ids)
        src_counts = torch.zeros(B, seq_len, 2, device=device)
        dst_counts = torch.zeros(B, seq_len, 2, device=device)

        for i in range(B):
            src_set = {}
            for nid in src_ngh_ids[i]:
                src_set[nid] = src_set.get(nid, 0) + 1
            dst_set = {}
            for nid in dst_ngh_ids[i]:
                dst_set[nid] = dst_set.get(nid, 0) + 1

            # For source sequence positions
            for j, nid in enumerate(src_ngh_ids[i]):
                if j >= seq_len:
                    break
                offset = seq_len - len(src_ngh_ids[i]) + j
                if offset >= 0:
                    src_counts[i, offset, 0] = src_set.get(nid, 0)
                    src_counts[i, offset, 1] = dst_set.get(nid, 0)

            for j, nid in enumerate(dst_ngh_ids[i]):
                if j >= seq_len:
                    break
                offset = seq_len - len(dst_ngh_ids[i]) + j
                if offset >= 0:
                    dst_counts[i, offset, 0] = dst_set.get(nid, 0)
                    dst_counts[i, offset, 1] = src_set.get(nid, 0)

        return src_counts, dst_counts

    def _patch_and_project(self, node_seq, edge_seq, time_seq, co_occ_enc):
        """Reshape sequences into patches and project each channel."""
        B = node_seq.shape[0]
        P = self.num_patches
        ps = self.patch_size

        # Reshape: [B, seq_len, d] -> [B, num_patches, patch_size * d]
        node_patches = node_seq.reshape(B, P, ps * self.hidden_dim)
        time_patches = time_seq.reshape(B, P, ps * self.time_dim)
        co_occ_patches = co_occ_enc.reshape(B, P, ps * self.co_occurrence_encoder.feat_dim)

        channels = [
            self.proj_node(node_patches),
            self.proj_time(time_patches),
            self.proj_co_occ(co_occ_patches),
        ]

        if self.proj_edge is not None:
            edge_dim = edge_seq.shape[-1]
            edge_patches = edge_seq.reshape(B, P, ps * edge_dim)
            channels.append(self.proj_edge(edge_patches))

        # Stack channels: [B, P, n_channels * channel_emb_dim]
        return torch.cat(channels, dim=-1)

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        device = src.device

        # Get interaction sequences for src and dst
        src_node_seq, src_edge_seq, src_time_seq, src_mask, src_ngh_ids = \
            self._get_interaction_sequence(src, t, node_feat, edge_feat)
        dst_node_seq, dst_edge_seq, dst_time_seq, dst_mask, dst_ngh_ids = \
            self._get_interaction_sequence(dst, t, node_feat, edge_feat)

        # Compute co-occurrence
        src_co_occ, dst_co_occ = self._compute_co_occurrence(
            src_ngh_ids, dst_ngh_ids, self.max_seq_len, device
        )
        src_co_occ_enc = self.co_occurrence_encoder(src_co_occ)
        dst_co_occ_enc = self.co_occurrence_encoder(dst_co_occ)

        # Patch and project
        src_patches = self._patch_and_project(src_node_seq, src_edge_seq, src_time_seq, src_co_occ_enc)
        dst_patches = self._patch_and_project(dst_node_seq, dst_edge_seq, dst_time_seq, dst_co_occ_enc)

        # Joint src+dst processing
        joint = torch.cat([src_patches, dst_patches], dim=1)  # [B, 2*P, d_model]

        # Padding mask for transformer
        src_patch_mask = ~src_mask.reshape(-1, self.num_patches, self.patch_size).any(dim=-1)
        dst_patch_mask = ~dst_mask.reshape(-1, self.num_patches, self.patch_size).any(dim=-1)
        joint_mask = torch.cat([src_patch_mask, dst_patch_mask], dim=1)

        for block in self.transformer:
            joint = block(joint, key_padding_mask=joint_mask)

        # Split and mean-pool
        src_out = joint[:, :self.num_patches]
        dst_out = joint[:, self.num_patches:]

        # Mean pool over valid patches
        src_valid = (~src_patch_mask).unsqueeze(-1).float()
        dst_valid = (~dst_patch_mask).unsqueeze(-1).float()

        src_emb = (src_out * src_valid).sum(dim=1) / src_valid.sum(dim=1).clamp(min=1)
        dst_emb = (dst_out * dst_valid).sum(dim=1) / dst_valid.sum(dim=1).clamp(min=1)

        src_emb = self.output_proj(src_emb)
        dst_emb = self.output_proj(dst_emb)

        node_states = (src_emb + dst_emb) / 2
        return {"node_states": node_states, "src_states": src_emb, "dst_states": dst_emb}
