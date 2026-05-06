"""
Common base class and utilities for temporal GNN backbones.

Includes:
  - Learnable time encoding (TGN/TGAT style: cos with log-spaced learnable freqs)
  - Fixed time encoding (GraphMixer style: non-trainable cos)
  - Temporal neighbor finder (binary-search based causal neighbor sampling)
  - MergeLayer for combining node pair embeddings
  - TemporalGNNBase with readout head
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
from bisect import bisect_left


class TimeEncoder(nn.Module):
    """
    Learnable time encoding with log-spaced initial frequencies.
    Used by TGN, TGAT, DyGFormer, CTAN, GeneralDyG.
    
    Phi(t) = cos(w * t), where w has learnable weights initialized
    to 1/10^linspace(0, 9, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        # Initialize with log-spaced frequencies
        with torch.no_grad():
            freqs = 1.0 / (10 ** torch.linspace(0, 9, dim))
            self.w.weight.copy_(freqs.unsqueeze(1))
            self.w.bias.zero_()

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        """t: [...] -> [..., dim]"""
        t = t.unsqueeze(-1).float()
        return torch.cos(self.w(t))


class FixedTimeEncoder(nn.Module):
    """
    Fixed (non-learnable) time encoding for GraphMixer.
    
    Same cos basis as TimeEncoder but weights are frozen.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        with torch.no_grad():
            freqs = 1.0 / (10 ** torch.linspace(0, 9, dim))
            self.w.weight.copy_(freqs.unsqueeze(1))
            self.w.bias.zero_()
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        t = t.unsqueeze(-1).float()
        return torch.cos(self.w(t))


class MergeLayer(nn.Module):
    """2-layer MLP for combining node pair embeddings -> scalar score."""

    def __init__(self, dim1: int, dim2: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x1: torch.FloatTensor, x2: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.cat([x1, x2], dim=-1)
        return self.fc2(self.act(self.fc1(x)))


class TemporalNeighborFinder:
    """
    Efficient temporal neighbor lookup using sorted adjacency lists.
    
    For each node, maintains a sorted list of (timestamp, neighbor_id, edge_idx).
    Supports causal queries: only returns neighbors with timestamp < query_time.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        # adj_list[node] = list of (timestamp, neighbor_id, edge_idx), sorted by time
        self.adj_list: List[List[Tuple[float, int, int]]] = [[] for _ in range(num_nodes)]

    def build_from_edges(
        self,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        t: torch.FloatTensor,
        edge_idx: Optional[torch.LongTensor] = None,
    ):
        """Build adjacency lists from edge data (must be time-sorted)."""
        n = len(src)
        if edge_idx is None:
            edge_idx = torch.arange(n)

        for i in range(n):
            s, d, ti, ei = src[i].item(), dst[i].item(), t[i].item(), edge_idx[i].item()
            self.adj_list[s].append((ti, d, ei))
            self.adj_list[d].append((ti, s, ei))

    def get_temporal_neighbors(
        self,
        node_ids: torch.LongTensor,
        timestamps: torch.FloatTensor,
        n_neighbors: int = 10,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Get most recent n_neighbors for each node before its timestamp.
        
        Returns:
            neighbor_ids: [B, n_neighbors] (0-padded)
            edge_indices: [B, n_neighbors] (0-padded)
            neighbor_times: [B, n_neighbors] (0-padded)
        """
        B = len(node_ids)
        ngh_ids = torch.zeros(B, n_neighbors, dtype=torch.long)
        ngh_eidx = torch.zeros(B, n_neighbors, dtype=torch.long)
        ngh_times = torch.zeros(B, n_neighbors, dtype=torch.float)

        for i in range(B):
            nid = node_ids[i].item()
            cut_time = timestamps[i].item()
            adj = self.adj_list[nid]

            # Binary search for events before cut_time
            # adj is sorted by time
            times_list = [a[0] for a in adj]
            pos = bisect_left(times_list, cut_time)

            # Take most recent n_neighbors (from pos-1 backwards)
            valid = adj[:pos]
            if len(valid) > n_neighbors:
                valid = valid[-n_neighbors:]  # most recent

            for j, (t_n, n_id, e_idx) in enumerate(valid):
                offset = n_neighbors - len(valid) + j  # right-align
                ngh_ids[i, offset] = n_id
                ngh_eidx[i, offset] = e_idx
                ngh_times[i, offset] = t_n

        return ngh_ids, ngh_eidx, ngh_times


class TemporalGNNBase(nn.Module):
    """Base class for temporal GNN backbones used with REGATE."""

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 64,
        time_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_neighbors: int = 10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.n_neighbors = n_neighbors

        # Node embedding for featureless nodes
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # Time encoder (default: learnable)
        self.time_encoder = TimeEncoder(time_dim)

        # Feature projection
        if node_feat_dim > 0:
            self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        else:
            self.node_proj = None

        if edge_feat_dim > 0:
            self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)
        else:
            self.edge_proj = None

        # Readout head (2-layer MLP -> scalar)
        self.readout_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Neighbor finder (populated at runtime)
        self.neighbor_finder: Optional[TemporalNeighborFinder] = None

    def set_neighbor_finder(self, nf: TemporalNeighborFinder):
        self.neighbor_finder = nf

    def get_node_dim(self) -> int:
        return self.hidden_dim

    def get_node_features(
        self,
        node_ids: torch.LongTensor,
        node_feat: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Get node feature representations."""
        if node_feat is not None and self.node_proj is not None:
            return self.node_proj(node_feat[node_ids])
        return self.node_embedding(node_ids)

    def readout(self, h: torch.FloatTensor) -> torch.FloatTensor:
        """Prediction readout: [B, d] -> [B]"""
        return self.readout_head(h).squeeze(-1)

    def forward(
        self,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        t: torch.FloatTensor,
        edge_feat: Optional[torch.FloatTensor] = None,
        node_feat: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
