"""
CTAN (Continuous-Time Anti-symmetric Network) — Gravina et al. 2024.

Faithful implementation:
  - Anti-symmetric weight: A = W - W^T - γI (eigenvalues on negative real axis)
  - Forward Euler ODE integration with shared weights across steps
  - Message passing Φ at event times via attention-based neighbor aggregation
  - Tanh activation (default)
  - Stability: anti-symmetric construction prevents gradient explosion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from .base import TemporalGNNBase, TimeEncoder, MergeLayer


class AntiSymmetricLayer(nn.Module):
    """
    Anti-symmetric linear transform: A = W - W^T - γI.
    
    Properties:
      - (W - W^T) is skew-symmetric with purely imaginary eigenvalues
      - -γI shifts eigenvalues to have real part -γ (controlled dissipation)
      - Forward Euler is stable when γ > 0
    """

    def __init__(self, dim: int, gamma: float = 0.1):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.register_buffer("eye", torch.eye(dim))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """x: [..., dim] -> [..., dim]"""
        # A = W - W^T - γI
        A = self.W - self.W.t() - self.gamma * self.eye
        return F.linear(x, A, self.bias)


class TemporalAttentionMessage(nn.Module):
    """
    Attention-based message passing Φ for CTAN.
    
    Aggregates neighbor information with temporal attention:
    Φ(X, N_i) = Σ_j α_ij (V_n h_j + V_e ê_ij)
    where ê_ij includes time encoding of elapsed time.
    """

    def __init__(self, dim: int, edge_dim: int, time_dim: int, num_heads: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim + edge_dim + time_dim, dim)
        self.V_n = nn.Linear(dim, dim)
        self.V_e = nn.Linear(edge_dim + time_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h_node, h_ngh, ngh_edge, ngh_time_enc, ngh_mask):
        """
        Args:
            h_node: [B, dim] current node states
            h_ngh: [B, N, dim] neighbor states
            ngh_edge: [B, N, edge_dim] edge features
            ngh_time_enc: [B, N, time_dim] time encoding of elapsed time
            ngh_mask: [B, N] True = valid
        Returns:
            message: [B, dim]
        """
        B, N, _ = h_ngh.shape

        q = self.W_q(h_node).unsqueeze(1)  # [B, 1, dim]
        k_input = torch.cat([h_ngh, ngh_edge, ngh_time_enc], dim=-1)
        k = self.W_k(k_input)  # [B, N, dim]

        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # [B, 1, N]
        scores = scores.masked_fill(~ngh_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Value: V_n h_j + V_e [edge || time]
        v_node = self.V_n(h_ngh)  # [B, N, dim]
        v_edge = self.V_e(torch.cat([ngh_edge, ngh_time_enc], dim=-1))  # [B, N, dim]
        v = v_node + v_edge

        out = torch.bmm(attn, v).squeeze(1)  # [B, dim]
        return self.out_proj(out)


class CTANODEBlock(nn.Module):
    """
    Forward Euler ODE integration with anti-symmetric dynamics.
    
    x_{k+1} = x_k + ε * σ(A * x_k + Φ(X, N) + b)
    
    Shared weights across all Euler steps.
    """

    def __init__(self, dim: int, edge_dim: int, time_dim: int,
                 num_steps: int = 1, epsilon: float = 0.1,
                 gamma: float = 0.1, num_heads: int = 2):
        super().__init__()
        self.num_steps = num_steps
        self.epsilon = epsilon

        self.anti_sym = AntiSymmetricLayer(dim, gamma)
        self.message_fn = TemporalAttentionMessage(dim, edge_dim, time_dim, num_heads)
        self.act = nn.Tanh()

    def forward(self, h, h_ngh, ngh_edge, ngh_time_enc, ngh_mask):
        """
        Euler integration: shared weights across steps.
        
        Args:
            h: [B, dim] current node states
            h_ngh: [B, N, dim] neighbor states
            ngh_edge: [B, N, edge_dim]
            ngh_time_enc: [B, N, time_dim]
            ngh_mask: [B, N]
        """
        for _ in range(self.num_steps):
            msg = self.message_fn(h, h_ngh, ngh_edge, ngh_time_enc, ngh_mask)
            dh = self.act(self.anti_sym(h) + msg)
            h = h + self.epsilon * dh
        return h


class CTAN(TemporalGNNBase):
    """
    Continuous-Time Anti-symmetric Network.
    
    Neural ODE with anti-symmetric weight (W - W^T - γI) for
    stable long-range propagation. Forward Euler integration.
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int = 1,
                 hidden_dim: int = 64, time_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, n_neighbors: int = 5,
                 ode_steps: int = 1, epsilon: float = 0.1, gamma: float = 0.1):
        super().__init__(num_nodes, node_feat_dim, edge_feat_dim, hidden_dim,
                         time_dim, num_layers, dropout, n_neighbors)

        edge_dim = hidden_dim if edge_feat_dim > 0 else 0

        # ODE blocks (one per layer, each can have multiple Euler steps)
        self.ode_blocks = nn.ModuleList([
            CTANODEBlock(hidden_dim, edge_dim, time_dim,
                         num_steps=ode_steps, epsilon=epsilon, gamma=gamma)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _get_neighbor_info(self, node_ids, timestamps, node_feat, edge_feat_all):
        """Get neighbor features, edge features, and time encodings."""
        B = len(node_ids)
        device = node_ids.device
        edge_dim = self.hidden_dim if self.edge_feat_dim > 0 else 0

        if self.neighbor_finder is not None:
            ngh_ids, ngh_eidx, ngh_times = self.neighbor_finder.get_temporal_neighbors(
                node_ids, timestamps, self.n_neighbors
            )
            ngh_ids = ngh_ids.to(device)
            ngh_times = ngh_times.to(device)
            ngh_mask = ngh_ids > 0

            # Neighbor features
            flat_ngh = ngh_ids.reshape(-1)
            h_ngh = self.get_node_features(flat_ngh, node_feat).reshape(B, self.n_neighbors, -1)
            h_ngh = h_ngh * ngh_mask.unsqueeze(-1).float()

            # Edge features
            if edge_feat_all is not None and self.edge_proj is not None:
                ngh_eidx_dev = ngh_eidx.to(device)
                safe_idx = ngh_eidx_dev.clamp(max=max(edge_feat_all.shape[0] - 1, 0))
                ngh_edge = self.edge_proj(edge_feat_all[safe_idx.reshape(-1)]).reshape(B, self.n_neighbors, -1)
                ngh_edge = ngh_edge * ngh_mask.unsqueeze(-1).float()
            else:
                ngh_edge = torch.zeros(B, self.n_neighbors, edge_dim, device=device)

            # Time encoding of elapsed time
            dt = (timestamps.unsqueeze(1) - ngh_times).clamp(min=0)
            ngh_time_enc = self.time_encoder(dt) * ngh_mask.unsqueeze(-1).float()
        else:
            ngh_mask = torch.zeros(B, self.n_neighbors, dtype=torch.bool, device=device)
            h_ngh = torch.zeros(B, self.n_neighbors, self.hidden_dim, device=device)
            ngh_edge = torch.zeros(B, self.n_neighbors, edge_dim, device=device)
            ngh_time_enc = torch.zeros(B, self.n_neighbors, self.time_dim, device=device)

        return h_ngh, ngh_edge, ngh_time_enc, ngh_mask

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        device = src.device

        # Initial node states
        src_h = self.get_node_features(src, node_feat)
        dst_h = self.get_node_features(dst, node_feat)

        # Get neighbor info for src and dst
        src_ngh, src_edge, src_time_enc, src_mask = \
            self._get_neighbor_info(src, t, node_feat, edge_feat)
        dst_ngh, dst_edge, dst_time_enc, dst_mask = \
            self._get_neighbor_info(dst, t, node_feat, edge_feat)

        # ODE integration with message passing
        for ode_block in self.ode_blocks:
            src_h = ode_block(src_h, src_ngh, src_edge, src_time_enc, src_mask)
            dst_h = ode_block(dst_h, dst_ngh, dst_edge, dst_time_enc, dst_mask)
            src_h = self.drop(src_h)
            dst_h = self.drop(dst_h)

        src_h = self.ln(src_h)
        dst_h = self.ln(dst_h)

        node_states = (src_h + dst_h) / 2
        return {"node_states": node_states, "src_states": src_h, "dst_states": dst_h}
