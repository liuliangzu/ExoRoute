"""
Bounded Gated Fusion module for REGATE.

Implements:
  - Reliability-tilted routing (Eq. 7-8 in paper)
  - Bounded residual application (Eq. 9)
  - Algorithm 1 from the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BoundedGateFusion(nn.Module):
    """
    Bounded gated fusion: inject exogenous token information into backbone
    node states via reliability-tilted attention + bounded residual.
    
    Corresponds to Algorithm 1 (REGATE at event time t).
    """

    def __init__(
        self,
        node_dim: int,
        token_dim: int,
        hidden_dim: int = 64,
        alpha_max: float = 0.3,
        gamma: float = 0.1,
        dropout: float = 0.1,
        eps: float = 1e-6,
        stable_init: bool = True,
    ):
        """
        Args:
            node_dim: dimension of backbone node states h_v
            token_dim: dimension of policy tokens p_i
            hidden_dim: dimension for Q/K/V projections
            alpha_max: upper bound on gate coefficient
            gamma: temporal decay rate for reliability
            dropout: token dropout rate during training
            eps: small constant for numerical stability
            stable_init: if True, initialize gate logit g << 0
        """
        super().__init__()
        self.node_dim = node_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.alpha_max = alpha_max
        self.gamma = gamma
        self.dropout_rate = dropout
        self.eps = eps

        # Q/K/V projections
        self.f_q = nn.Linear(node_dim, hidden_dim)
        self.f_k = nn.Linear(token_dim, hidden_dim)
        self.f_v = nn.Linear(token_dim, node_dim)

        # LayerNorm for the aggregated proposal
        self.ln = nn.LayerNorm(node_dim)

        # Learnable gate logit g: alpha = alpha_max * sigmoid(g)
        init_val = -5.0 if stable_init else 0.0
        self.gate_logit = nn.Parameter(torch.tensor(init_val))

        self.scale = math.sqrt(hidden_dim)

    @property
    def alpha(self):
        """Current gate value, bounded in [0, alpha_max]."""
        return self.alpha_max * torch.sigmoid(self.gate_logit)

    def forward(
        self,
        h_base: torch.FloatTensor,
        token_embs: torch.FloatTensor,
        token_times: torch.FloatTensor,
        token_confs: torch.FloatTensor,
        event_times: torch.FloatTensor,
        attn_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Apply bounded gated fusion.
        
        Args:
            h_base: [B, d] backbone node states
            token_embs: [B, K, d_p] policy token embeddings
            token_times: [B, K] token timestamps
            token_confs: [B, K] token confidence scores
            event_times: [B] current event times
            attn_mask: [B, K] True = valid token
            
        Returns:
            h_fused: [B, d] fused node states
            delta_h: [B, d] exogenous proposal (for loss computation)
            attn_weights: [B, K] routing attention weights
        """
        B, K, _ = token_embs.shape
        device = h_base.device

        if K == 0 or (attn_mask is not None and not attn_mask.any()):
            return h_base, torch.zeros_like(h_base), torch.zeros(B, 0, device=device)

        # Compute reliability weights: rho_i(t) = c_i * exp(-gamma * [t - t_i]+)
        # [B, K]
        time_diff = (event_times.unsqueeze(1) - token_times).clamp(min=0)
        rho = token_confs * torch.exp(-self.gamma * time_diff)

        # Token dropout during training
        if self.training and self.dropout_rate > 0:
            drop_mask = torch.bernoulli(
                torch.full((B, K), 1 - self.dropout_rate, device=device)
            ).bool()
            if attn_mask is not None:
                attn_mask = attn_mask & drop_mask
            else:
                attn_mask = drop_mask

        # Q/K/V projections
        q = self.f_q(h_base)         # [B, d_k]
        k = self.f_k(token_embs)     # [B, K, d_k]
        v = self.f_v(token_embs)     # [B, K, d]

        # Scaled dot-product attention logits: S_vi = <q_v, k_i> / sqrt(d_k)
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) / self.scale
        # [B, K]

        # Add log-reliability: S_vi + log(eps + rho_i)
        scores = scores + torch.log(self.eps + rho)

        # Mask invalid tokens
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)  # [B, K]
        attn_weights = attn_weights.masked_fill(
            torch.isnan(attn_weights), 0.0
        )

        # Aggregate: delta_h = LN(sum_i a_vi * v_i)
        weighted_v = torch.bmm(attn_weights.unsqueeze(1), v).squeeze(1)  # [B, d]
        delta_h = self.ln(weighted_v)

        # Bounded residual: h_fused = h_base + alpha * delta_h
        alpha = self.alpha
        h_fused = h_base + alpha * delta_h

        return h_fused, alpha * delta_h, attn_weights

    def intervention_loss(self, delta_h: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute L_delta = E[||alpha * delta_h||^2] for regularization.
        
        Args:
            delta_h: [B, d] the alpha * delta_h values
        """
        return (delta_h ** 2).sum(dim=-1).mean()

    def entropy_loss(self, attn_weights: torch.FloatTensor) -> torch.FloatTensor:
        """
        Routing entropy regularizer L_ent to prevent degenerate concentration.
        
        L_ent = E[-sum_i a_vi * log(a_vi + eps)]
        """
        log_weights = torch.log(attn_weights + self.eps)
        entropy = -(attn_weights * log_weights).sum(dim=-1)
        return entropy.mean()
