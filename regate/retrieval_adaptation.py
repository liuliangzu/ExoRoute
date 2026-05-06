"""
Closed-loop retrieval adaptation via routing distillation.

Implements:
  - Lightweight retrieval scorer (Eq. 5-6)
  - Routing-based distillation target (Eq. 7-8)
  - KL distillation loss (Eq. 9)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RetrievalScorer(nn.Module):
    """
    Lightweight MLP scorer: s_phi(r, b_k) = f_phi(enc(r), feat(b_k)).
    One-hidden-layer tanh network as described in the paper.
    """

    def __init__(self, doc_dim: int, bin_feat_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(doc_dim + bin_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, doc_encodings: torch.FloatTensor, bin_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            doc_encodings: [N_docs, doc_dim] frozen text encoder outputs
            bin_features: [bin_feat_dim] bin-level features (time index, etc.)
            
        Returns:
            scores: [N_docs] retrieval scores
        """
        bin_feat_exp = bin_features.unsqueeze(0).expand(len(doc_encodings), -1)
        combined = torch.cat([doc_encodings, bin_feat_exp], dim=-1)
        return self.scorer(combined).squeeze(-1)


class RetrievalAdaptation(nn.Module):
    """
    Closed-loop retrieval adaptation module.
    
    During training, distills routing attention into a lightweight scorer
    via KL divergence without manual relevance labels.
    """

    def __init__(
        self,
        doc_dim: int,
        bin_feat_dim: int = 16,
        hidden_dim: int = 64,
        tau_sel: float = 1.0,
        top_k: int = 10,
    ):
        """
        Args:
            doc_dim: dimension of frozen document encodings
            bin_feat_dim: dimension of bin-level features
            hidden_dim: hidden dimension for scorer MLP
            tau_sel: temperature for selection softmax
            top_k: K_ret, number of documents to retrieve at inference
        """
        super().__init__()
        self.scorer = RetrievalScorer(doc_dim, bin_feat_dim, hidden_dim)
        self.tau_sel = tau_sel
        self.top_k = top_k

    def compute_selection_distribution(
        self,
        doc_encodings: torch.FloatTensor,
        bin_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute pi_phi(r | b_k) = softmax(s_phi(r, b_k) / tau_sel).
        
        Returns:
            pi: [N_docs] selection distribution
        """
        scores = self.scorer(doc_encodings, bin_features)
        return F.softmax(scores / self.tau_sel, dim=0)

    def retrieve_top_k(
        self,
        doc_encodings: torch.FloatTensor,
        bin_features: torch.FloatTensor,
    ) -> torch.LongTensor:
        """
        Retrieve Top-K_ret documents by scorer.
        
        Returns:
            indices: [K_ret] indices of top-K documents
        """
        scores = self.scorer(doc_encodings, bin_features)
        k = min(self.top_k, len(scores))
        _, top_indices = scores.topk(k)
        return top_indices

    @staticmethod
    def compute_routing_mass(
        attn_weights: torch.FloatTensor,
        token_source_ids: torch.LongTensor,
        num_docs: int,
        eps: float = 1e-8,
    ) -> torch.FloatTensor:
        """
        Compute routing mass m(r, b_k) for each document.
        Eq. 7: average routing attention mass per document.
        
        Args:
            attn_weights: [N_events, K_tokens] routing weights from fusion
            token_source_ids: [K_tokens] source document ID per token
            num_docs: total number of candidate documents
            
        Returns:
            mass: [num_docs] aggregated routing mass per document
        """
        mass = torch.zeros(num_docs, device=attn_weights.device)

        for doc_id in range(num_docs):
            # Find tokens from this document
            token_mask = token_source_ids == doc_id
            if token_mask.sum() == 0:
                continue

            # Average routing attention over tokens from this doc and events
            doc_attn = attn_weights[:, token_mask]  # [N_events, n_tokens_doc]
            # Average over tokens, then over events
            per_event = doc_attn.mean(dim=1)  # [N_events]
            mass[doc_id] = per_event.mean()

        return mass

    @staticmethod
    def compute_distillation_target(
        routing_mass: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute y^gate(r | b_k) = Normalize(m(r, b_k)).
        Eq. 8: normalized routing mass as distillation target.
        """
        total = routing_mass.sum()
        if total < 1e-10:
            return torch.ones_like(routing_mass) / len(routing_mass)
        return routing_mass / total

    def distillation_loss(
        self,
        doc_encodings: torch.FloatTensor,
        bin_features: torch.FloatTensor,
        y_gate: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute L_retr = KL(sg(y_gate) || pi_phi).
        Eq. 9: KL divergence between stop-gradient target and scorer.
        
        Args:
            doc_encodings: [N_docs, doc_dim]
            bin_features: [bin_feat_dim]
            y_gate: [N_docs] distillation target (stop-gradient applied here)
            
        Returns:
            loss: scalar KL divergence
        """
        pi = self.compute_selection_distribution(doc_encodings, bin_features)
        y_gate_sg = y_gate.detach()  # stop gradient

        # KL(y_gate || pi) = sum y_gate * log(y_gate / pi)
        # Use F.kl_div which expects log(pi)
        loss = F.kl_div(
            torch.log(pi + 1e-10),
            y_gate_sg,
            reduction="sum",
            log_target=False,
        )
        return loss
