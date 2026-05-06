"""
REGATE wrapper that attaches the exogenous channel to any backbone.

Implements the full pipeline:
  1. Document extraction -> token encoding
  2. Time-indexed memory
  3. Bounded gated fusion
  4. Retrieval adaptation (training only)
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from .token_memory import TokenMemory
from .bounded_gate_fusion import BoundedGateFusion
from .retrieval_adaptation import RetrievalAdaptation
from .document_extractor import TokenEncoder, DocumentExtractor, ExtractedTuple


class REGATEWrapper(nn.Module):
    """
    Wraps a backbone temporal GNN with the REGATE exogenous channel.
    
    The backbone must implement:
      - forward(src, dst, t, edge_feat, node_feat) -> node_states [B, d]
      - get_node_dim() -> int
    """

    def __init__(
        self,
        backbone: nn.Module,
        node_dim: int,
        token_dim: int = 64,
        hidden_dim: int = 64,
        alpha_max: float = 0.3,
        gamma: float = 0.1,
        w_att: float = 14.0,
        dropout: float = 0.1,
        stable_init: bool = True,
        # Loss weights
        lambda_delta: float = 0.01,
        lambda_retr: float = 0.1,
        lambda_ent: float = 0.01,
        lambda_mask: float = 0.0,
        # Retrieval
        doc_dim: int = 768,
        bin_feat_dim: int = 16,
        top_k_ret: int = 10,
        tau_sel: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.node_dim = node_dim

        # Token encoder
        self.token_encoder = TokenEncoder(token_dim=token_dim)

        # Token memory
        self.token_memory = TokenMemory(token_dim=token_dim, w_att=w_att)

        # Bounded gated fusion
        self.gate_fusion = BoundedGateFusion(
            node_dim=node_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            alpha_max=alpha_max,
            gamma=gamma,
            dropout=dropout,
            stable_init=stable_init,
        )

        # Retrieval adaptation
        self.retrieval = RetrievalAdaptation(
            doc_dim=doc_dim,
            bin_feat_dim=bin_feat_dim,
            hidden_dim=hidden_dim,
            tau_sel=tau_sel,
            top_k=top_k_ret,
        )

        # Loss weights
        self.lambda_delta = lambda_delta
        self.lambda_retr = lambda_retr
        self.lambda_ent = lambda_ent
        self.lambda_mask = lambda_mask

    def populate_memory(
        self,
        extracted_tuples: List[ExtractedTuple],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Encode extracted tuples and populate the token memory.
        Called once before training/inference.
        """
        if not extracted_tuples:
            return

        categories = [t.category for t in extracted_tuples]
        trends = [t.trend for t in extracted_tuples]
        risks = [t.risk for t in extracted_tuples]
        durations = [t.duration for t in extracted_tuples]
        conf_raws = [t.confidence_raw for t in extracted_tuples]
        timestamps = [t.timestamp for t in extracted_tuples]
        source_ids = [t.source_doc_id for t in extracted_tuples]

        cat_idx, trend_idx, risk_idx, dur_idx = self.token_encoder.encode_attributes(
            categories, trends, risks, durations
        )

        cat_idx = cat_idx.to(device)
        trend_idx = trend_idx.to(device)
        risk_idx = risk_idx.to(device)
        dur_idx = dur_idx.to(device)
        conf_raw = torch.tensor(conf_raws, dtype=torch.float, device=device)
        ts = torch.tensor(timestamps, dtype=torch.float, device=device)
        src_ids = torch.tensor(source_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            token_embs, confidences = self.token_encoder(
                cat_idx, trend_idx, risk_idx, dur_idx, conf_raw, ts
            )

        self.token_memory.reset()
        self.token_memory.to(device)
        self.token_memory.add_tokens(token_embs, ts, confidences, src_ids)

    def forward(
        self,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        t: torch.FloatTensor,
        edge_feat: Optional[torch.FloatTensor] = None,
        node_feat: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with REGATE fusion.
        
        Returns dict with:
          - h_fused: [B, d] fused node states
          - h_base: [B, d] backbone states
          - delta_h: [B, d] intervention
          - attn_weights: [B, K] routing weights
          - logits: [B] prediction logits (from backbone head)
        """
        # Get backbone node states
        backbone_out = self.backbone(src, dst, t, edge_feat, node_feat)
        h_base = backbone_out["node_states"]  # [B, d]

        # Retrieve active tokens from memory
        token_embs, token_times, token_confs, token_src_ids, attn_mask = \
            self.token_memory.get_active_set_batch(t)

        # Apply bounded gated fusion
        h_fused, delta_h, attn_weights = self.gate_fusion(
            h_base, token_embs, token_times, token_confs, t, attn_mask
        )

        # Compute logits using backbone's readout head
        logits = self.backbone.readout(h_fused)

        return {
            "h_fused": h_fused,
            "h_base": h_base,
            "delta_h": delta_h,
            "attn_weights": attn_weights,
            "logits": logits,
        }

    def compute_aux_losses(
        self,
        delta_h: torch.FloatTensor,
        attn_weights: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        """Compute auxiliary losses for REGATE."""
        losses = {}

        # Intervention magnitude penalty
        if self.lambda_delta > 0:
            losses["l_delta"] = self.lambda_delta * self.gate_fusion.intervention_loss(delta_h)

        # Routing entropy regularizer
        if self.lambda_ent > 0 and attn_weights.numel() > 0:
            losses["l_ent"] = -self.lambda_ent * self.gate_fusion.entropy_loss(attn_weights)

        return losses
