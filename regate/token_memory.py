"""
Time-indexed Token Memory for REGATE.

Stores structured policy tokens with timestamps and confidence scores.
Supports time-causal retrieval via sliding attention windows.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PolicyToken:
    """A single structured exogenous token."""
    embedding: torch.FloatTensor    # p_i: [d_p]
    timestamp: float                # t_i
    confidence: float               # c_i
    source_doc_id: int              # r(i): pointer to source document
    entity_prior: Optional[List[int]] = None  # Pi_i: optional entity set


class TokenMemory(nn.Module):
    """
    Time-indexed token memory M = {(p_i, t_i, c_i, r(i), Pi_i)}.
    
    At event time t, returns the active subset A(t) = {i | t_i in [t - W_att, t]}.
    """

    def __init__(self, token_dim: int, w_att: float = 14.0):
        super().__init__()
        self.token_dim = token_dim
        self.w_att = w_att

        # Storage tensors (dynamically populated)
        self.register_buffer("embeddings", torch.zeros(0, token_dim))
        self.register_buffer("timestamps", torch.zeros(0))
        self.register_buffer("confidences", torch.zeros(0))
        self.register_buffer("source_ids", torch.zeros(0, dtype=torch.long))

    def reset(self):
        """Clear all stored tokens."""
        device = self.embeddings.device
        self.embeddings = torch.zeros(0, self.token_dim, device=device)
        self.timestamps = torch.zeros(0, device=device)
        self.confidences = torch.zeros(0, device=device)
        self.source_ids = torch.zeros(0, dtype=torch.long, device=device)

    def add_tokens(
        self,
        embeddings: torch.FloatTensor,
        timestamps: torch.FloatTensor,
        confidences: torch.FloatTensor,
        source_ids: torch.LongTensor,
    ):
        """
        Add batch of tokens to memory.
        
        Args:
            embeddings: [N, d_p]
            timestamps: [N]
            confidences: [N]
            source_ids: [N]
        """
        self.embeddings = torch.cat([self.embeddings, embeddings.detach()], dim=0)
        self.timestamps = torch.cat([self.timestamps, timestamps.detach()], dim=0)
        self.confidences = torch.cat([self.confidences, confidences.detach()], dim=0)
        self.source_ids = torch.cat([self.source_ids, source_ids.detach()], dim=0)

    def get_active_set(self, t: float):
        """
        Retrieve the active token subset A(t) = {i | t_i in [t - W_att, t]}.
        
        Returns:
            embeddings: [K, d_p] where K = |A(t)|
            timestamps: [K]
            confidences: [K]
            source_ids: [K]
            mask: boolean tensor of active indices
        """
        if len(self.timestamps) == 0:
            return None, None, None, None, None

        mask = (self.timestamps >= t - self.w_att) & (self.timestamps <= t)

        if mask.sum() == 0:
            return None, None, None, None, None

        return (
            self.embeddings[mask],
            self.timestamps[mask],
            self.confidences[mask],
            self.source_ids[mask],
            mask,
        )

    def get_active_set_batch(self, times: torch.FloatTensor):
        """
        Batch retrieval: for each time t in times, get active set.
        Returns padded tensors with attention mask.
        
        Args:
            times: [B] event times
            
        Returns:
            token_embs: [B, max_K, d_p]
            token_times: [B, max_K]
            token_confs: [B, max_K]
            token_src_ids: [B, max_K]
            attn_mask: [B, max_K] (True = valid token)
        """
        B = len(times)
        if len(self.timestamps) == 0:
            return (
                torch.zeros(B, 0, self.token_dim, device=times.device),
                torch.zeros(B, 0, device=times.device),
                torch.zeros(B, 0, device=times.device),
                torch.zeros(B, 0, dtype=torch.long, device=times.device),
                torch.zeros(B, 0, dtype=torch.bool, device=times.device),
            )

        # Compute mask: [B, N_tokens]
        t_exp = times.unsqueeze(1)  # [B, 1]
        ts_exp = self.timestamps.unsqueeze(0)  # [1, N]
        mask = (ts_exp >= t_exp - self.w_att) & (ts_exp <= t_exp)  # [B, N]

        max_k = mask.sum(dim=1).max().item()
        if max_k == 0:
            return (
                torch.zeros(B, 0, self.token_dim, device=times.device),
                torch.zeros(B, 0, device=times.device),
                torch.zeros(B, 0, device=times.device),
                torch.zeros(B, 0, dtype=torch.long, device=times.device),
                torch.zeros(B, 0, dtype=torch.bool, device=times.device),
            )

        # Pad to max_k
        token_embs = torch.zeros(B, max_k, self.token_dim, device=times.device)
        token_times = torch.zeros(B, max_k, device=times.device)
        token_confs = torch.zeros(B, max_k, device=times.device)
        token_src_ids = torch.zeros(B, max_k, dtype=torch.long, device=times.device)
        attn_mask = torch.zeros(B, max_k, dtype=torch.bool, device=times.device)

        for b in range(B):
            active_idx = mask[b].nonzero(as_tuple=True)[0]
            k = len(active_idx)
            if k > 0:
                token_embs[b, :k] = self.embeddings[active_idx]
                token_times[b, :k] = self.timestamps[active_idx]
                token_confs[b, :k] = self.confidences[active_idx]
                token_src_ids[b, :k] = self.source_ids[active_idx]
                attn_mask[b, :k] = True

        return token_embs, token_times, token_confs, token_src_ids, attn_mask
