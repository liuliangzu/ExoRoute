"""
Training loop for REGATE.

Implements:
  - Full objective: L = L_pred + lambda_delta * L_delta + lambda_retr * L_retr
                      + lambda_ent * L_ent + lambda_mask * L_mask
  - Strict temporal ordering
  - Multi-seed evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm

from utils.metrics import evaluate_all


class REGATETrainer:
    """Trainer for REGATE with temporal GNN backbones."""

    def __init__(
        self,
        model: nn.Module,
        lr_model: float = 1e-4,
        lr_gate: float = 1e-2,
        weight_decay: float = 1e-5,
        batch_size: int = 200,
        num_epochs: int = 50,
        patience: int = 10,
        device: str = "cpu",
        task: str = "node",  # "node" or "edge"
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.task = task

        # Separate optimizer groups: backbone vs REGATE-specific params
        backbone_params = []
        gate_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["gate_fusion", "token_encoder", "retrieval"]):
                gate_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": lr_model, "weight_decay": weight_decay},
            {"params": gate_params, "lr": lr_gate, "weight_decay": weight_decay},
        ])

    def train_epoch(self, data) -> Dict[str, float]:
        """Train one epoch with strict temporal ordering."""
        self.model.train()

        # Get training events
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        n_train = len(train_idx)
        # Maintain temporal order (no shuffling)
        losses = {"total": 0.0, "pred": 0.0, "delta": 0.0, "ent": 0.0}
        n_batches = 0

        for start in range(0, n_train, self.batch_size):
            end = min(start + self.batch_size, n_train)
            batch_idx = train_idx[start:end]

            src = data.src[batch_idx].to(self.device)
            dst = data.dst[batch_idx].to(self.device)
            t = data.t[batch_idx].to(self.device)
            edge_feat = None
            if data.edge_feat is not None:
                edge_feat = data.edge_feat[batch_idx].to(self.device)

            node_feat = None
            if data.node_feat is not None:
                node_feat = data.node_feat.to(self.device)

            # Forward
            out = self.model(src, dst, t, edge_feat, node_feat)
            logits = out["logits"]

            # Get labels based on task
            if self.task == "node":
                # Node-level: use source node labels
                labels = data.labels[data.src[batch_idx]].to(self.device)
            else:
                # Edge-level: use edge labels
                labels = data.labels[batch_idx].to(self.device)

            # Filter out unlabeled (-1)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue

            logits_valid = logits[valid_mask]
            labels_valid = labels[valid_mask].float()

            # Prediction loss (BCE)
            l_pred = F.binary_cross_entropy_with_logits(logits_valid, labels_valid)

            # Auxiliary losses
            total_loss = l_pred
            aux_losses = {}

            if hasattr(self.model, "compute_aux_losses"):
                aux_losses = self.model.compute_aux_losses(
                    out["delta_h"], out["attn_weights"]
                )
                for k, v in aux_losses.items():
                    total_loss = total_loss + v

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses["total"] += total_loss.item()
            losses["pred"] += l_pred.item()
            losses["delta"] += aux_losses.get("l_delta", torch.tensor(0.0)).item()
            losses["ent"] += aux_losses.get("l_ent", torch.tensor(0.0)).item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, data, mask: torch.BoolTensor, 
                 time_range: Optional[tuple] = None) -> Dict[str, float]:
        """Evaluate on a subset of events."""
        self.model.eval()

        eval_idx = mask.nonzero(as_tuple=True)[0]

        # Optional time range filter
        if time_range is not None:
            t_vals = data.t[eval_idx]
            time_mask = (t_vals >= time_range[0]) & (t_vals <= time_range[1])
            eval_idx = eval_idx[time_mask]

        if len(eval_idx) == 0:
            return {"auc": 0.0, "ap": 0.0, "ece": 0.0}

        all_logits = []
        all_labels = []

        for start in range(0, len(eval_idx), self.batch_size):
            end = min(start + self.batch_size, len(eval_idx))
            batch_idx = eval_idx[start:end]

            src = data.src[batch_idx].to(self.device)
            dst = data.dst[batch_idx].to(self.device)
            t = data.t[batch_idx].to(self.device)
            edge_feat = None
            if data.edge_feat is not None:
                edge_feat = data.edge_feat[batch_idx].to(self.device)
            node_feat = None
            if data.node_feat is not None:
                node_feat = data.node_feat.to(self.device)

            out = self.model(src, dst, t, edge_feat, node_feat)
            logits = out["logits"]

            if self.task == "node":
                labels = data.labels[data.src[batch_idx]]
            else:
                labels = data.labels[batch_idx]

            valid = labels >= 0
            if valid.sum() > 0:
                all_logits.append(torch.sigmoid(logits[valid]).cpu().numpy())
                all_labels.append(labels[valid].cpu().numpy())

        if not all_logits:
            return {"auc": 0.0, "ap": 0.0, "ece": 0.0}

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)

        return evaluate_all(y_true, y_score)

    def train(self, data, verbose: bool = True) -> Dict[str, list]:
        """Full training loop with early stopping on validation AP."""
        best_val_ap = -1.0
        patience_counter = 0
        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(self.num_epochs):
            # Reset TGN memory if applicable
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "tgn_memory"):
                self.model.backbone.tgn_memory.reset()
            elif hasattr(self.model, "tgn_memory"):
                self.model.tgn_memory.reset()

            losses = self.train_epoch(data)
            history["train_loss"].append(losses["total"])

            # Validate
            val_metrics = self.evaluate(data, data.val_mask)
            history["val_metrics"].append(val_metrics)

            if verbose:
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Loss: {losses['total']:.4f} (pred={losses['pred']:.4f}) | "
                    f"Val AUC: {val_metrics['auc']:.4f} AP: {val_metrics['ap']:.4f} "
                    f"ECE: {val_metrics['ece']:.4f}"
                )

            # Early stopping on AP
            if val_metrics["ap"] > best_val_ap:
                best_val_ap = val_metrics["ap"]
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        self.model.load_state_dict(best_state)
        return history
