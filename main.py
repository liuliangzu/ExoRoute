"""
Main entry point for REGATE experiments.

Usage:
    # Train with default config
    python main.py --config configs/default.yaml

    # Train specific backbone + dataset
    python main.py --backbone tgn --dataset elliptic --use_regate

    # Train baseline (no REGATE)
    python main.py --backbone tgn --dataset elliptic --no_regate

    # Multi-seed evaluation
    python main.py --backbone tgn --dataset elliptic --use_regate --num_seeds 5
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict

from data.dataset import load_dataset
from backbones import BACKBONE_REGISTRY
from regate import REGATEWrapper
from regate.document_extractor import DocumentExtractor
from utils.trainer import REGATETrainer
from utils.metrics import evaluate_all


def parse_args():
    parser = argparse.ArgumentParser(description="REGATE: Temporal Graph + Exogenous Text")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--use_regate", action="store_true", default=None)
    parser.add_argument("--no_regate", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha_max", type=float, default=None)
    parser.add_argument("--lr_gate", type=float, default=None)
    parser.add_argument("--lr_model", type=float, default=None)
    parser.add_argument("--stable_init", type=bool, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    return parser.parse_args()


def load_config(args) -> Dict:
    """Load YAML config and override with CLI args."""
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # CLI overrides
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    if args.no_regate:
        cfg["use_regate"] = False
    elif args.use_regate is not None:
        cfg["use_regate"] = True
    if args.num_seeds is not None:
        cfg["num_seeds"] = args.num_seeds
    if args.device is not None:
        cfg["device"] = args.device
    if args.alpha_max is not None:
        cfg["alpha_max"] = args.alpha_max
    if args.lr_gate is not None:
        cfg["lr_gate"] = args.lr_gate
    if args.lr_model is not None:
        cfg["lr_model"] = args.lr_model
    if args.stable_init is not None:
        cfg["stable_init"] = args.stable_init
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs

    # Defaults
    cfg.setdefault("dataset", "elliptic")
    cfg.setdefault("backbone", "tgn")
    cfg.setdefault("use_regate", True)
    cfg.setdefault("hidden_dim", 64)
    cfg.setdefault("time_dim", 64)
    cfg.setdefault("num_layers", 2)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("token_dim", 64)
    cfg.setdefault("alpha_max", 0.1)
    cfg.setdefault("gamma", 0.1)
    cfg.setdefault("w_att", 14.0)
    cfg.setdefault("stable_init", True)
    cfg.setdefault("lambda_delta", 0.01)
    cfg.setdefault("lambda_retr", 0.1)
    cfg.setdefault("lambda_ent", 0.01)
    cfg.setdefault("lambda_mask", 0.0)
    cfg.setdefault("lr_model", 1e-4)
    cfg.setdefault("lr_gate", 1e-2)
    cfg.setdefault("weight_decay", 1e-5)
    cfg.setdefault("batch_size", 200)
    cfg.setdefault("num_epochs", 50)
    cfg.setdefault("patience", 10)
    cfg.setdefault("num_seeds", 5)
    cfg.setdefault("exo_source", "synthetic")
    cfg.setdefault("num_synthetic_docs", 50)

    if "device" not in cfg:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def run_single_seed(cfg: Dict, seed: int) -> Dict:
    """Run one experiment with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["device"])
    print(f"\n{'='*60}")
    print(f"Seed {seed} | {cfg['backbone'].upper()} | {cfg['dataset']} | "
          f"REGATE={'ON' if cfg['use_regate'] else 'OFF'}")
    print(f"{'='*60}")

    # Load data
    data = load_dataset(cfg["dataset"])
    print(f"Loaded {cfg['dataset']}: {data.num_nodes} nodes, {len(data.src)} events, "
          f"task={data.task}")

    # Determine feature dimensions
    node_feat_dim = data.node_feat.shape[1] if data.node_feat is not None else 0
    edge_feat_dim = data.edge_feat.shape[1] if data.edge_feat is not None else 0

    # Build backbone
    BackboneCls = BACKBONE_REGISTRY[cfg["backbone"]]
    backbone = BackboneCls(
        num_nodes=data.num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=cfg["hidden_dim"],
        time_dim=cfg["time_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

    if cfg["use_regate"]:
        # Wrap with REGATE
        model = REGATEWrapper(
            backbone=backbone,
            node_dim=cfg["hidden_dim"],
            token_dim=cfg["token_dim"],
            hidden_dim=cfg["hidden_dim"],
            alpha_max=cfg["alpha_max"],
            gamma=cfg["gamma"],
            w_att=cfg["w_att"],
            dropout=cfg["dropout"],
            stable_init=cfg["stable_init"],
            lambda_delta=cfg["lambda_delta"],
            lambda_retr=cfg["lambda_retr"],
            lambda_ent=cfg["lambda_ent"],
            lambda_mask=cfg["lambda_mask"],
        )

        # Generate/load exogenous tokens
        if cfg["exo_source"] == "synthetic":
            if cfg["dataset"] == "elliptic":
                time_range = (1.0, 49.0)
            else:
                t_min, t_max = data.t.min().item(), data.t.max().item()
                time_range = (t_min, t_max)

            tuples = DocumentExtractor.generate_synthetic(
                num_docs=cfg["num_synthetic_docs"],
                time_range=time_range,
                seed=seed,
            )
        elif cfg["exo_source"] == "preextracted":
            tuples = DocumentExtractor.load_preextracted(cfg["exo_json_path"])
        else:
            tuples = DocumentExtractor.generate_synthetic(
                num_docs=cfg["num_synthetic_docs"],
                time_range=(data.t.min().item(), data.t.max().item()),
                seed=seed,
            )

        model.populate_memory(tuples, device=device)
        print(f"Populated token memory with {len(tuples)} tokens")
    else:
        # Baseline: backbone only with a simple wrapper
        model = BaselineWrapper(backbone)

    # Train
    trainer = REGATETrainer(
        model=model,
        lr_model=cfg["lr_model"],
        lr_gate=cfg["lr_gate"],
        weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"],
        num_epochs=cfg["num_epochs"],
        patience=cfg["patience"],
        device=cfg["device"],
        task=data.task,
    )

    trainer.train(data, verbose=True)

    # Evaluate
    results = {}
    test_metrics = trainer.evaluate(data, data.test_mask)
    results["test"] = test_metrics
    print(f"\nTest Results: AUC={test_metrics['auc']:.4f} AP={test_metrics['ap']:.4f} "
          f"ECE={test_metrics['ece']:.4f}")

    # Elliptic-specific: before/after regime shift windows
    if cfg["dataset"] == "elliptic":
        before = trainer.evaluate(data, data.test_mask, time_range=(40, 42))
        after = trainer.evaluate(data, data.test_mask, time_range=(43, 47))
        results["before_shift"] = before
        results["after_shift"] = after
        print(f"Before shift (t=40-42): AUC={before['auc']:.4f} AP={before['ap']:.4f} "
              f"ECE={before['ece']:.4f}")
        print(f"After shift  (t=43-47): AUC={after['auc']:.4f} AP={after['ap']:.4f} "
              f"ECE={after['ece']:.4f}")

    return results


class BaselineWrapper(torch.nn.Module):
    """Simple wrapper for baseline (no REGATE) to match interface."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, src, dst, t, edge_feat=None, node_feat=None):
        out = self.backbone(src, dst, t, edge_feat, node_feat)
        logits = self.backbone.readout(out["node_states"])
        return {
            "h_fused": out["node_states"],
            "h_base": out["node_states"],
            "delta_h": torch.zeros_like(out["node_states"]),
            "attn_weights": torch.zeros(len(src), 0, device=src.device),
            "logits": logits,
        }


def main():
    args = parse_args()
    cfg = load_config(args)

    print("Configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    all_results = []
    for seed in range(cfg["num_seeds"]):
        results = run_single_seed(cfg, seed)
        all_results.append(results)

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (mean ± std)")
    print("=" * 60)

    for key in all_results[0]:
        metrics = all_results[0][key]
        print(f"\n{key}:")
        for metric_name in metrics:
            values = [r[key][metric_name] for r in all_results]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
