"""
Run full experiment grid: all backbones x datasets.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --datasets elliptic --backbones tgn tgat
    python scripts/run_experiments.py --num_seeds 3
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_single_seed, load_config
import numpy as np


ALL_BACKBONES = ["tgn", "tgat", "graphmixer", "dygformer", "ctan", "generaldyg"]
ALL_DATASETS = ["elliptic", "btc_alpha", "btc_otc"]

# Per-model hyperparameters from Table A.1
HYPERPARAM_GRID = {
    ("btc_alpha", "graphmixer"): {"lr_gate": 1e-2, "stable_init": False, "alpha_max": 0.1},
    ("btc_alpha", "ctan"):      {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_alpha", "tgn"):       {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_alpha", "tgat"):      {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_alpha", "dygformer"): {"lr_gate": 1e-2, "stable_init": False, "alpha_max": 0.5},
    ("btc_alpha", "generaldyg"):{"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.25},

    ("btc_otc", "graphmixer"):  {"lr_gate": 1e-3, "stable_init": False, "alpha_max": 0.1},
    ("btc_otc", "ctan"):        {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_otc", "tgn"):         {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_otc", "tgat"):        {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("btc_otc", "dygformer"):   {"lr_gate": 1e-2, "stable_init": False, "alpha_max": 0.5},
    ("btc_otc", "generaldyg"):  {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.25},

    ("elliptic", "graphmixer"): {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("elliptic", "ctan"):       {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.3},
    ("elliptic", "tgn"):        {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("elliptic", "tgat"):       {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.1},
    ("elliptic", "dygformer"):  {"lr_gate": 1e-2, "stable_init": False, "alpha_max": 0.8},
    ("elliptic", "generaldyg"): {"lr_gate": 1e-2, "stable_init": True,  "alpha_max": 0.5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in args.datasets:
        for backbone in args.backbones:
            for use_regate in [False, True]:
                variant = "REGATE" if use_regate else "Original"
                key = f"{dataset}/{backbone}/{variant}"
                print(f"\n{'#'*60}")
                print(f"Running: {key}")
                print(f"{'#'*60}")

                # Build config
                cfg = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "use_regate": use_regate,
                    "num_seeds": args.num_seeds,
                    "hidden_dim": 64,
                    "time_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "token_dim": 64,
                    "gamma": 0.1,
                    "w_att": 14.0 if dataset == "elliptic" else 7.0,
                    "lambda_delta": 0.01,
                    "lambda_retr": 0.1,
                    "lambda_ent": 0.01,
                    "lambda_mask": 0.0,
                    "lr_model": 1e-4,
                    "weight_decay": 1e-5,
                    "batch_size": 200,
                    "num_epochs": 50,
                    "patience": 10,
                    "exo_source": "synthetic",
                    "num_synthetic_docs": 50,
                }

                # Apply per-model hyperparameters
                hp = HYPERPARAM_GRID.get((dataset, backbone), {})
                cfg.update(hp)

                # Auto-detect device
                import torch
                cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

                seed_results = []
                for seed in range(args.num_seeds):
                    try:
                        result = run_single_seed(cfg, seed)
                        seed_results.append(result)
                    except Exception as e:
                        print(f"ERROR seed {seed}: {e}")
                        continue

                if seed_results:
                    all_results[key] = seed_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_{timestamp}.json"

    # Convert to serializable format
    serializable = {}
    for key, seed_results in all_results.items():
        serializable[key] = []
        for sr in seed_results:
            entry = {}
            for k, v in sr.items():
                entry[k] = {mk: float(mv) for mk, mv in v.items()}
            serializable[key].append(entry)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, seed_results in all_results.items():
        print(f"\n{key}:")
        for metric_key in seed_results[0]:
            for metric_name in seed_results[0][metric_key]:
                values = [r[metric_key][metric_name] for r in seed_results]
                print(f"  {metric_key}/{metric_name}: "
                      f"{np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    main()
