"""
Exogenous document stream generation/download script.

For each dataset, builds a time-stamped document stream from public sources:
  - Elliptic: OFAC sanctions, FinCEN advisories, DOJ/FBI enforcement releases
  - BTC-Alpha/OTC: Bitcointalk scam/reputation threads, CoinMetrics bulletins

Usage:
    python scripts/prepare_exo_docs.py --dataset elliptic --output data/elliptic/exo_docs.json
    python scripts/prepare_exo_docs.py --dataset btc_alpha --output data/btc_alpha/exo_docs.json
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def generate_elliptic_docs(seed: int = 42) -> List[Dict]:
    """
    Generate synthetic exogenous documents mimicking:
      - OFAC sanctions list updates
      - FinCEN AML advisories
      - DOJ/FBI enforcement releases
    
    In production, these would be fetched from:
      - https://www.treasury.gov/ofac/downloads/
      - https://www.fincen.gov/resources/advisories
      - https://www.justice.gov/news (filtered by AML keywords)
    
    Each document has:
      - doc_id: unique identifier
      - timestamp: publication time (mapped to Elliptic timesteps 1-49)
      - source: document source type
      - extractions: schema-guided structured attributes
    """
    rng = np.random.RandomState(seed)
    docs = []

    categories = ["sanctions", "enforcement", "advisory"]
    trends = ["increasing", "decreasing", "stable"]
    risks = ["low", "medium", "high", "critical"]
    durations = ["immediate", "90d", "1y", "indefinite"]

    doc_id = 0
    # Generate ~15 docs per source type across timeline
    for source in categories:
        n_docs = rng.randint(12, 18)
        times = sorted(rng.uniform(1, 49, n_docs))

        for t in times:
            n_extractions = rng.randint(1, 4)
            extractions = []
            for _ in range(n_extractions):
                # Simulate regime-shift signal around t=43
                if t >= 40:
                    risk_weights = [0.05, 0.15, 0.4, 0.4]
                    trend_weights = [0.6, 0.2, 0.2]
                else:
                    risk_weights = [0.3, 0.4, 0.2, 0.1]
                    trend_weights = [0.2, 0.2, 0.6]

                extractions.append({
                    "category": source,
                    "trend": rng.choice(trends, p=trend_weights),
                    "risk": rng.choice(risks, p=risk_weights),
                    "duration": rng.choice(durations),
                    "confidence": float(rng.uniform(0.6, 1.0)),
                    "evidence": {
                        "cat_span": f"[synthetic {source} evidence]",
                        "trend_span": "[synthetic trend evidence]",
                        "risk_span": "[synthetic risk evidence]",
                        "duration_span": "[synthetic duration evidence]",
                    },
                })

            docs.append({
                "doc_id": doc_id,
                "timestamp": float(t),
                "source": source,
                "extractions": extractions,
            })
            doc_id += 1

    return docs


def generate_btc_docs(name: str = "btc_otc", seed: int = 42) -> List[Dict]:
    """
    Generate synthetic exogenous documents mimicking:
      - Bitcointalk scam/reputation threads
      - CoinMetrics market/chain bulletins
    
    In production:
      - Bitcointalk: scrape scam-accusation boards with keyword filters
      - CoinMetrics: use public API for market indicators, template to text
    """
    rng = np.random.RandomState(seed)
    docs = []

    categories = ["scam_report", "reputation", "market_bulletin"]
    trends = ["increasing", "decreasing", "stable"]
    risks = ["low", "medium", "high", "critical"]
    durations = ["immediate", "30d", "90d", "indefinite"]

    doc_id = 0
    for source in categories:
        n_docs = rng.randint(15, 25)
        # Use Unix-like timestamps for BTC datasets
        times = sorted(rng.uniform(1e9, 1.6e9, n_docs))

        for t in times:
            n_ext = rng.randint(1, 3)
            extractions = []
            for _ in range(n_ext):
                extractions.append({
                    "category": source,
                    "trend": rng.choice(trends),
                    "risk": rng.choice(risks),
                    "duration": rng.choice(durations),
                    "confidence": float(rng.uniform(0.4, 1.0)),
                    "evidence": {
                        "cat_span": f"[synthetic {source} evidence]",
                        "trend_span": "[synthetic trend evidence]",
                        "risk_span": "[synthetic risk evidence]",
                        "duration_span": "[synthetic duration evidence]",
                    },
                })

            docs.append({
                "doc_id": doc_id,
                "timestamp": float(t),
                "source": source,
                "extractions": extractions,
            })
            doc_id += 1

    return docs


def main():
    parser = argparse.ArgumentParser(description="Prepare exogenous document stream")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["elliptic", "btc_alpha", "btc_otc"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/{args.dataset}/exo_docs.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "elliptic":
        docs = generate_elliptic_docs(args.seed)
    else:
        docs = generate_btc_docs(args.dataset, args.seed)

    with open(args.output, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Generated {len(docs)} documents -> {args.output}")
    total_extractions = sum(len(d["extractions"]) for d in docs)
    print(f"Total extraction tuples: {total_extractions}")


if __name__ == "__main__":
    main()
