"""
Data loaders for Elliptic, BTC-Alpha, BTC-OTC temporal graphs.

Each loader returns a TemporalGraphData object with:
  - src, dst: source/destination node indices (LongTensor)
  - t: timestamps (FloatTensor)
  - edge_feat: edge features (FloatTensor or None)
  - node_feat: node features (FloatTensor or None)
  - labels: node or edge labels (LongTensor, -1 for unlabeled)
  - train_mask, val_mask, test_mask
"""
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TemporalGraphData:
    src: torch.LongTensor
    dst: torch.LongTensor
    t: torch.FloatTensor
    edge_feat: Optional[torch.FloatTensor] = None
    node_feat: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None
    num_nodes: int = 0
    task: str = "node"  # "node" or "edge"
    # Temporal split masks (over events)
    train_mask: Optional[torch.BoolTensor] = None
    val_mask: Optional[torch.BoolTensor] = None
    test_mask: Optional[torch.BoolTensor] = None
    # For Elliptic: timestep per node
    node_timesteps: Optional[torch.LongTensor] = None


def load_elliptic(data_dir: str = "data/elliptic") -> TemporalGraphData:
    """
    Load Elliptic dataset for node-level AML detection.
    Temporal split: train=steps 1-34, val=35-39, test=40-49.
    """
    data_dir = Path(data_dir)

    # Load features (node_id, timestep, 165 features)
    feat_df = pd.read_csv(
        data_dir / "elliptic_txs_features.csv", header=None
    )
    node_ids = feat_df.iloc[:, 0].values
    timesteps = feat_df.iloc[:, 1].values.astype(int)
    features = feat_df.iloc[:, 2:].values.astype(np.float32)

    # Build node ID -> index mapping
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # Z-score normalize features using train period stats (steps 1-34)
    train_node_mask = timesteps <= 34
    feat_mean = features[train_node_mask].mean(axis=0)
    feat_std = features[train_node_mask].std(axis=0) + 1e-8
    features = (features - feat_mean) / feat_std

    # Load labels
    classes_df = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    label_map = {"1": 1, "2": 0, "unknown": -1}  # 1=illicit, 2=licit
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for _, row in classes_df.iterrows():
        idx = id_to_idx.get(row["txId"])
        if idx is not None:
            labels[idx] = label_map.get(str(row["class"]), -1)

    # Load edges
    edge_df = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    src_list, dst_list, t_list = [], [], []
    for _, row in edge_df.iterrows():
        s = id_to_idx.get(row["txId1"])
        d = id_to_idx.get(row["txId2"])
        if s is not None and d is not None:
            src_list.append(s)
            dst_list.append(d)
            # Use source node's timestep as edge time
            t_list.append(float(timesteps[s]))

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    t = torch.tensor(t_list, dtype=torch.float)

    # Sort by time
    sort_idx = torch.argsort(t)
    src, dst, t = src[sort_idx], dst[sort_idx], t[sort_idx]

    # Temporal split masks (on edges)
    edge_timesteps = t.numpy()
    train_mask = torch.tensor(edge_timesteps <= 34, dtype=torch.bool)
    val_mask = torch.tensor(
        (edge_timesteps >= 35) & (edge_timesteps <= 39), dtype=torch.bool
    )
    test_mask = torch.tensor(edge_timesteps >= 40, dtype=torch.bool)

    return TemporalGraphData(
        src=src,
        dst=dst,
        t=t,
        node_feat=torch.tensor(features, dtype=torch.float),
        labels=torch.tensor(labels, dtype=torch.long),
        num_nodes=num_nodes,
        task="node",
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_timesteps=torch.tensor(timesteps, dtype=torch.long),
    )


def load_btc_trust(data_dir: str, name: str = "btc_otc") -> TemporalGraphData:
    """
    Load BTC-Alpha or BTC-OTC trust network.
    Format: SOURCE, TARGET, RATING, TIME
    Edge-level trust prediction task.
    Temporal split: 70/10/20 chronological.
    """
    data_dir = Path(data_dir)
    if name == "btc_alpha":
        csv_file = data_dir / "soc-sign-bitcoinalpha.csv"
    else:
        csv_file = data_dir / "soc-sign-bitcoinotc.csv"

    df = pd.read_csv(csv_file, header=None, names=["src", "dst", "rating", "time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Remap node IDs
    all_nodes = sorted(set(df["src"].tolist() + df["dst"].tolist()))
    node_map = {n: i for i, n in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    src = torch.tensor([node_map[s] for s in df["src"]], dtype=torch.long)
    dst = torch.tensor([node_map[d] for d in df["dst"]], dtype=torch.long)
    t = torch.tensor(df["time"].values, dtype=torch.float)
    ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    # Binary labels: positive rating (>0) = 1, negative = 0
    labels = (ratings > 0).long()

    # Edge features: normalized rating
    edge_feat = ratings.unsqueeze(1).float() / 10.0

    # Temporal split
    n = len(src)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return TemporalGraphData(
        src=src,
        dst=dst,
        t=t,
        edge_feat=edge_feat,
        labels=labels,
        num_nodes=num_nodes,
        task="edge",
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_btc_alpha(data_dir: str = "data/btc_alpha") -> TemporalGraphData:
    return load_btc_trust(data_dir, name="btc_alpha")


def load_btc_otc(data_dir: str = "data/btc_otc") -> TemporalGraphData:
    return load_btc_trust(data_dir, name="btc_otc")


DATASET_REGISTRY = {
    "elliptic": load_elliptic,
    "btc_alpha": load_btc_alpha,
    "btc_otc": load_btc_otc,
}


def load_dataset(name: str, data_dir: str = None) -> TemporalGraphData:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY.keys())}")
    if data_dir is None:
        data_dir = f"data/{name}"
    return DATASET_REGISTRY[name](data_dir)
