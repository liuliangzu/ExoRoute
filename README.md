# ExoRoute

**ExoRoute: Reliability-Weighted Integration of Temporally Aligned Exogenous Text for Dynamic Graphs**

Liangzu Liu, Mengzhe Ruan, Yinjun Wu, Guanjun Wang, Yang Liu

**NeurIPS 2026** · [Paper and discussion](https://openreview.net/forum?id=hi0wCR0EYw)

ExoRoute integrates time-aligned external documents into temporal graph neural networks. It converts documents available at each event time into auditable structured tokens, routes them using node relevance, relative reliability, and recency, and fuses the evidence through a capped residual update. A lightweight document scorer learns from routing attention during training and remains fixed at inference.

The paper evaluates financial temporal graphs, including node-level anti-money-laundering detection on Elliptic and edge-level trust prediction on BTC-Alpha and BTC-OTC.

## Installation

Clone the repository and install its dependencies from the repository root.

```bash
git clone https://github.com/liuliangzu/ExoRoute.git
cd ExoRoute
python -m pip install -r requirements.txt
```

The requirements include Python >= 3.8, PyTorch >= 2.0.0, PyTorch Geometric >= 2.4.0, and Transformers >= 4.30.0. See [requirements.txt](requirements.txt) for the dependency list. Install PyTorch and the PyTorch Geometric extensions for your Python, CUDA, and hardware environment.

## Data preparation

Place the extracted graph dataset files in the following directories before training.

```text
data/
├── elliptic/
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
├── btc_alpha/
│   └── soc-sign-bitcoinalpha.csv
└── btc_otc/
    └── soc-sign-bitcoinotc.csv
```

The dataset identifiers accepted by `--dataset` are `elliptic`, `btc_alpha`, and `btc_otc`. The loaders use chronological splits: steps 1–34 / 35–39 / 40–49 for Elliptic and 70% / 10% / 20% for the BTC trust datasets. See [data/dataset.py](data/dataset.py) for the expected file formats and preprocessing.

### Exogenous documents

The [default configuration](configs/default.yaml) uses `exo_source: synthetic` to generate document tuples for pipeline checks. For experiments with real document streams, use pre-extracted tuples with timestamps aligned to the graph.

Copy the default configuration to a separate file.

```bash
cp configs/default.yaml configs/elliptic_documents.yaml
```

Set the following fields in the copied configuration.

```yaml
exo_source: preextracted
exo_json_path: data/elliptic/documents.json
```

The JSON file contains a list of document records. Each record supplies `doc_id`, `timestamp`, and an `extractions` list. Each extraction contains `category`, `trend`, `risk`, `duration`, `confidence`, and `evidence`. The evidence object stores the supporting text spans. The schema, extraction prompt, supported attribute values, and JSON loader are defined in [regate/document_extractor.py](regate/document_extractor.py).

Use the same time coordinate system for documents and graph events, and include only documents available at the corresponding prediction time. Elliptic uses discrete time steps; the BTC loaders retain the input timestamps.

## Training and evaluation

The implementation retains the `regate` package name, the `REGATEWrapper` class, and the `--use_regate` / `--no_regate` flags. These names refer to ExoRoute.

Run the default configuration after preparing the graph files.

```bash
python main.py --config configs/default.yaml
```

Run TGN with ExoRoute on Elliptic using the pre-extracted document configuration.

```bash
python main.py --config configs/elliptic_documents.yaml --backbone tgn --dataset elliptic --use_regate
```

Run the corresponding backbone-only baseline.

```bash
python main.py --config configs/elliptic_documents.yaml --backbone tgn --dataset elliptic --no_regate
```

Evaluate over five seeds.

```bash
python main.py --config configs/elliptic_documents.yaml --backbone tgn --dataset elliptic --use_regate --num_seeds 5
```

Command-line arguments override matching configuration fields. Use `python main.py --help` to inspect the available arguments. The runner selects CUDA when available and otherwise uses the CPU; `--device` overrides this choice. It prints per-seed metrics and aggregate means and standard deviations. For Elliptic, it also reports the pre-shift window at steps 40–42 and the post-shift window at steps 43–47.

### Supported backbones

| Backbone | `--backbone` value |
| --- | --- |
| TGAT | `tgat` |
| TGN | `tgn` |
| GraphMixer | `graphmixer` |
| DyGFormer | `dygformer` |
| CTAN | `ctan` |
| GeneralDyG | `generaldyg` |

Model and training settings are configured in [configs/default.yaml](configs/default.yaml), including the residual cap, temporal decay, attention window, learning rates, auxiliary loss weights, and number of seeds.

## Repository structure

```text
ExoRoute/
├── backbones/          # Temporal graph backbones
├── configs/            # Model and training configurations
├── data/               # Dataset loaders and local graph files
├── regate/             # ExoRoute components, retaining the original package name
│   ├── bounded_gate_fusion.py
│   ├── document_extractor.py
│   ├── regate_wrapper.py
│   ├── retrieval_adaptation.py
│   └── token_memory.py
├── scripts/            # Utility scripts
├── utils/              # Training and evaluation utilities
├── main.py             # Experiment entry point
└── requirements.txt
```

## Citation

If you use ExoRoute in your research, please cite the paper.

```bibtex
@inproceedings{liu2026exoroute,
  title     = {{ExoRoute}: Reliability-Weighted Integration of Temporally Aligned Exogenous Text for Dynamic Graphs},
  author    = {Liu, Liangzu and Ruan, Mengzhe and Wu, Yinjun and Wang, Guanjun and Liu, Yang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=hi0wCR0EYw}
}
```

## Contact

Please open an [issue](https://github.com/liuliangzu/ExoRoute/issues) for questions about the code.

## License

MIT License.
