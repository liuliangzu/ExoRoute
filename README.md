# REGATE

REGATE: Retrieval-Augmented Gate Fusion for Temporal Graph Networks with Exogenous Documents

This implementation integrates retrieval-augmented document processing with temporal graph networks for improved node classification on temporal graphs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train with default config
```bash
python main.py --config configs/default.yaml
```

### Train specific backbone + dataset
```bash
python main.py --backbone tgn --dataset elliptic --use_regate
```

### Train baseline (no REGATE)
```bash
python main.py --backbone tgn --dataset elliptic --no_regate
```

### Multi-seed evaluation
```bash
python main.py --backbone tgn --dataset elliptic --use_regate --num_seeds 5
```

## Supported Backbones

- TGAT (Temporal Graph Attention)
- TGN (Temporal Graph Network)
- DyGFormer
- GraphMixer
- CTAN
- GeneralDYG

## Project Structure

```
regate_code/
├── backbones/          # Temporal graph network implementations
├── configs/            # Configuration files
├── data/               # Dataset loaders
├── regate/             # REGATE components
│   ├── bounded_gate_fusion.py
│   ├── document_extractor.py
│   ├── regate_wrapper.py
│   ├── retrieval_adaptation.py
│   └── token_memory.py
├── scripts/            # Utility scripts
├── utils/              # Training utilities
└── main.py            # Main entry point
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- Transformers >= 4.30.0

## License

MIT License
