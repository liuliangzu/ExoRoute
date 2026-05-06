from .tgn import TGN
from .tgat import TGAT
from .graph_mixer import GraphMixer
from .dygformer import DyGFormer
from .ctan import CTAN
from .general_dyg import GeneralDyG

BACKBONE_REGISTRY = {
    "tgn": TGN,
    "tgat": TGAT,
    "graphmixer": GraphMixer,
    "dygformer": DyGFormer,
    "ctan": CTAN,
    "generaldyg": GeneralDyG,
}
