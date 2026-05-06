"""
Schema-guided document extraction pipeline.

Converts unstructured documents into structured policy tokens with:
  - Category, Trend, Risk, Duration fields
  - Calibrated confidence scores
  - Evidence spans for auditability
"""
import torch
import torch.nn as nn
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Schema field definitions
CATEGORIES = [
    "sanctions", "enforcement", "advisory", "scam_report",
    "market_bulletin", "reputation", "regulatory_update",
    "rate_decision", "enforcement_action", "supervisory_guidance",
    "consumer_protection", "monetary_policy", "fiscal_policy",
    "health_advisory", "market_regulation", "emergency_order",
]
TRENDS = ["increasing", "decreasing", "stable", "tightening", "easing", "neutral",
          "risk_increasing", "stabilizing", "risk_decreasing"]
RISKS = ["low", "medium", "high", "critical"]
DURATIONS = ["immediate", "30d", "60d", "90d", "1y", "indefinite"]


@dataclass
class ExtractedTuple:
    """A single extracted attribute tuple from a document."""
    category: str
    trend: str
    risk: str
    duration: str
    confidence_raw: float
    evidence_spans: Dict[str, str]
    source_doc_id: int
    timestamp: float


class TokenEncoder(nn.Module):
    """
    Maps extracted attribute tuples to continuous tokens.
    
    p_i = psi_theta(a_i)
    a_i = [emb(cat_i), trend_i, risk_i, duration_i]
    """

    def __init__(
        self,
        token_dim: int = 64,
        num_categories: int = len(CATEGORIES),
        num_trends: int = len(TRENDS),
        num_risks: int = len(RISKS),
        num_durations: int = len(DURATIONS),
        cat_emb_dim: int = 16,
    ):
        super().__init__()
        self.cat_embedding = nn.Embedding(num_categories + 1, cat_emb_dim)  # +1 for unknown
        self.trend_embedding = nn.Embedding(num_trends + 1, 8)
        self.risk_embedding = nn.Embedding(num_risks + 1, 8)
        self.duration_embedding = nn.Embedding(num_durations + 1, 8)

        input_dim = cat_emb_dim + 8 + 8 + 8
        self.projector = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
        )

        # Confidence calibration: c_i = sigma(w_c^T * xi(conf_raw, t_i))
        self.confidence_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        # Maps for encoding
        self.cat_map = {c: i for i, c in enumerate(CATEGORIES)}
        self.trend_map = {t: i for i, t in enumerate(TRENDS)}
        self.risk_map = {r: i for i, r in enumerate(RISKS)}
        self.dur_map = {d: i for i, d in enumerate(DURATIONS)}

    def encode_attributes(
        self,
        categories: List[str],
        trends: List[str],
        risks: List[str],
        durations: List[str],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Map string attributes to indices."""
        cat_idx = [self.cat_map.get(c, len(CATEGORIES)) for c in categories]
        trend_idx = [self.trend_map.get(t, len(TRENDS)) for t in trends]
        risk_idx = [self.risk_map.get(r, len(RISKS)) for r in risks]
        dur_idx = [self.dur_map.get(d, len(DURATIONS)) for d in durations]
        return (
            torch.tensor(cat_idx, dtype=torch.long),
            torch.tensor(trend_idx, dtype=torch.long),
            torch.tensor(risk_idx, dtype=torch.long),
            torch.tensor(dur_idx, dtype=torch.long),
        )

    def forward(
        self,
        cat_idx: torch.LongTensor,
        trend_idx: torch.LongTensor,
        risk_idx: torch.LongTensor,
        dur_idx: torch.LongTensor,
        conf_raw: torch.FloatTensor,
        timestamps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Encode extracted tuples to tokens with calibrated confidence.
        
        Args:
            cat_idx, trend_idx, risk_idx, dur_idx: [N] attribute indices
            conf_raw: [N] raw extractor confidence
            timestamps: [N] token timestamps
            
        Returns:
            token_embs: [N, token_dim] policy tokens p_i
            confidences: [N] calibrated confidence c_i
        """
        # Attribute embedding: a_i = [emb(cat), emb(trend), emb(risk), emb(dur)]
        a = torch.cat([
            self.cat_embedding(cat_idx),
            self.trend_embedding(trend_idx),
            self.risk_embedding(risk_idx),
            self.duration_embedding(dur_idx),
        ], dim=-1)

        # Token projection: p_i = psi_theta(a_i)
        token_embs = self.projector(a)

        # Calibrated confidence: c_i = sigma(w_c^T * xi(conf_raw, t_i))
        conf_input = torch.stack([conf_raw, timestamps], dim=-1)
        confidences = torch.sigmoid(self.confidence_mlp(conf_input).squeeze(-1))

        return token_embs, confidences


class DocumentExtractor:
    """
    Schema-guided extraction pipeline.
    
    In a real deployment, this would call an LLM (GPT-5, Qwen3, etc.)
    with a fixed JSON schema. Here we provide:
      1. The schema/prompt template for LLM-based extraction
      2. A loader for pre-extracted JSON files
      3. A synthetic generator for experiments without real docs
    """

    EXTRACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": CATEGORIES},
                        "trend": {"type": "string", "enum": TRENDS},
                        "risk": {"type": "string", "enum": RISKS},
                        "duration": {"type": "string", "enum": DURATIONS},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {
                            "type": "object",
                            "properties": {
                                "cat_span": {"type": "string"},
                                "trend_span": {"type": "string"},
                                "risk_span": {"type": "string"},
                                "duration_span": {"type": "string"},
                            },
                        },
                    },
                    "required": ["category", "trend", "risk", "confidence", "evidence"],
                },
            }
        },
    }

    EXTRACTION_PROMPT = """Extract structured policy/risk attributes from the following document.
Use the provided JSON schema. For each extracted attribute, provide a verbatim 
evidence span from the document that supports the extraction.

IMPORTANT:
- Each evidence span MUST be an exact substring of the source document.
- If you cannot find evidence for a field, set it to "unknown".
- Do NOT use knowledge from outside the provided document.
- Timestamps are NOT part of the extraction; they come from metadata.

Document:
{document_text}

Output JSON:"""

    @staticmethod
    def load_preextracted(json_path: str) -> List[ExtractedTuple]:
        """Load pre-extracted tuples from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        tuples = []
        for item in data:
            for ext in item.get("extractions", []):
                tuples.append(ExtractedTuple(
                    category=ext.get("category", "unknown"),
                    trend=ext.get("trend", "stable"),
                    risk=ext.get("risk", "medium"),
                    duration=ext.get("duration", "unknown"),
                    confidence_raw=ext.get("confidence", 0.5),
                    evidence_spans=ext.get("evidence", {}),
                    source_doc_id=item.get("doc_id", 0),
                    timestamp=item.get("timestamp", 0.0),
                ))
        return tuples

    @staticmethod
    def generate_synthetic(
        num_docs: int = 50,
        time_range: Tuple[float, float] = (1.0, 49.0),
        tuples_per_doc: int = 3,
        seed: int = 42,
    ) -> List[ExtractedTuple]:
        """
        Generate synthetic extracted tuples for experiments without real docs.
        Useful for testing the pipeline.
        """
        rng = np.random.RandomState(seed)
        tuples = []

        for doc_id in range(num_docs):
            doc_time = rng.uniform(time_range[0], time_range[1])
            n_tuples = rng.randint(1, tuples_per_doc + 1)
            for _ in range(n_tuples):
                tuples.append(ExtractedTuple(
                    category=rng.choice(CATEGORIES),
                    trend=rng.choice(TRENDS),
                    risk=rng.choice(RISKS),
                    duration=rng.choice(DURATIONS),
                    confidence_raw=rng.uniform(0.3, 1.0),
                    evidence_spans={},
                    source_doc_id=doc_id,
                    timestamp=doc_time,
                ))
        return tuples
