"""D4: Faithfulness — entailment probability and contradiction score via NLI."""
import numpy as np
from dataclasses import dataclass
import torch
from transformers import pipeline as hf_pipeline

_nli_pipe = None


def _get_nli():
    global _nli_pipe
    if _nli_pipe is None:
        _nli_pipe = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nli_pipe


@dataclass
class D4Features:
    entailment_prob: float
    contradiction_score: float


def extract_d4(hypothesis: str, premise: str) -> D4Features:
    """hypothesis: query/answer, premise: passage text."""
    nli = _get_nli()
    result = nli(premise, candidate_labels=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"])
    scores = dict(zip(result["labels"], result["scores"]))
    return D4Features(
        entailment_prob=scores.get("ENTAILMENT", 0.0),
        contradiction_score=scores.get("CONTRADICTION", 0.0),
    )


def d4_to_array(feat: D4Features) -> np.ndarray:
    return np.array([feat.entailment_prob, feat.contradiction_score])
