"""D4: Faithfulness — entailment probability and contradiction score via NLI.

An NLI cross-encoder scores the ordered pair (premise = passage, hypothesis =
claim), where the claim is the HyDE pseudo-answer for the query. A passage that
entails the expected answer is faithful/useful; one that contradicts it is
actively harmful.
"""
import numpy as np
from dataclasses import dataclass

_nli_model = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def _get_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(_MODEL_NAME)
    return _nli_model


@dataclass
class D4Features:
    entailment_prob: float
    contradiction_score: float


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_d4(claim: str, passage: str) -> D4Features:
    """Score how well `passage` (premise) supports `claim` (hypothesis).

    `claim` is the HyDE pseudo-answer for the query — a declarative sentence,
    never the raw question.
    """
    model = _get_nli()
    logits = np.asarray(model.predict([(passage, claim)])[0], dtype=float)
    probs = _softmax(logits)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    by_label = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return D4Features(
        entailment_prob=by_label.get("entailment", 0.0),
        contradiction_score=by_label.get("contradiction", 0.0),
    )


def d4_to_array(feat: D4Features) -> np.ndarray:
    return np.array([feat.entailment_prob, feat.contradiction_score])
