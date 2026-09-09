"""Evaluation metrics: Spearman ρ, NDCG@k, MRR, EM, F1, combined_score."""
import re
import string
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    pred = _normalize(prediction)
    return float(any(_normalize(a) == pred for a in gold_answers))


def token_f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = set(_normalize(prediction).split())
    best = 0.0
    for gold in gold_answers:
        gold_tokens = set(_normalize(gold).split())
        if not pred_tokens or not gold_tokens:
            continue
        common = pred_tokens & gold_tokens
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        best = max(best, f1)
    return best


def combined_score(prediction: str, gold_answers: list[str]) -> float:
    return 0.5 * exact_match_score(prediction, gold_answers) + \
           0.5 * token_f1_score(prediction, gold_answers)


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    # ndcg_score requires non-negative relevance
    y_shifted = y_true - y_true.min()
    return float(ndcg_score(y_shifted[np.newaxis, :], y_pred[np.newaxis, :], k=k))


def mrr(relevant_ranks: list[int]) -> float:
    """Mean Reciprocal Rank. relevant_ranks: 1-indexed rank of first relevant doc per query."""
    if not relevant_ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in relevant_ranks if r > 0]))


def exact_match(predictions: list[str], references: list[str]) -> float:
    return float(np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)]))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
    recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    qa_predictions: list[str] | None = None,
    qa_references: list[str] | None = None,
    k: int = 10,
) -> dict:
    results = {
        "spearman_rho": spearman_rho(y_true, y_pred),
        "ndcg_at_k": ndcg_at_k(y_true, y_pred, k=k),
    }
    if qa_predictions and qa_references:
        results["em"] = float(np.mean([
            exact_match_score(p, [r]) for p, r in zip(qa_predictions, qa_references)
        ]))
        results["f1"] = float(np.mean([
            token_f1_score(p, [r]) for p, r in zip(qa_predictions, qa_references)
        ]))
    return results
