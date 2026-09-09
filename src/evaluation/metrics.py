"""Evaluation metrics: Spearman ρ, NDCG@k, MRR, EM, F1."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    return float(ndcg_score(y_true[np.newaxis, :], y_pred[np.newaxis, :], k=k))


def mrr(relevant_ranks: list[int]) -> float:
    """Mean Reciprocal Rank. relevant_ranks: 1-indexed rank of first relevant doc per query."""
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
        results["em"] = exact_match(qa_predictions, qa_references)
        results["f1"] = float(np.mean([
            token_f1(p, r) for p, r in zip(qa_predictions, qa_references)
        ]))
    return results
