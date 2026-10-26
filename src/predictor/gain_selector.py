"""A passage selector trained on utilities, in the manner of GainRAG (Jiang et al., ACL 2025).

GainRAG fine-tunes a cross-encoder reranker (bge-reranker-base) so that, within each query, its scores follow the
"gain" of every passage for the LLM, using knowledge distillation (a listwise KL between the softmax of the
scores and the softmax of the gain signal). This module does the same with the Eq. (1) utility of this paper as the
gain signal, on the same development questions MPUP is fitted on, so the two methods see the same supervision. It is a
re-implementation of the training idea under our protocol, not the original code; the original builds its gain
signal with its own LLM pipeline. Score = the cross-encoder logit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "BAAI/bge-reranker-base"


def kd_loss(scores: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Mean over queries of KL(softmax(targets / T) || softmax(scores)); both are (queries, passages).

    The softmax makes the loss blind to a constant shift of the scores of a query, as the order is what counts.
    """
    p = F.softmax(targets / temperature, dim=-1)
    return F.kl_div(F.log_softmax(scores, dim=-1), p, reduction="batchmean")


class GainSelector:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, max_length: int = 512, _loaded=None):
        if _loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            _loaded = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device),
                       AutoTokenizer.from_pretrained(model_name))
        self.model, self.tokenizer = _loaded
        self.batch_size, self.max_length = batch_size, max_length

    def _logits(self, query: str, passages: list[str]) -> torch.Tensor:
        enc = self.tokenizer([query] * len(passages), passages, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        device = getattr(self.model, "device", torch.device("cpu"))
        return self.model(**{k: v.to(device) for k, v in enc.items()}).logits.view(-1)

    def fit(self, queries: list[str], groups: list[list[str]], targets: list[np.ndarray], epochs: int = 2,
            lr: float = 6e-5, temperature: float = 0.2, queries_per_step: int = 2, seed: int = 0) -> "GainSelector":
        """One group = the candidates of one query; `targets[i]` are their utilities (Eq. 1)."""
        rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            order = rng.permutation(len(queries))
            for s in range(0, len(order), queries_per_step):
                loss = 0.0
                batch = order[s : s + queries_per_step]
                for i in batch:
                    if len(groups[i]) < 2 or np.ptp(targets[i]) < 1e-12:
                        continue  # nothing to rank
                    t = torch.as_tensor(np.asarray(targets[i], dtype=np.float32), device=getattr(self.model, "device", None))
                    loss = loss + kd_loss(self._logits(queries[i], groups[i])[None, :], t[None, :], temperature)
                if torch.is_tensor(loss):
                    opt.zero_grad()
                    (loss / len(batch)).backward()
                    opt.step()
        self.model.eval()
        return self

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """(n_passages,) float32 cross-encoder logits in the order of `passages`."""
        out = np.zeros(len(passages), np.float32)
        order = np.argsort([len(p) for p in passages])
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            out[idx] = self._logits(query, [passages[i] for i in idx]).float().cpu().numpy()
        return out
