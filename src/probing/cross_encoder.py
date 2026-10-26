"""Cross-encoder relevance score, the extra D1 feature of MPUP+ (bge-reranker-v2-m3 in the larger study)."""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 128, max_length: int = 512, _loaded=None):
        if _loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            _loaded = (AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device),
                       AutoTokenizer.from_pretrained(model_name))
        self.model, self.tokenizer = _loaded
        self.batch_size, self.max_length = batch_size, max_length

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance score of every (query, passage), float32, in the order of `passages`."""
        out = np.zeros(len(passages), np.float32)
        order = np.argsort([len(p) for p in passages])  # similar lengths together keep the padding small
        device = getattr(self.model, "device", torch.device("cpu"))
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            enc = self.tokenizer([query] * len(idx), [passages[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out[idx] = self.model(**enc).logits.view(-1).float().cpu().numpy()
        return out
