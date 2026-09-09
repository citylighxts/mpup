"""Ground-truth utility labeling via source LLM inference.

u_i_true = delta_metric(with_passage_p_i) - metric(without_passage)

Sesuai Notion doc Section 4 — Training Predictor Engine:
  Ground-truth labeling: untuk setiap (q, p_i), hitung u_i_true dari
  perubahan performa jawaban pada LLM sumber.

Metrik: EM (Exact Match) atau token-F1 (atau keduanya, rata-rata).
"""
from __future__ import annotations
import re
import string
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# QA metrics (token-level)
# ---------------------------------------------------------------------------

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
    """Average of EM and F1 — balances precision with partial credit."""
    return 0.5 * exact_match_score(prediction, gold_answers) + \
           0.5 * token_f1_score(prediction, gold_answers)


# ---------------------------------------------------------------------------
# Source LLM answer generator
# ---------------------------------------------------------------------------

class SourceLLMAnswerer:
    """Generates answers using a source LLM for utility labeling."""

    def __init__(self, model_name_or_path: str, device: str = "auto", max_new_tokens: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def answer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def answer_no_context(self, query: str) -> str:
        prompt = f"Answer the following question in one sentence.\nQuestion: {query}\nAnswer:"
        return self.answer(prompt)

    def answer_with_context(self, query: str, passage: str) -> str:
        prompt = (
            f"Use the following context to answer the question in one sentence.\n"
            f"Context: {passage}\n"
            f"Question: {query}\nAnswer:"
        )
        return self.answer(prompt)


# ---------------------------------------------------------------------------
# Utility labeler
# ---------------------------------------------------------------------------

@dataclass
class LabeledPassage:
    doc_id: str
    text: str
    u_true: float          # ground-truth utility in [-1, 1] (roughly)
    score_no_ctx: float    # combined_score without this passage
    score_with_ctx: float  # combined_score with this passage


class UtilityLabeler:
    """
    Computes u_i_true = combined_score(with p_i) - combined_score(without p_i).

    Positive  → passage helps the source LLM answer correctly.
    Near-zero → redundant / neutral.
    Negative  → passage confuses / misleads the source LLM.
    """

    def __init__(self, answerer: SourceLLMAnswerer, metric_fn: Callable = combined_score):
        self.answerer = answerer
        self.metric_fn = metric_fn

    def label_query(self, query: str, gold_answers: list[str], passages: list[dict]) -> list[LabeledPassage]:
        """
        passages: list of {"id": str, "text": str}
        Returns list of LabeledPassage in the same order.
        """
        # baseline score — no context at all
        pred_no_ctx = self.answerer.answer_no_context(query)
        score_no_ctx = self.metric_fn(pred_no_ctx, gold_answers)

        labeled = []
        for p in passages:
            pred_with = self.answerer.answer_with_context(query, p["text"])
            score_with = self.metric_fn(pred_with, gold_answers)
            u_true = float(np.clip(score_with - score_no_ctx, -1.0, 1.0))
            labeled.append(LabeledPassage(
                doc_id=p["id"],
                text=p["text"],
                u_true=u_true,
                score_no_ctx=score_no_ctx,
                score_with_ctx=score_with,
            ))
        return labeled
