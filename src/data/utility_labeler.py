"""Ground-truth utility labeling via source LLM inference.

u_i_true = delta_metric(with_passage_p_i) - metric(without_passage)

Sesuai Notion doc Section 4 — Training Predictor Engine:
  Ground-truth labeling: untuk setiap (q, p_i), hitung u_i_true dari
  perubahan performa jawaban pada LLM sumber.

Metrik: EM (Exact Match) atau token-F1 (atau keduanya, rata-rata).
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import metrics from evaluation module
from src.evaluation.metrics import (
    exact_match_score,
    token_f1_score,
    combined_score,
)


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


# ---------------------------------------------------------------------------
# Batched labeler — ported from the UtilityTransfer effort
# ---------------------------------------------------------------------------

NO_CONTEXT_PROMPT = (
    "Answer the question as briefly as possible, with just the answer span and no "
    "explanation.\n\nQuestion: {query}\nAnswer:"
)
WITH_CONTEXT_PROMPT = (
    "Answer the question as briefly as possible, with just the answer span and no "
    "explanation.\n\nPassage: {passage}\n\nQuestion: {query}\nAnswer:"
)


class BatchedUtilityLabeler:
    """Label a whole corpus at once, batching generation across queries and passages.

    `UtilityLabeler` answers one passage per forward pass, which leaves most of the GPU
    idle. Batching brought this to ~0.3s per labelled passage on an RTX 4060, making a
    35k-label run a matter of hours rather than days.

    The no-context answer is generated once per query and reused across that query's
    passages — it is the same quantity for all of them, and recomputing it would triple the
    generation cost for nothing.

    `metric_fn` defaults to `relaxed_combined_score`: with strict EM, an instruction-tuned
    model's sentence-form answers score as wrong and nearly every utility label collapses
    to zero.
    """

    def __init__(self, llm, metric_fn: Callable = None):
        from src.evaluation.metrics import relaxed_combined_score

        self.llm = llm
        self.metric_fn = metric_fn or relaxed_combined_score

    def label_corpus(self, items: list[dict]) -> list[dict]:
        """items: [{"query_id", "query", "gold_answers", "passages": [{"id", "text"}]}]

        Returns one row per (query, passage) with u_true and both component scores.
        """
        no_context = self.llm.generate(
            [NO_CONTEXT_PROMPT.format(query=item["query"]) for item in items]
        )
        baseline = {
            item["query_id"]: self.metric_fn(answer, item["gold_answers"])
            for item, answer in zip(items, no_context)
        }
        baseline_answer = {
            item["query_id"]: answer for item, answer in zip(items, no_context)
        }

        flat, owners = [], []
        for item in items:
            for passage in item["passages"]:
                flat.append(
                    WITH_CONTEXT_PROMPT.format(query=item["query"], passage=passage["text"])
                )
                owners.append((item, passage))

        with_context = self.llm.generate(flat)

        rows = []
        for (item, passage), answer in zip(owners, with_context):
            score_with = self.metric_fn(answer, item["gold_answers"])
            score_without = baseline[item["query_id"]]
            rows.append(
                {
                    "query_id": item["query_id"],
                    "doc_id": passage["id"],
                    "u_true": float(np.clip(score_with - score_without, -1.0, 1.0)),
                    "score_no_ctx": score_without,
                    "score_with_ctx": score_with,
                    "answer_no_ctx": baseline_answer[item["query_id"]],
                    "answer_with_ctx": answer,
                }
            )
        return rows
