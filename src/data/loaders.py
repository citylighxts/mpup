"""Dataset loaders for MPUP benchmarks.

Supported datasets (sesuai Notion doc Section 6.3):
  - galileo-ai/ragbench     (HuggingFace, config per subset; default hotpotqa)
  - natural-questions        (HuggingFace: google-research-datasets/natural_questions)
  - ms_marco                 (HuggingFace: microsoft/ms_marco, v2.1)
  - trivia_qa                (HuggingFace: mandarjoshi/trivia_qa, unfiltered)

Known limits (measured 2026-10, not fixed here): the Natural Questions loader turns every
long-answer candidate into a passage (median 66 per query, none marked positive), and the
TriviaQA loader uses whole Wikipedia/web pages as passages (median ~1,200 words).

Each loader yields dicts:
  {
    "query": str,
    "answers": list[str],          # gold answer strings
    "passages": [                  # candidate passages (retrieved or provided)
      {"id": str, "text": str, "is_positive": bool}
    ]
  }
"""
from __future__ import annotations
from typing import Iterator
from datasets import load_dataset


# ---------------------------------------------------------------------------
# RAGBench
# ---------------------------------------------------------------------------

def load_ragbench(
    split: str = "test", max_samples: int | None = None, subset: str = "hotpotqa"
) -> Iterator[dict]:
    """galileo-ai/ragbench — already has context passages per query.

    RAGBench is a multi-config dataset (one config per source dataset), so `subset` is
    required by the Hub. It has no gold-answer column: `answers` holds the dataset's
    reference `response`, a generated long-form answer, so EM/F1 against it is only a proxy.
    `is_positive` marks passages containing at least one annotated-relevant sentence.
    """
    ds = load_dataset("galileo-ai/ragbench", subset, split=split)
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        relevant = set(row.get("all_relevant_sentence_keys") or [])
        sentences = row.get("documents_sentences") or []
        passages = []
        for j, ctx in enumerate(row.get("documents", [])):
            keys = {s[0] for s in sentences[j]} if j < len(sentences) else set()
            passages.append(
                {"id": f"ragbench_{i}_{j}", "text": ctx, "is_positive": bool(keys & relevant)}
            )
        response = row["response"]
        yield {
            "query": row["question"],
            "answers": [response] if isinstance(response, str) else response,
            "passages": passages,
            "dataset": "ragbench",
        }


# ---------------------------------------------------------------------------
# Natural Questions
# ---------------------------------------------------------------------------

def load_natural_questions(split: str = "validation", max_samples: int | None = None) -> Iterator[dict]:
    """google/natural_questions — uses long-answer candidates as passages."""
    ds = load_dataset("google-research-datasets/natural_questions", split=split)
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        annotations = row.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        gold_texts: list[str] = []
        for sa in short_answers:
            if isinstance(sa, dict):
                gold_texts.extend(sa.get("text", []) or [])
        if not gold_texts:
            continue

        candidates = row.get("long_answer_candidates", {})
        tokens = row.get("document", {}).get("tokens", {})
        token_vals = tokens.get("token", [])
        is_html = tokens.get("is_html", [])
        plain = [t for t, h in zip(token_vals, is_html) if not h]

        passages = []
        starts = candidates.get("start_token", [])
        ends = candidates.get("end_token", [])
        for j, (s, e) in enumerate(zip(starts, ends)):
            text = " ".join(plain[s:e]).strip()
            if text:
                passages.append({"id": f"nq_{i}_{j}", "text": text, "is_positive": False})

        if not passages:
            continue

        yield {
            "query": row["question"]["text"],
            "answers": gold_texts,
            "passages": passages,
            "dataset": "natural_questions",
        }


# ---------------------------------------------------------------------------
# MS MARCO
# ---------------------------------------------------------------------------

def load_msmarco(split: str = "validation", max_samples: int | None = None) -> Iterator[dict]:
    """microsoft/ms_marco v2.1 — passages already associated per query."""
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split)
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue

        passages_raw = row.get("passages", {})
        texts = passages_raw.get("passage_text", [])
        is_selected = passages_raw.get("is_selected", [0] * len(texts))

        passages = [
            {
                "id": f"msmarco_{i}_{j}",
                "text": t,
                "is_positive": bool(sel),
            }
            for j, (t, sel) in enumerate(zip(texts, is_selected))
            if t.strip()
        ]

        if not passages:
            continue

        yield {
            "query": row["query"],
            "answers": answers,
            "passages": passages,
            "dataset": "msmarco",
        }


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

def load_triviaqa(split: str = "validation", max_samples: int | None = None) -> Iterator[dict]:
    """trivia_qa unfiltered — uses Wikipedia/web evidence as passages."""
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=split)
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        answers = row.get("answer", {}).get("aliases", [])
        if not answers:
            answers = [row["answer"]["value"]]

        passages = []
        for src_key in ("wikipedia", "web"):
            evidence = row.get("entity_pages", {}) if src_key == "wikipedia" else row.get("search_results", {})
            for j, text in enumerate(evidence.get("wiki_context", evidence.get("search_context", []))):
                if text and text.strip():
                    passages.append({"id": f"triviaqa_{i}_{src_key}_{j}", "text": text, "is_positive": False})

        if not passages:
            continue

        yield {
            "query": row["question"],
            "answers": answers,
            "passages": passages,
            "dataset": "triviaqa",
        }


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

LOADERS = {
    "ragbench": load_ragbench,
    "natural_questions": load_natural_questions,
    "msmarco": load_msmarco,
    "triviaqa": load_triviaqa,
}


def load_benchmark(name: str, split: str = "validation", max_samples: int | None = None) -> Iterator[dict]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS)}")
    return LOADERS[name](split=split, max_samples=max_samples)
