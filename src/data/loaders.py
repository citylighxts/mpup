"""Dataset loaders for MPUP benchmarks.

Supported datasets (sesuai Notion doc Section 6.3):
  - galileo-ai/ragbench     (HuggingFace)
  - natural-questions        (HuggingFace: google/natural_questions)
  - ms_marco                 (HuggingFace: microsoft/ms_marco, v2.1)
  - trivia_qa                (HuggingFace: trivia_qa, unfiltered)

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

def load_ragbench(split: str = "test", max_samples: int | None = None) -> Iterator[dict]:
    """galileo-ai/ragbench — already has context passages per query."""
    ds = load_dataset("galileo-ai/ragbench", split=split)
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        passages = [
            {"id": f"ragbench_{i}_{j}", "text": ctx, "is_positive": True}
            for j, ctx in enumerate(row.get("documents", []))
        ]
        yield {
            "query": row["question"],
            "answers": [row["answer"]] if isinstance(row["answer"], str) else row["answer"],
            "passages": passages,
            "dataset": "ragbench",
        }


# ---------------------------------------------------------------------------
# Natural Questions
# ---------------------------------------------------------------------------

def load_natural_questions(split: str = "validation", max_samples: int | None = None) -> Iterator[dict]:
    """google/natural_questions — uses long-answer candidates as passages."""
    ds = load_dataset("google/natural_questions", split=split, trust_remote_code=True)
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
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split, trust_remote_code=True)
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
    ds = load_dataset("trivia_qa", "unfiltered", split=split, trust_remote_code=True)
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
