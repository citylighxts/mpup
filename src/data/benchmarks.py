"""The evaluation datasets of the multi-dataset experiment, as one uniform item format.

Every builder returns items

    {"dataset", "query_id", "query", "gold_answers", "pool", "passages"}

where `pool` says where candidate passages come from:

  * "provided"  — the dataset ships them (HotpotQA distractor, 2WikiMultiHopQA, MuSiQue);
                  `passages` is filled, every provided paragraph is a candidate.
  * "retrieved" — the dataset has questions and answers only; `passages` is empty and is
                  filled later by BM25 over the Wikipedia corpus (`src.retrieval.stream_bm25`).

Sources were checked against the originals (see README, "Datasets"): FlashRAG's copies of
NQ-open, TriviaQA, WebQuestions, PopQA, SQuAD, AmbigQA, Bamboogle and 2Wiki match the original
splits; its HotpotQA and MuSiQue do not (wrong paragraph set / paragraphs dropped), so those two
are read from the original Hub repos. Filters follow from the evaluation, not from taste:

  * yes/no answers are dropped (HotpotQA, 2Wiki): a model is right half the time without any
    passage, so utility is mostly noise;
  * PopQA questions whose text occurs more than once are dropped: "What is Will Smith's
    occupation?" has different gold answers for different people.

Passage text uses the Wikipedia corpus format, a quoted title line then the body, so passages
from every source look the same to the labeller and to the feature extractors.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

FLASHRAG = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main"
ENTITYQ_URL = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"

RETRIEVED_DATASETS = [
    "nq", "triviaqa", "webquestions", "entityquestions", "popqa", "squad", "ambigqa", "bamboogle",
]
PROVIDED_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
ALL_DATASETS = RETRIEVED_DATASETS + PROVIDED_DATASETS


# ------------------------------------------------------------------ helpers

def passage_text(title: str, body: str) -> str:
    return f'"{title}"\n{body}'


def is_yes_no(answers: list[str]) -> bool:
    return any(str(a).strip().lower() in ("yes", "no") for a in answers)


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample of n items, kept in their original order."""
    if len(items) <= n:
        return list(items)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(items), size=n, replace=False))
    return [items[i] for i in keep]


def fetch(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("://", 1)[1].replace("/", "__")
    path = cache_dir / name
    if path.exists():
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "mpup-pipeline"})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.rename(path)
    return path


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _item(dataset, idx, query, gold, pool, passages=None) -> dict:
    return {
        "dataset": dataset,
        "query_id": f"{dataset}:{idx}",
        "query": query,
        "gold_answers": list(gold),
        "pool": pool,
        "passages": passages or [],
    }


# -------------------------------------------------------------- pure parsers

def items_from_flashrag_rows(rows: list[dict], dataset: str, dedupe_questions: bool = False) -> list[dict]:
    if dedupe_questions:
        seen: dict[str, int] = {}
        for r in rows:
            seen[r["question"]] = seen.get(r["question"], 0) + 1
        rows = [r for r in rows if seen[r["question"]] == 1]
    out = []
    for i, r in enumerate(rows):
        gold = [str(a) for a in r.get("golden_answers", []) if str(a).strip()]
        if gold:
            out.append(_item(dataset, r.get("id", i), r["question"], gold, "retrieved"))
    return out


def items_from_hotpot_rows(rows: list[dict]) -> list[dict]:
    """hotpotqa/hotpot_qa, config `distractor`: context = {title: [...], sentences: [[...]]}."""
    out = []
    for r in rows:
        gold = [r["answer"]]
        if is_yes_no(gold):
            continue
        ctx = r["context"]
        passages = [
            {"id": f"hotpotqa:{r['id']}:p{j}", "text": passage_text(t, " ".join(s))}
            for j, (t, s) in enumerate(zip(ctx["title"], ctx["sentences"]))
        ]
        out.append(_item("hotpotqa", r["id"], r["question"], gold, "provided", passages))
    return out


def items_from_2wiki_rows(rows: list[dict]) -> list[dict]:
    """FlashRAG's 2WikiMultihopQA dev (identical to the original): metadata.context.{title,content}."""
    out = []
    for r in rows:
        gold = [str(a) for a in r["golden_answers"] if str(a).strip()]
        if not gold or is_yes_no(gold):
            continue
        ctx = r["metadata"]["context"]
        passages = [
            {"id": f"2wikimultihopqa:{r['id']}:p{j}",
             "text": passage_text(t, " ".join(c) if isinstance(c, list) else str(c))}
            for j, (t, c) in enumerate(zip(ctx["title"], ctx["content"]))
        ]
        out.append(_item("2wikimultihopqa", r["id"], r["question"], gold, "provided", passages))
    return out


def items_from_musique_rows(rows: list[dict]) -> list[dict]:
    """dgslibisey/MuSiQue: paragraphs = [{idx, title, paragraph_text, is_supporting}]."""
    out = []
    for r in rows:
        if not r.get("answerable", True):
            continue
        gold = [r["answer"], *r.get("answer_aliases", [])]
        gold = [str(a) for a in dict.fromkeys(gold) if str(a).strip()]
        passages = [
            {"id": f"musique:{r['id']}:p{p['idx']}", "text": passage_text(p["title"], p["paragraph_text"])}
            for p in r["paragraphs"]
        ]
        out.append(_item("musique", r["id"], r["question"], gold, "provided", passages))
    return out


def items_from_ragbench_rows(rows: list[dict], subset: str = "hotpotqa") -> list[dict]:
    """galileo-ai/ragbench rows: no gold answer exists, only sentence annotations.

    Each passage carries `proxy_label`, the fraction of its sentences that annotators marked as
    used (Eq. 5). That is a different quantity from the utility of Eq. (1), so these items are
    never pooled with the others; they are the *training* side of the proxy-transfer experiment.
    """
    out = []
    for i, r in enumerate(rows):
        docs = r["documents"]
        if len(docs) < 2:
            continue
        used = set(r["all_utilized_sentence_keys"])
        passages = []
        for j, (doc, sents) in enumerate(zip(docs, r["documents_sentences"])):
            keys = {s[0] for s in sents}
            passages.append({
                "id": f"ragbench-{subset}:{i}:p{j}",
                "text": doc,
                "proxy_label": len(keys & used) / max(len(sents), 1),
            })
        out.append(_item(f"ragbench-{subset}", i, r["question"], [], "provided", passages))
    return out


def build_ragbench(subset: str = "hotpotqa", split: str = "test") -> list[dict]:
    from datasets import load_dataset

    return items_from_ragbench_rows(list(load_dataset("galileo-ai/ragbench", subset, split=split)), subset)


def items_from_entityq_zip(data: bytes) -> list[dict]:
    """EntityQuestions test split: one JSON file per Wikidata relation, [{question, answers}]."""
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(n for n in z.namelist() if "/test/" in n and n.endswith(".json")):
            relation = Path(name).name.split(".")[0]
            for j, ex in enumerate(json.loads(z.read(name))):
                gold = [str(a) for a in ex["answers"] if str(a).strip()]
                if gold:
                    out.append(_item("entityquestions", f"{relation}-{j}", ex["question"], gold, "retrieved"))
    return out


# ------------------------------------------------------------------ builders

_FLASHRAG_FILES = {
    "nq": "nq/test.jsonl",
    "triviaqa": "triviaqa/test.jsonl",
    "webquestions": "web_questions/test.jsonl",
    "popqa": "popqa/test.jsonl",
    "squad": "squad/dev.jsonl",
    "ambigqa": "ambig_qa/dev.jsonl",
    "bamboogle": "bamboogle/test.jsonl",
}


def build_dataset(name: str, n: int = 100, seed: int = 0, cache_dir: str | Path = "data/raw/cache") -> list[dict]:
    """Return n items of one dataset (all of them if it has fewer), filters applied."""
    cache_dir = Path(cache_dir)
    if name in _FLASHRAG_FILES:
        rows = read_jsonl(fetch(f"{FLASHRAG}/{_FLASHRAG_FILES[name]}", cache_dir))
        items = items_from_flashrag_rows(rows, name, dedupe_questions=(name == "popqa"))
    elif name == "entityquestions":
        items = items_from_entityq_zip(fetch(ENTITYQ_URL, cache_dir).read_bytes())
    elif name == "2wikimultihopqa":
        items = items_from_2wiki_rows(read_jsonl(fetch(f"{FLASHRAG}/2wikimultihopqa/dev.jsonl", cache_dir)))
    elif name == "hotpotqa":
        from datasets import load_dataset

        items = items_from_hotpot_rows(list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")))
    elif name == "musique":
        from datasets import load_dataset

        items = items_from_musique_rows(list(load_dataset("dgslibisey/MuSiQue", split="validation")))
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {ALL_DATASETS}")
    # A per-dataset offset keeps the samples independent while staying reproducible.
    return sample_items(items, n, seed + ALL_DATASETS.index(name))
