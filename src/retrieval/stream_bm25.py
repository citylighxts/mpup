"""Exact BM25 over a corpus too large to index, by streaming it twice.

The Wikipedia corpus used here (`wiki18_100w`, 21,015,324 passages of 100 words) is a 14 GB
JSONL file inside a 5 GB zip. Building an inverted index for it needs tens of GB of RAM; but a
benchmark only has a few hundred queries, so their terms fit in a small dictionary. Two passes:

  1. `collect_stats` — N, average passage length and the document frequency of every query term;
  2. `retrieve`      — score each passage against only the queries that share a term with it
                       and keep a top-k heap per query.

The result is the same ranking an index would give (verified against a brute-force scorer in
the tests), with no index to build or store.

Scoring is Okapi BM25 with the Lucene IDF, log(1 + (N - df + 0.5) / (df + 0.5)), which is never
negative: the classic IDF goes negative for terms in more than half the documents and then
*penalises* a passage for containing them. Stopwords are removed, there is no stemming, and a
passage is its title plus its text. Passage length is counted after stopword removal, the same
way in both passes.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import struct
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being below "
    "between both but by can could did do does doing down during each few for from further had has have "
    "having he her here hers him his how i if in into is it its itself just me more most my no nor not now "
    "of off on once only or other our out over own same she should so some such than that the their them "
    "then there these they this those through to too under until up very was we were what when where which "
    "while who whom why will with would you your".split()
)
_WORD = re.compile(r"\w+")

K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def query_terms(query: str) -> list[str]:
    """Unique query terms, in order of first appearance."""
    return list(dict.fromkeys(tokenize(query)))


def bm25_idf(df: int, n_docs: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass
class CorpusStats:
    n_docs: int = 0
    avgdl: float = 1.0
    df: dict[str, int] = field(default_factory=dict)

    def idf(self, term: str) -> float:
        return bm25_idf(self.df.get(term, 0), self.n_docs)


def score_tokens(terms: Iterable[str], tokens: list[str], stats: CorpusStats, k1: float = K1, b: float = B) -> float:
    """BM25 of one tokenised passage against a set of query terms."""
    tf = Counter(tokens)
    norm = k1 * (1.0 - b + b * len(tokens) / stats.avgdl)
    return sum(
        stats.idf(t) * tf[t] * (k1 + 1.0) / (tf[t] + norm) for t in terms if tf.get(t)
    )


def score_text(terms: Iterable[str], text: str, stats: CorpusStats) -> float:
    return score_tokens(terms, tokenize(text), stats)


# ------------------------------------------------------------------ reading

def parse_line(raw: bytes | str) -> tuple[str, str, str]:
    """One JSONL line -> (id, title, text). `contents` is '"Title"\\ntext'."""
    obj = json.loads(raw)
    contents = obj["contents"]
    head, _, body = contents.partition("\n")
    return str(obj["id"]), head.strip().strip('"'), body


def iter_zip_lines(path: str, chunk: int = 1 << 22) -> Iterator[bytes]:
    """Lines of the single JSONL member of a zip, streamed without extracting it.

    The corpus zip uses Zip64 (sizes 0xFFFFFFFF in the local header), so the member is read as
    a raw deflate stream from the first local header rather than through `zipfile`.
    """
    with open(path, "rb") as f:
        header = f.read(30)
        sig, _, _, method, *_rest, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if sig != 0x04034B50 or method != 8:
            raise ValueError("expected a deflate-compressed zip member")
        f.read(name_len + extra_len)
        d = zlib.decompressobj(-15)
        rest = b""
        while True:
            data = f.read(chunk)
            if not data:
                break
            rest += d.decompress(data)
            *lines, rest = rest.split(b"\n")
            yield from (ln for ln in lines if ln.strip())
            if d.eof:
                break
        rest += d.flush()
        if rest.strip():
            yield rest


# ------------------------------------------------------------------- workers

_G: dict = {}


def _init(term_to_q, idf, avgdl, k):
    _G.update(term_to_q=term_to_q, idf=idf, avgdl=avgdl, k=k)


def _stats_batch(lines: list[bytes]) -> tuple[int, int, Counter]:
    wanted = _G["terms"]
    n = total = 0
    df: Counter = Counter()
    for raw in lines:
        _, title, body = parse_line(raw)
        toks = tokenize(title + " " + body)
        n += 1
        total += len(toks)
        for t in wanted.intersection(toks):
            df[t] += 1
    return n, total, df


def _init_stats(terms):
    _G["terms"] = frozenset(terms)


def _retrieve_batch(args: tuple[int, list[bytes]]) -> dict[int, list[tuple]]:
    base, lines = args
    term_to_q, idf, avgdl, k = _G["term_to_q"], _G["idf"], _G["avgdl"], _G["k"]
    heaps: dict[int, list] = defaultdict(list)
    for offset, raw in enumerate(lines):
        pid, title, body = parse_line(raw)
        toks = tokenize(title + " " + body)
        hits = Counter(t for t in toks if t in term_to_q)
        if not hits:
            continue
        norm = K1 * (1.0 - B + B * len(toks) / avgdl)
        scores: dict[int, float] = defaultdict(float)
        for t, tf in hits.items():
            w = idf[t] * tf * (K1 + 1.0) / (tf + norm)
            for qi in term_to_q[t]:
                scores[qi] += w
        for qi, s in scores.items():
            h = heaps[qi]
            item = (s, -(base + offset), pid, title, body)  # ties go to the earlier passage
            if len(h) < k:
                heapq.heappush(h, item)
            elif item[:2] > h[0][:2]:
                heapq.heapreplace(h, item)
    return dict(heaps)


def _batches(lines: Iterator[bytes], size: int) -> Iterator[tuple[int, list[bytes]]]:
    base, buf = 0, []
    for raw in lines:
        buf.append(raw)
        if len(buf) == size:
            yield base, buf
            base, buf = base + len(buf), []
    if buf:
        yield base, buf


def _run(pool_fn, tasks: Iterator, workers: int, initializer, initargs, window: int | None = None):
    """Map with a bounded number of batches in flight, so a 14 GB stream never sits in RAM."""
    if workers <= 1:
        initializer(*initargs)
        for t in tasks:
            yield pool_fn(t)
        return
    window = window or workers * 3
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        pending: deque = deque()
        for t in tasks:
            pending.append(ex.submit(pool_fn, t))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


# ------------------------------------------------------------------ the passes

def collect_stats(
    open_lines: Callable[[], Iterator[bytes]], terms: Iterable[str], workers: int = 1, batch: int = 20000
) -> CorpusStats:
    terms = set(terms)
    n = total = 0
    df: Counter = Counter()
    stats_tasks = (lines for _, lines in _batches(open_lines(), batch))
    for bn, btotal, bdf in _run(_stats_batch, stats_tasks, workers, _init_stats, (terms,)):
        n += bn
        total += btotal
        df.update(bdf)
    return CorpusStats(n_docs=n, avgdl=total / max(n, 1), df=dict(df))


def retrieve(
    open_lines: Callable[[], Iterator[bytes]],
    queries: dict[str, str],
    stats: CorpusStats,
    k: int = 5,
    workers: int = 1,
    batch: int = 20000,
) -> dict[str, list[dict]]:
    """Top-k passages per query id. Queries with no usable term get an empty list."""
    qids = list(queries)
    term_to_q: dict[str, list[int]] = defaultdict(list)
    for qi, qid in enumerate(qids):
        for t in query_terms(queries[qid]):
            term_to_q[t].append(qi)
    idf = {t: stats.idf(t) for t in term_to_q}
    merged: dict[int, list] = defaultdict(list)
    for part in _run(_retrieve_batch, _batches(open_lines(), batch), workers, _init, (dict(term_to_q), idf, stats.avgdl, k)):
        for qi, items in part.items():
            h = merged[qi]
            for item in items:
                if len(h) < k:
                    heapq.heappush(h, item)
                elif item[:2] > h[0][:2]:
                    heapq.heapreplace(h, item)
    out: dict[str, list[dict]] = {}
    for qi, qid in enumerate(qids):
        ranked = sorted(merged.get(qi, []), key=lambda it: (-it[0], -it[1]))
        out[qid] = [{"id": pid, "title": title, "text": body, "bm25": score} for score, _, pid, title, body in ranked]
    return out
