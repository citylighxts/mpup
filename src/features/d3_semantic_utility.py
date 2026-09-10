"""D3: Semantic Utility — entity overlap (NER) and HyDE answerability."""
import re
import numpy as np
from dataclasses import dataclass

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "and", "or", "this", "that", "these", "those", "it", "as",
    "by", "with", "from", "here", "there", "which", "what", "who", "where", "when",
}


@dataclass
class D3Features:
    entity_overlap: float       # fraction of query entities found in passage
    hyde_answerability: float   # lexical overlap between HyDE pseudo-answer and passage


def _extract_entities(text: str) -> set[str]:
    if _nlp is None:
        import string
        words = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
        return set(words)
    doc = _nlp(text)
    return {ent.text.lower() for ent in doc.ents}


def _content_tokens(text: str) -> set[str]:
    toks = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {t for t in toks if t not in _STOP and len(t) > 1}


def _answerability(hyde_answer: str, passage: str) -> float:
    """Fraction of the pseudo-answer's content tokens present in the passage."""
    a = _content_tokens(hyde_answer)
    if not a:
        return 0.0
    p = _content_tokens(passage)
    return len(a & p) / len(a)


def extract_d3(
    query: str,
    passage: str,
    hyde_answer: str | None = None,
) -> D3Features:
    q_entities = _extract_entities(query)
    p_entities = _extract_entities(passage)
    overlap = len(q_entities & p_entities) / max(len(q_entities), 1)
    answerability = _answerability(hyde_answer, passage) if hyde_answer else 0.0
    return D3Features(entity_overlap=overlap, hyde_answerability=answerability)


def d3_to_array(feat: D3Features) -> np.ndarray:
    return np.array([feat.entity_overlap, feat.hyde_answerability])
