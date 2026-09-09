"""D3: Semantic Utility — entity overlap (NER) and HyDE answerability."""
import numpy as np
from dataclasses import dataclass

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


@dataclass
class D3Features:
    entity_overlap: float   # fraction of query entities found in passage
    hyde_answerability: float  # cosine sim between HyDE hypothesis and passage


def _extract_entities(text: str) -> set[str]:
    if _nlp is None:
        return set(text.lower().split())
    doc = _nlp(text)
    return {ent.text.lower() for ent in doc.ents}


def extract_d3(
    query: str,
    passage: str,
    hyde_answerability: float = 0.0,
) -> D3Features:
    q_entities = _extract_entities(query)
    p_entities = _extract_entities(passage)
    overlap = len(q_entities & p_entities) / max(len(q_entities), 1)
    return D3Features(entity_overlap=overlap, hyde_answerability=hyde_answerability)


def d3_to_array(feat: D3Features) -> np.ndarray:
    return np.array([feat.entity_overlap, feat.hyde_answerability])
