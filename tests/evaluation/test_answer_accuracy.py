import numpy as np
import pytest

from experiments.evaluate_answer_accuracy import (
    build_prompt, generate_with_cache, plan_prompts, score_rows, summarise, top_m_indices,
)


def test_top_m_orders_by_score_and_keeps_candidate_order_on_ties():
    assert top_m_indices([0.1, 0.9, 0.5], 2) == [1, 2]
    assert top_m_indices([0.5, 0.5, 0.2], 2) == [0, 1]


def test_single_passage_prompt_matches_the_labelling_prompt():
    from src.data.utility_labeler import WITH_CONTEXT_PROMPT
    assert build_prompt("Q?", ["p"]) == WITH_CONTEXT_PROMPT.format(query="Q?", passage="p")


def test_prompt_without_passages_is_the_no_context_prompt():
    from src.data.utility_labeler import NO_CONTEXT_PROMPT
    assert build_prompt("Q?", []) == NO_CONTEXT_PROMPT.format(query="Q?")


def test_prompt_lists_several_passages_in_order():
    p = build_prompt("Q?", ["first", "second"])
    assert p.index("Passage 1: first") < p.index("Passage 2: second") < p.index("Question: Q?")


class OracleLLM:
    """Answers with the text after 'Passage:' when it contains the gold word, so a better ranking gives better accuracy."""
    def generate(self, prompts, max_new_tokens=None):
        return ["gold" if "gold" in p.split("Question:")[0] else "unknown" for p in prompts]


def _toy(n_queries=6):
    items, feats = {}, {"X": [], "y": [], "query_ids": [], "passage_ids": [], "dataset": []}
    for i in range(n_queries):
        qid = f"toy:{i}"
        items[qid] = {"query": "what?", "gold_answers": ["gold"],
                      "passages": [{"id": f"{qid}:a", "text": "noise"}, {"id": f"{qid}:b", "text": "gold answer here"}]}
        # dense likes the noise passage, MPUP likes the useful one
        for pid, dense, util in ((f"{qid}:a", 0.9, 0.0), (f"{qid}:b", 0.1, 1.0)):
            feats["X"].append([0.5, dense] + [0.0] * 12); feats["y"].append(util)
            feats["query_ids"].append(qid); feats["passage_ids"].append(pid); feats["dataset"].append("toy")
    feats = {k: np.array(v) for k, v in feats.items()}
    return items, feats


def test_better_ranking_gives_higher_accuracy_and_a_positive_paired_difference(tmp_path):
    items, feats = _toy(14)  # with n queries the smallest possible two-sided p is 2/2**n
    mpup = feats["y"].copy()  # a perfect predictor
    rows, prompts = plan_prompts(feats, mpup, items, [1], seed=0, limit=None)
    answers = generate_with_cache(OracleLLM(), prompts, tmp_path / "cache.json")
    score_rows(rows, answers)
    s = summarise(rows, [1], n_perm=2000, seed=0)
    assert s["accuracy"]["mpup@1"]["macro"] == pytest.approx(1.0)
    assert s["accuracy"]["dense@1"]["macro"] == pytest.approx(0.0)
    assert s["accuracy"]["oracle@1"]["macro"] == pytest.approx(1.0)
    assert s["accuracy"]["none@0"]["macro"] == pytest.approx(0.0)
    assert s["vs_dense"]["mpup@1"]["mean_diff"] == pytest.approx(1.0)
    assert s["vs_dense"]["mpup@1"]["p_permutation"] < 0.01


def test_generation_cache_is_reused(tmp_path):
    calls = []

    class Counting:
        def generate(self, prompts, max_new_tokens=None):
            calls.append(len(prompts)); return ["x"] * len(prompts)

    path = tmp_path / "c.json"
    generate_with_cache(Counting(), ["a", "b"], path)
    generate_with_cache(Counting(), ["a", "b", "c"], path)
    assert calls == [2, 1]
