"""HyDE — generate a hypothetical answer sentence for a query.

The pseudo-answer is a short declarative claim that a correct document would
support. It is consumed by two feature groups:
  - D3 (semantic utility): lexical overlap(pseudo-answer, passage) = answerability
  - D4 (faithfulness): NLI premise=passage, hypothesis=pseudo-answer

Generating it once per query (not per passage) keeps the extra LLM cost to one
short decode per query.
"""
from __future__ import annotations

_QUESTION_PREFIXES = (
    "what is", "what are", "what was", "what were", "who is", "who was",
    "who are", "where is", "where was", "where are", "when is", "when was",
    "when did", "how many", "how much", "how did", "how does", "which",
    "why is", "why did", "why does",
)


def query_to_statement(query: str) -> str:
    """Rule-based fallback: turn a question into a declarative stub (no LLM)."""
    q = query.strip().rstrip("?").strip()
    lowered = q.lower()
    for prefix in _QUESTION_PREFIXES:
        if lowered.startswith(prefix):
            rest = q[len(prefix):].strip()
            return f"{rest} is described in this text." if rest else f"This text answers: {q}."
    return f"This text answers the question: {q}."


class HyDEGenerator:
    """Generates a one-sentence hypothetical answer using an MLX LLM (Apple Silicon)."""

    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        try:
            from mlx_lm import load as mlx_load, generate as mlx_generate
        except ImportError as e:
            raise ImportError("pip install mlx mlx-lm") from e
        self._generate = mlx_generate
        self._model, self._tokenizer = mlx_load(model_name)

    def generate(self, query: str, max_tokens: int = 48) -> str:
        prompt = (
            "Write exactly one short factual sentence that directly answers the "
            "question. Do not add explanation.\n"
            f"Question: {query}\nAnswer:"
        )
        try:
            text = self._generate(
                self._model, self._tokenizer, prompt=prompt,
                max_tokens=max_tokens, verbose=False,
            )
        except TypeError:
            text = self._generate(self._model, self._tokenizer, prompt=prompt, verbose=False)
        first_line = text.strip().split("\n")[0].strip()
        return first_line or query_to_statement(query)


class MockHyDEGenerator:
    """Deterministic fallback — rule-based statement, no model needed."""

    def generate(self, query: str, max_tokens: int = 48) -> str:
        return query_to_statement(query)
