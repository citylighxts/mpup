"""Batched greedy generation on Apple Silicon (MLX), with the interface `BatchedUtilityLabeler` expects.

`QuantizedLLM` (src/data/llm_backend.py) does this for NVIDIA GPUs through bitsandbytes; this is
its Apple Silicon counterpart, so the true utility of Eq. (1) can be labelled on a Mac. Both expose

    generate(prompts: list[str], max_new_tokens=None) -> list[str]

where each prompt is a *user message*: chat formatting is applied here, once, and the tokenizer
is called with `add_special_tokens=False` because the chat template already writes the BOS token
(passing it again gives Llama-3 two BOS tokens).
"""
from __future__ import annotations


class MLXLLM:
    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        batch_size: int = 8,
        max_new_tokens: int = 32,
        _loaded=None,
        _batch_generate=None,
    ) -> None:
        if _loaded is None:
            from mlx_lm import load

            _loaded = load(model_name)
        if _batch_generate is None:
            from mlx_lm import batch_generate as _batch_generate
        self.model, self.tokenizer = _loaded
        self._batch_generate = _batch_generate
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def chat_format(self, user_message: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
        )

    def _encode(self, user_message: str) -> list[int]:
        return self.tokenizer.encode(self.chat_format(user_message), add_special_tokens=False)

    def generate(self, prompts: list[str], max_new_tokens: int | None = None) -> list[str]:
        """Greedy answers, returned in the order of `prompts`.

        Prompts are processed in length order within each batch so padding stays small, then put
        back in the caller's order.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        encoded = [self._encode(p) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
        out: list[str] = [""] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            response = self._batch_generate(
                self.model, self.tokenizer, [encoded[i] for i in idx], max_tokens=max_new_tokens
            )
            for i, text in zip(idx, response.texts):
                out[i] = text.strip()
        return out

    def answer_logprobs(self, prompts: list[str], answers: list[str]) -> list[float]:
        """Mean token log-probability of each `answer` as the reply to the matching prompt (teacher forcing)."""
        import mlx.core as mx

        out = []
        for prompt, answer in zip(prompts, answers):
            context = self._encode(prompt)
            reply = self.tokenizer.encode(answer, add_special_tokens=False)
            logits = self.model(mx.array([context + reply]))[0, len(context) - 1 : -1].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            picked = mx.take_along_axis(logp, mx.array(reply)[:, None], axis=-1)
            out.append(float(mx.mean(picked).item()))
        return out
