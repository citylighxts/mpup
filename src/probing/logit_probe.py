"""Logit probing mechanics — D5 feature extraction.

Performs 1-pass forward pass on M_target to compute:
  - Base entropy  H_base  (query only)
  - Context entropy H_ctx  (passage + query)
  - Entropy delta  ΔH = H_base - H_ctx
"""
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LogitProbeResult:
    h_base: float       # Shannon entropy without context
    h_ctx: float        # Shannon entropy with context
    delta_h: float      # ΔH = H_base - H_ctx (D5 signal)
    base_perplexity: float
    ctx_perplexity: float


class LogitProber:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @torch.inference_mode()
    def _entropy_from_logits(self, logits: torch.Tensor) -> tuple[float, float]:
        """Return (entropy, perplexity) from next-token logits (shape: [vocab])."""
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        perplexity = torch.exp(-torch.sum(probs * log_probs)).item()
        return entropy, perplexity

    @torch.inference_mode()
    def probe(self, query: str, passage: str, max_length: int = 512) -> LogitProbeResult:
        # Base probe: query only
        base_prompt = f"Query: {query}"
        base_ids = self.tokenizer(
            base_prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(self.device)
        base_logits = self.model(base_ids).logits[0, -1]  # last token logits

        # Context probe: passage + query
        ctx_prompt = f"Context: {passage}\nQuery: {query}"
        ctx_ids = self.tokenizer(
            ctx_prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(self.device)
        ctx_logits = self.model(ctx_ids).logits[0, -1]

        h_base, ppl_base = self._entropy_from_logits(base_logits)
        h_ctx, ppl_ctx = self._entropy_from_logits(ctx_logits)

        return LogitProbeResult(
            h_base=h_base,
            h_ctx=h_ctx,
            delta_h=h_base - h_ctx,
            base_perplexity=ppl_base,
            ctx_perplexity=ppl_ctx,
        )

    def probe_batch(
        self, query: str, passages: list[str], max_length: int = 512
    ) -> list[LogitProbeResult]:
        return [self.probe(query, p, max_length) for p in passages]


class QuantizedProber:
    """LogitProber over a 4-bit NF4 model, batched, sharing one model with HyDE.

    `LogitProber` loads in fp16: a 7B model then needs ~15GB and will not fit an 8GB card,
    so the D5 features could not be computed on consumer NVIDIA hardware at all.

    It also recomputes `h_base` once per *passage*, although the base prompt contains only
    the query — for a 4-passage question that is 3 wasted forward passes. Here the base
    entropy is computed once per query and the context probes are batched.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", llm=None, batch_size: int = 4):
        if llm is None:
            from src.data.llm_backend import QuantizedLLM

            llm = QuantizedLLM(model_name, batch_size=batch_size)
        self.llm = llm
        self.batch_size = batch_size

    @torch.inference_mode()
    def _last_token_entropies(self, prompts: list[str], max_length: int = 512) -> list[tuple[float, float]]:
        tokenizer, model = self.llm.tokenizer, self.llm.model
        tokenizer.padding_side = "right"
        out: list[tuple[float, float]] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(model.device)
            logits = model(**encoded).logits
            # With right padding the final real token sits at attention_mask.sum() - 1.
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            for i, last in enumerate(lengths.tolist()):
                probs = F.softmax(logits[i, last].float(), dim=-1)
                entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
                out.append((entropy, float(np.exp(entropy))))
            del logits
        return out

    def probe_batch(
        self, query: str, passages: list[str], max_length: int = 512
    ) -> list[LogitProbeResult]:
        h_base, ppl_base = self._last_token_entropies([f"Query: {query}"], max_length)[0]
        ctx = self._last_token_entropies(
            [f"Context: {p}\nQuery: {query}" for p in passages], max_length
        )
        return [
            LogitProbeResult(
                h_base=h_base,
                h_ctx=h_ctx,
                delta_h=h_base - h_ctx,
                base_perplexity=ppl_base,
                ctx_perplexity=ppl_ctx,
            )
            for h_ctx, ppl_ctx in ctx
        ]

    def probe(self, query: str, passage: str, max_length: int = 512) -> LogitProbeResult:
        return self.probe_batch(query, [passage], max_length)[0]


class MLXProber:
    """LogitProber backed by an MLX model — no GPU needed, runs on Apple Silicon.

    Pass `llm` (an `MLXLLM`) to probe with the model that also writes the HyDE answer and the
    utility labels, instead of loading a second copy. As in `QuantizedProber`, the base entropy
    depends on the query alone, so `probe_batch` computes it once per query.
    """

    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit", llm=None):
        try:
            import mlx.core as mx
            self._mx = mx
        except ImportError as e:
            raise ImportError("pip install mlx mlx-lm") from e
        if llm is not None:
            self._model, self._tokenizer = llm.model, llm.tokenizer
        else:
            from mlx_lm import load as mlx_load
            self._model, self._tokenizer = mlx_load(model_name)

    def _tokens(self, prompt: str, max_length: int) -> list[int]:
        tokens = self._tokenizer.encode(prompt)
        if len(tokens) > max_length:
            tokens = tokens[:1] + tokens[-(max_length - 1):]  # keep BOS and the end, where the query is
        return tokens

    def _entropy(self, prompt: str, max_length: int = 512) -> tuple[float, float]:
        """(entropy in nats, perplexity = exp(entropy)) of the next-token distribution."""
        mx = self._mx
        x = mx.array([self._tokens(prompt, max_length)])
        logits = self._model(x)
        last = logits[0, -1, :].astype(mx.float32)
        probs = mx.softmax(last)
        h = float(-mx.sum(probs * mx.log(probs + 1e-10)).item())
        ppl = float(mx.exp(mx.array(h)).item())
        return h, ppl

    def _result(self, base: tuple[float, float], ctx: tuple[float, float]) -> LogitProbeResult:
        return LogitProbeResult(
            h_base=base[0], h_ctx=ctx[0], delta_h=base[0] - ctx[0],
            base_perplexity=base[1], ctx_perplexity=ctx[1],
        )

    def probe(self, query: str, passage: str, max_length: int = 512) -> LogitProbeResult:
        return self._result(
            self._entropy(f"Query: {query}", max_length),
            self._entropy(f"Context: {passage}\nQuery: {query}", max_length),
        )

    def probe_batch(self, query: str, passages: list[str], max_length: int = 512) -> list[LogitProbeResult]:
        if not passages:
            return []
        base = self._entropy(f"Query: {query}", max_length)
        return [
            self._result(base, self._entropy(f"Context: {p}\nQuery: {query}", max_length))
            for p in passages
        ]


    def hidden_batch(
        self, query: str, passages: list[str], layers: tuple[int, ...] = (14, -1), max_length: int = 512
    ) -> np.ndarray:
        """Last-token hidden states of the probing prompt, shape (len(passages), len(layers), hidden_size).

        Layer `i >= 0` is the residual stream after transformer block `i`; `-1` is the final, post-norm
        state (what `model.model(x)` returns). The prompt and its truncation are those of `probe_batch`.
        """
        from mlx_lm.models.base import create_attention_mask

        mx = self._mx
        inner = self._model.model
        wanted = set(layers)
        out = np.zeros((len(passages), len(layers), inner.args.hidden_size), dtype=np.float32)
        for row, p in enumerate(passages):
            x = mx.array([self._tokens(f"Context: {p}\nQuery: {query}", max_length)])
            h = inner.embed_tokens(x)
            mask = create_attention_mask(h, None)
            states = {}
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask, None)
                if i in wanted:
                    states[i] = h[0, -1]
            if -1 in wanted:
                states[-1] = inner.norm(h)[0, -1]
            for col, layer_index in enumerate(layers):
                out[row, col] = np.array(states[layer_index].astype(mx.float32))
        return out


class MockProber:
    """Deterministic mock for testing without GPU."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    def probe(self, query: str, passage: str, max_length: int = 512) -> LogitProbeResult:
        rng = np.random.default_rng(hash(query + passage) % (2**31))
        h_base = float(rng.uniform(2.5, 4.0))
        h_ctx = float(rng.uniform(0.5, 5.0))
        delta_h = h_base - h_ctx
        return LogitProbeResult(
            h_base=h_base,
            h_ctx=h_ctx,
            delta_h=delta_h,
            base_perplexity=float(np.exp(h_base)),
            ctx_perplexity=float(np.exp(h_ctx)),
        )

    def probe_batch(self, query: str, passages: list[str], max_length: int = 512) -> list[LogitProbeResult]:
        return [self.probe(query, p, max_length) for p in passages]
