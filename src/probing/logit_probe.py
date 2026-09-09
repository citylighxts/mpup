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
