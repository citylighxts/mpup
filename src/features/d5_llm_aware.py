"""D5: LLM-Aware Signals — base/context perplexity and ΔH from logit probing."""
import numpy as np
from dataclasses import dataclass
from ..probing.logit_probe import LogitProbeResult


@dataclass
class D5Features:
    base_perplexity: float
    ctx_perplexity: float
    delta_h: float   # H_base - H_ctx


def extract_d5(probe_result: LogitProbeResult) -> D5Features:
    return D5Features(
        base_perplexity=probe_result.base_perplexity,
        ctx_perplexity=probe_result.ctx_perplexity,
        delta_h=probe_result.delta_h,
    )


def d5_to_array(feat: D5Features) -> np.ndarray:
    return np.array([feat.base_perplexity, feat.ctx_perplexity, feat.delta_h])
