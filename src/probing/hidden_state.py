"""The hidden state of `M` that the MPUP+ probe reads.

For every candidate, `M` receives the prompt that defines the true utility (Eq. 1, with the passage, chat
formatted) and the hidden state of one layer at the last prompt token is kept. Gold answers are never part of
the input. The same state was cached for the larger study by `experiments.extract_extra`.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.utility_labeler import WITH_CONTEXT_PROMPT

LAYER = 21  # of 28 for Qwen2.5-7B-Instruct, as in the larger study


class HiddenStateExtractor:
    def __init__(self, llm, layer: int = LAYER, batch_size: int = 8):
        self.llm, self.layer, self.batch_size = llm, layer, batch_size

    @torch.inference_mode()
    def hidden_batch(self, query: str, passages: list[str]) -> np.ndarray:
        """(n_passages, hidden_size) float32, in the order of `passages`."""
        tok, model = self.llm.tokenizer, self.llm.model
        ids = [tok(self.llm.chat_format(WITH_CONTEXT_PROMPT.format(query=query, passage=p)),
                   add_special_tokens=False).input_ids for p in passages]
        out: list[np.ndarray | None] = [None] * len(ids)
        order = np.argsort([len(i) for i in ids])  # similar lengths together keep the padding small
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            m = max(len(ids[i]) for i in idx)
            batch = torch.full((len(idx), m), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), m), dtype=torch.long)
            for r, i in enumerate(idx):
                batch[r, : len(ids[i])] = torch.tensor(ids[i])
                mask[r, : len(ids[i])] = 1  # padding on the right, so the real tokens keep their positions
            states = model(input_ids=batch.to(model.device), attention_mask=mask.to(model.device),
                           output_hidden_states=True).hidden_states[self.layer]
            for r, i in enumerate(idx):
                out[i] = states[r, len(ids[i]) - 1].float().cpu().numpy()
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)
