"""Zero-shot LLM utility judgment: the baseline that asks the LLM directly whether a passage helps.

For every (query, passage) the LLM `M` receives a yes/no question about the usefulness of the passage for answering
the query, and the score is the logit of "Yes" minus the logit of "No" at the first answer position. It costs one
forward pass per passage, like the hidden-state pass of MPUP, but needs no training and no label. This stands for
the family of methods that probe the LLM directly to judge utility (no gold answer is ever shown to the model).
"""
from __future__ import annotations

import numpy as np
import torch

JUDGE_PROMPT = (
    "Question: {query}\n\nPassage: {passage}\n\n"
    "Is the passage useful for answering the question? Answer with Yes or No."
)


class UtilityJudge:
    def __init__(self, llm, batch_size: int = 8):
        self.llm, self.batch_size = llm, batch_size
        tok = llm.tokenizer
        self.yes, self.no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """(n_passages,) float32: logit(Yes) - logit(No) after the prompt, in the order of `passages`."""
        tok, model = self.llm.tokenizer, self.llm.model
        ids = [tok(self.llm.chat_format(JUDGE_PROMPT.format(query=query, passage=p)), add_special_tokens=False).input_ids
               for p in passages]
        out = np.zeros(len(ids), np.float32)
        order = np.argsort([len(i) for i in ids])  # similar lengths together keep the padding small
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            m = max(len(ids[i]) for i in idx)
            batch = torch.full((len(idx), m), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), m), dtype=torch.long)
            for r, i in enumerate(idx):
                batch[r, : len(ids[i])] = torch.tensor(ids[i])
                mask[r, : len(ids[i])] = 1  # padding on the right, so the real tokens keep their positions
            logits = model(input_ids=batch.to(model.device), attention_mask=mask.to(model.device)).logits
            for r, i in enumerate(idx):
                last = logits[r, len(ids[i]) - 1].float()
                out[i] = float(last[self.yes] - last[self.no])
        return out
