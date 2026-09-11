"""Quantized, batched local LLM backend for utility labelling and logit probing.

Ported from the UtilityTransfer effort, where it labelled ~35k passages across seven models
on a single 8GB RTX 4060.

Why this exists alongside `SourceLLMAnswerer`: that class loads in fp16 and answers one
passage per call. A 7B model in fp16 needs roughly 15GB, so it cannot run at all on an 8GB
card, and unbatched generation is several times slower than it needs to be. This backend
loads in 4-bit NF4 (~5GB for 7B) and batches both generation and scoring.

It also exposes token-level logprobs, which an HTTP-style server (Ollama and friends) does
not, and which the perplexity/entropy features require.

Measured throughput on an RTX 4060: ~0.3s per labelled passage including both generations.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LogprobResult:
    sum_logprob: float
    n_tokens: int

    @property
    def mean_logprob(self) -> float:
        return self.sum_logprob / self.n_tokens if self.n_tokens else 0.0

    @property
    def perplexity(self) -> float:
        return float(torch.exp(torch.tensor(-self.mean_logprob)))


class QuantizedLLM:
    def __init__(
        self,
        model_name_or_path: str,
        quantization: str = "nf4",
        dtype: str = "bfloat16",
        batch_size: int = 8,
        logprob_batch_size: int = 4,
        max_new_tokens: int = 32,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.logprob_batch_size = logprob_batch_size
        self.max_new_tokens = max_new_tokens

        torch_dtype = getattr(torch, dtype)
        quant_config = None
        if quantization == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            dtype=torch_dtype,  # transformers v5 renamed torch_dtype -> dtype
            device_map={"": 0} if device == "cuda" else device,
        )
        self.model.eval()

    def chat_format(self, user_message: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int | None = None) -> list[str]:
        """Greedy batched generation. Prompts are user messages; chat formatting applied here."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        self.tokenizer.padding_side = "left"  # decoder-only batching needs left padding
        outputs: list[str] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = [self.chat_format(p) for p in prompts[start : start + self.batch_size]]
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            for row, input_ids in zip(generated, encoded["input_ids"]):
                completion = row[len(input_ids) :]
                outputs.append(
                    self.tokenizer.decode(completion, skip_special_tokens=True).strip()
                )
        return outputs

    @torch.inference_mode()
    def sequence_logprobs(
        self, pairs: list[tuple[str, str]], add_special_tokens: bool = True
    ) -> list[LogprobResult]:
        """Logprob of each continuation given its context.

        Callers passing an already chat-formatted context must set
        `add_special_tokens=False`, or Llama-3 receives two BOS tokens.
        """
        results: list[LogprobResult] = []

        for start in range(0, len(pairs), self.logprob_batch_size):
            batch = pairs[start : start + self.logprob_batch_size]
            sequences, prompt_lengths = [], []
            for context, continuation in batch:
                context_ids = self.tokenizer(
                    context, add_special_tokens=add_special_tokens
                ).input_ids
                continuation_ids = self.tokenizer(
                    continuation, add_special_tokens=False
                ).input_ids
                sequences.append(context_ids + continuation_ids)
                prompt_lengths.append(len(context_ids))

            max_len = max(len(s) for s in sequences)
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
            for i, seq in enumerate(sequences):
                input_ids[i, : len(seq)] = torch.tensor(seq)
                attention_mask[i, : len(seq)] = 1
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Score only the continuation positions. Upcasting the whole logits tensor to
            # float32 costs batch x seq_len x vocab floats — gigabytes per call at a 152k
            # vocabulary, which exhausted both VRAM and host RAM before this was fixed.
            for i, seq in enumerate(sequences):
                start_idx = prompt_lengths[i] - 1
                end_idx = len(seq) - 1
                if end_idx <= start_idx:
                    results.append(LogprobResult(0.0, 0))
                    continue
                span_logits = logits[i, start_idx:end_idx].float()
                targets = input_ids[i, start_idx + 1 : end_idx + 1]
                token_logprobs = torch.log_softmax(span_logits, dim=-1).gather(
                    1, targets.unsqueeze(1)
                )
                results.append(
                    LogprobResult(float(token_logprobs.sum()), int(token_logprobs.numel()))
                )
                del span_logits, token_logprobs
            del logits
        return results

    def unload(self) -> None:
        """Free VRAM — necessary when cycling several 7B models through one 8GB GPU."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
