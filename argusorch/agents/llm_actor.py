from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from argusorch.env import AgentAction, AgentObservation
from argusorch.agents.types import PolicyEval


class LLMActor:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: nn.Module,
        generation_config: Dict[str, object],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def act(self, obs: AgentObservation) -> AgentAction:
        """Generate a completion and compute its log-prob in a single forward pass.

        Previously, act() called generate() then compute_logprob() separately —
        two full forward passes through the LLM. Now we extract log-probs directly
        from the generation scores (output_scores=True), halving the VRAM peak
        and the wall-clock time of the rollout phase.
        """
        text, logprob = self._generate_with_logprob(obs.prompt)
        return AgentAction(text=text, logprob=logprob)

    def evaluate_action(self, obs: AgentObservation, act: AgentAction) -> PolicyEval:
        """Re-compute log-prob with gradient tracking (PPO update pass)."""
        logprobs = self.compute_logprob(obs.prompt, act.text, no_grad=False)
        return PolicyEval(logprobs=logprobs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_with_logprob(self, prompt: str) -> Tuple[str, float]:
        """Run model.generate() once and extract both the text and its log-prob."""
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length: int = inputs["input_ids"].shape[1]

        gen_kwargs = (
            self.generation_config if isinstance(self.generation_config, dict) else {}
        )

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

        completion_ids = outputs.sequences[0][input_length:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Compute token-level log-probs from generation scores.
        # outputs.scores is a tuple of (vocab_size,) tensors, one per generated token.
        logprob = 0.0
        if outputs.scores and len(completion_ids) > 0:
            # Stack → [T, 1, vocab_size], then gather generated token ids
            scores = torch.stack(outputs.scores, dim=0)  # [T, 1, V]
            log_probs = F.log_softmax(scores.float(), dim=-1)
            n = min(len(completion_ids), scores.shape[0])
            token_log_probs = log_probs[:n, 0, completion_ids[:n]]
            logprob = token_log_probs.sum().item()
            del scores, log_probs, token_log_probs

        return completion_text, logprob

    def generate(self, prompt: str) -> str:
        """Kept for backwards compatibility; prefer _generate_with_logprob."""
        text, _ = self._generate_with_logprob(prompt)
        return text

    def compute_logprob(
        self, prompt: str, completion: str, no_grad: bool = True
    ) -> torch.Tensor:
        """Full forward pass to get a differentiable log-prob (used in PPO update)."""
        device = self.model.device

        full_text = prompt + completion
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(device)

        prompt_len: int = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"].shape[1]

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                outputs = self.model(**full_inputs, use_cache=False)
                logits = outputs.logits

                shift_logits = logits[0, prompt_len - 1 : -1, :].contiguous()
                shift_labels = full_inputs["input_ids"][0, prompt_len:].contiguous()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, 1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                result = token_log_probs.sum()

                del outputs, logits, shift_logits, log_probs

        return result
