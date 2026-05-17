from typing import Any, Dict

import torch
import torch.nn as nn

from argusorch.env import AgentAction, AgentObservation
from argusorch.agents.types import PolicyEval


class LLMActor:
    def __init__(
        self, model: nn.Module, tokenizer: Any, generation_config: Dict[str, Any]
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def act(self, obs: AgentObservation) -> AgentAction:
        completion = self.generate(obs.prompt)
        logprob = self.compute_logprob(obs.prompt, completion)
        return AgentAction(text=completion, logprob=logprob)

    def evaluate_action(self, obs: AgentObservation, act: AgentAction) -> PolicyEval:
        logprobs = self.compute_logprob(obs.prompt, act.text)
        return PolicyEval(logprobs=logprobs)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = (
            self.generation_config if isinstance(self.generation_config, dict) else {}
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        completion_ids = outputs[0][input_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def compute_logprob(self, prompt: str, completion: str) -> float:
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_length = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[
            1
        ]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        shift_logits = logits[0, prompt_length - 1 : -1, :].contiguous()
        shift_labels = inputs["input_ids"][0, prompt_length:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(1)).squeeze(
            1
        )

        return token_log_probs.sum().item()
