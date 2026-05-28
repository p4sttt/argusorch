from typing import Dict

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
        completion = self.generate(obs.prompt)
        logprob = self.compute_logprob(obs.prompt, completion)
        return AgentAction(text=completion, logprob=logprob.item())

    def evaluate_action(self, obs: AgentObservation, act: AgentAction) -> PolicyEval:
        logprobs = self.compute_logprob(obs.prompt, act.text, no_grad=False)
        return PolicyEval(logprobs=logprobs)

    def generate(self, prompt: str) -> str:
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        gen_kwargs = (
            self.generation_config if isinstance(self.generation_config, dict) else {}
        )

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                outputs = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        completion_ids = outputs[0][input_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def compute_logprob(
        self, prompt: str, completion: str, no_grad: bool = True
    ) -> torch.Tensor:
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
