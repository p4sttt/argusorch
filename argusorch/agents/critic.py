from typing import Any, List

import torch
import torch.nn as nn

from argusorch.env import JointState
from argusorch.agents.types import ValuePrediction
from argusorch.agents.prompt_builder import PromptBuilder


class CentralizedCritic(nn.Module):
    def __init__(
        self, model: nn.Module, tokenizer: Any, prompt_builder: PromptBuilder
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder

        if hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        elif hasattr(model.config, "n_embd"):
            hidden_size = model.config.n_embd
        else:
            raise ValueError("Не удалось определить hidden_size из конфига модели.")

        self.value_head = nn.Linear(hidden_size, 1, bias=False).to(
            device=model.device, dtype=model.dtype
        )

    def evaluate_state(self, joint_state: JointState) -> ValuePrediction:
        critic_prompt = self.prompt_builder.build(joint_state)
        value = self.forward_value(critic_prompt)
        return ValuePrediction(values=value, critic_prompt=critic_prompt)

    def evaluate_states(self, joint_states: List[JointState]) -> ValuePrediction:
        prompts = [self.prompt_builder.build(js) for js in joint_states]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )

        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        values = self.value_head(last_hidden).squeeze(-1)  # Shape: (batch_size,)

        return ValuePrediction(
            values=values, critic_prompt=prompts[0] if prompts else ""
        )

    def forward_value(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0, -1, :]
            value = self.value_head(last_hidden).item()

        return value
