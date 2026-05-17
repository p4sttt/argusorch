from typing import List
from argusorch.env import JointState
from .types import ValuePrediction


class CentralizedCritic:
    def __init__(self, model, tokenizer, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder

    def evaluate_state(self, joint_state: JointState) -> ValuePrediction:
        critic_prompt = self.prompt_builder.build(joint_state)
        value = self.forward_value(critic_prompt)
        return ValuePrediction(values=value, critic_prompt=critic_prompt)

    def evaluate_states(self, joint_states: List[JointState]) -> ValuePrediction:
        # Ожидается батчевая оценка состояний (используется в MAACUpdater)
        raise NotImplementedError("Батчевая оценка состояний пока не реализована")

    def forward_value(self, prompt: str) -> float:
        raise NotImplementedError(
            "Метод forward (оценка ценности промпта) должен быть реализован"
        )
