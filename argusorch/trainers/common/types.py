from dataclasses import dataclass
from typing import Dict, List

import torch

from argusorch.env import AgentAction, AgentObservation, JointState


@dataclass
class TrainingBatch:
    _agent_ids: List[str]
    _observations: Dict[str, List[AgentObservation]]
    _actions: Dict[str, List[AgentAction]]
    _old_logprobs: Dict[str, torch.Tensor]
    _advantages: Dict[str, torch.Tensor]
    joint_states: List[JointState]
    value_targets: torch.Tensor

    def agent_ids(self) -> List[str]:
        return self._agent_ids

    def observations(self, agent_id: str) -> List[AgentObservation]:
        return self._observations[agent_id]

    def actions(self, agent_id: str) -> List[AgentAction]:
        return self._actions[agent_id]

    def old_logprobs(self, agent_id: str) -> torch.Tensor:
        return self._old_logprobs[agent_id]

    def advantages(self, agent_id: str) -> torch.Tensor:
        return self._advantages[agent_id]
