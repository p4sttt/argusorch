from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from argusorch.env.types import AgentAction, AgentObservation


@dataclass
class Transition:
    obs: AgentObservation
    action: AgentAction
    reward: float
    value: float
    next_obs: AgentObservation
    done: bool

    advantage: float = 0.0
    value_target: float = 0.0


class MultiAgentTrajectory:
    def __init__(self):
        self._agent_trajectories: Dict[str, List[Transition]] = {}

    def add(
        self,
        observations: Dict[str, AgentObservation],
        actions: Dict[str, AgentAction],
        reward: float,
        value: float,
        next_observations: Dict[str, AgentObservation],
        done: bool,
    ) -> None:
        for agent_id, obs in observations.items():
            if agent_id not in self._agent_trajectories:
                self._agent_trajectories[agent_id] = []

            transition = Transition(
                obs=obs,
                action=actions[agent_id],
                reward=reward,
                value=value,
                next_obs=next_observations[agent_id],
                done=done,
            )
            self._agent_trajectories[agent_id].append(transition)

    def by_agent(self) -> Iterable[List[Transition]]:
        return self._agent_trajectories.values()

    def get_agent_trajectory(self, agent_id: str) -> List[Transition]:
        return self._agent_trajectories.get(agent_id, [])

    def __len__(self) -> int:
        if not self._agent_trajectories:
            return 0
        first_agent = next(iter(self._agent_trajectories))
        return len(self._agent_trajectories[first_agent])
