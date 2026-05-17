from typing import List

from argusorch.env import AgentAction, AgentObservation

from argusorch.agents.llm_actor import LLMActor
from argusorch.agents.types import PolicyEval


class AgentsGroup:
    def __init__(self, actors: List[LLMActor]) -> None:
        self.actors = actors

    def act(self, obs: List[AgentObservation]) -> List[AgentAction]:
        return [actor.act(obs) for actor, obs in zip(self.actors, obs)]

    def evaluate_actions(
        self, obs: List[AgentObservation], acts: List[AgentAction]
    ) -> List[PolicyEval]:
        return [
            actor.evaluate_action(o, a) for actor, o, a in zip(self.actors, obs, acts)
        ]
