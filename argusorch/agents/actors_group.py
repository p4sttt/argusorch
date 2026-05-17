from typings import List

from argusorch.env import AgentActions, AgentObservation, PolicyEval

from .llm_actor import LLMActor


class AgentsGroup:
    def __init__(self, actors: List[LLMActor]):
        self.actors = actors

    def act(self, obs: List[AgentObservation]) -> List[AgentActions]:
        return [actor.act(obs) for actor, obs in zip(self.actors, obs)]

    def evaluate_actions(
        self, obs: List[AgentObservation], acts: List[AgentActions]
    ) -> List[PolicyEval]:
        return [
            actor.evaluate_actions(obs, act)
            for actor, obs, act in zip(self.actors, obs, acts)
        ]
