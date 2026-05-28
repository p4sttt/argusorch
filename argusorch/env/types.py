from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentObservation:
    agent_id: str
    prompt: str
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    text: str
    logprob: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvStep:
    next_observations: Dict[str, AgentObservation]
    reward: float
    done: bool
    per_agent_rewards: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JointState:
    item: Dict[str, Any]
    history: List[Dict[str, Any]]
    turn: int
