from typing import Any, Dict

from argusorch.agents.actors_group import AgentsGroup
from argusorch.agents.critic import CentralizedCritic
from argusorch.env.text_env import MultiAgentTextEnv
from argusorch.env.trajectory import MultiAgentTrajectory


class RolloutCollector:
    def __init__(
        self, env: MultiAgentTextEnv, actors: AgentsGroup, critic: CentralizedCritic
    ) -> None:
        self.env = env
        self.actors = actors
        self.critic = critic

    def collect(self, item: Dict[str, Any]) -> tuple[MultiAgentTrajectory, Dict[str, float]]:
        observations = self.env.reset(item)
        trajectory = MultiAgentTrajectory()
        total_reward = 0.0
        steps_count = 0

        while True:
            joint_state = self.env.joint_state()
            value = self.critic.evaluate_state(joint_state)

            obs_list = list(observations.values())
            actions_list = self.actors.act(obs_list)
            actions = {obs.agent_id: act for obs, act in zip(obs_list, actions_list)}

            step = self.env.step(actions)
            total_reward += step.reward
            steps_count += 1

            trajectory.add(
                observations=observations,
                actions=actions,
                reward=step.reward,
                value=value,
                next_observations=step.next_observations,
                done=step.done,
            )

            if step.done:
                break

            observations = step.next_observations

        metrics = {
            "reward": total_reward,
            "episode_length": float(steps_count)
        }
        return trajectory, metrics
