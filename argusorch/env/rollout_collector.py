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
            value_pred = self.critic.evaluate_state(joint_state)
            value_scalar: float = (
                value_pred.values.item()
                if hasattr(value_pred.values, "item")
                else float(value_pred.values)
            )

            obs_list = list(observations.values())
            actions_list = self.actors.act(obs_list)
            actions = {obs.agent_id: act for obs, act in zip(obs_list, actions_list)}

            step = self.env.step(actions)
            total_reward += step.reward
            steps_count += 1

            # On terminal steps next_observations is an empty dict; build
            # placeholder observations so Transition.next_obs is always valid.
            next_obs_for_traj = step.next_observations
            if step.done and not next_obs_for_traj:
                from argusorch.env.types import AgentObservation as _Obs

                next_obs_for_traj = {
                    obs.agent_id: _Obs(agent_id=obs.agent_id, prompt="")
                    for obs in observations.values()
                }

            trajectory.add(
                observations=observations,
                actions=actions,
                reward=step.reward,
                per_agent_rewards=step.per_agent_rewards,
                value=value_scalar,
                next_observations=next_obs_for_traj,
                done=step.done,
                joint_state=joint_state,
            )

            if step.done:
                break

            observations = step.next_observations

        metrics = {
            "reward": total_reward,
            "episode_length": float(steps_count),
        }
        return trajectory, metrics
