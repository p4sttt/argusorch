from trajectory import MultiAgentTrajectory


class RolloutCollector:
    def __init__(self, env, actors, critic):
        self.env = env
        self.actors = actors
        self.critic = critic

    def collect(self, item: dict) -> MultiAgentTrajectory:
        observations = self.env.reset(item)
        trajectory = MultiAgentTrajectory()

        while True:
            joint_state = self.env.joint_state()
            value = self.critic.evaluate_state(joint_state)

            actions = self.actors.act(observations)
            step = self.env.step(actions)

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

        return trajectory
