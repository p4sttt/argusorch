from abc import ABC, abstractmethod

from argusorch.env.trajectory import MultiAgentTrajectory


class TargetEstimator(ABC):
    @abstractmethod
    def compute(self, trajectory: MultiAgentTrajectory) -> MultiAgentTrajectory:
        pass


class TD0TargetEstimator(TargetEstimator):
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def compute(self, trajectory: MultiAgentTrajectory) -> MultiAgentTrajectory:
        for agent_traj in trajectory.by_agent():
            for t, transition in enumerate(agent_traj):
                r = transition.per_agent_reward
                if transition.done:
                    target = r
                else:
                    target = r + self.gamma * agent_traj[t + 1].value

                transition.value_target = target
                transition.advantage = target - transition.value

        return trajectory


class GAEEstimator(TargetEstimator):
    def __init__(self, gamma: float, lambda_: float) -> None:
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute(self, trajectory: MultiAgentTrajectory) -> MultiAgentTrajectory:
        for agent_traj in trajectory.by_agent():
            gae = 0.0

            for t in reversed(range(len(agent_traj))):
                transition = agent_traj[t]
                r = transition.per_agent_reward
                next_value = 0.0 if transition.done else agent_traj[t + 1].value

                delta = r + self.gamma * next_value - transition.value

                gae = delta + self.gamma * self.lambda_ * gae

                transition.advantage = gae
                transition.value_target = transition.value + gae

        return trajectory


class TDLambdaEstimator(TargetEstimator):
    def __init__(self, gamma: float, lambda_: float) -> None:
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute(self, trajectory: MultiAgentTrajectory) -> MultiAgentTrajectory:
        for agent_traj in trajectory.by_agent():
            g_lambda = 0.0
            for t in reversed(range(len(agent_traj))):
                transition = agent_traj[t]
                r = transition.per_agent_reward
                if transition.done:
                    g_lambda = r
                else:
                    next_value = agent_traj[t + 1].value
                    g_lambda = r + self.gamma * (
                        (1 - self.lambda_) * next_value + self.lambda_ * g_lambda
                    )

                transition.value_target = g_lambda
                transition.advantage = g_lambda - transition.value

        return trajectory
