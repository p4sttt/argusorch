from typing import List, Optional

import torch

from argusorch.trainers.common.types import TrainingBatch
from argusorch.env.trajectory import MultiAgentTrajectory


class ReplayBuffer:
    def __init__(self, capacity: int, device: Optional[torch.device] = None) -> None:
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.trajectories: List[MultiAgentTrajectory] = []

    def add(
        self,
        trajectory: MultiAgentTrajectory,
        targets: Optional[MultiAgentTrajectory] = None,
    ) -> None:
        self.trajectories.append(targets if targets is not None else trajectory)

    def ready(self) -> bool:
        return len(self.trajectories) >= self.capacity

    def sample(self) -> TrainingBatch:
        agent_ids = list(self.trajectories[0]._agent_trajectories.keys())

        obs_dict = {a_id: [] for a_id in agent_ids}
        act_dict = {a_id: [] for a_id in agent_ids}
        logprobs_dict = {a_id: [] for a_id in agent_ids}
        adv_dict = {a_id: [] for a_id in agent_ids}

        joint_states = []
        value_targets = []

        for traj in self.trajectories:
            first_agent = agent_ids[0]
            for trans in traj._agent_trajectories[first_agent]:
                if trans.joint_state is not None:
                    joint_states.append(trans.joint_state)
                value_targets.append(trans.value_target)

            for a_id in agent_ids:
                for trans in traj._agent_trajectories[a_id]:
                    obs_dict[a_id].append(trans.obs)
                    act_dict[a_id].append(trans.action)
                    logprobs_dict[a_id].append(trans.action.logprob)
                    adv_dict[a_id].append(trans.advantage)

        for a_id in agent_ids:
            logprobs_dict[a_id] = torch.tensor(
                logprobs_dict[a_id], dtype=torch.float32, device=self.device
            )
            adv_dict[a_id] = torch.tensor(
                adv_dict[a_id], dtype=torch.float32, device=self.device
            )

        value_targets_tensor = torch.tensor(
            value_targets, dtype=torch.float32, device=self.device
        )

        self.trajectories = []

        return TrainingBatch(
            _agent_ids=agent_ids,
            _observations=obs_dict,
            _actions=act_dict,
            _old_logprobs=logprobs_dict,
            _advantages=adv_dict,
            joint_states=joint_states,
            value_targets=value_targets_tensor,
        )
