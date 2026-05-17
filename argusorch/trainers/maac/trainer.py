from typing import Any, Iterable

from argusorch.env.rollout_collector import RolloutCollector
from argusorch.trainers.common.replay_buffer import ReplayBuffer
from argusorch.trainers.common.target_estimation import TargetEstimator
from argusorch.trainers.maac.updater import MAACUpdater


class MAACTrainer:
    def __init__(
        self,
        config: Any,
        dataloader: Iterable[Any],
        rollout_collector: RolloutCollector,
        estimator: TargetEstimator,
        replay_buffer: ReplayBuffer,
        updater: MAACUpdater,
    ) -> None:
        self.config = config
        self.dataloader = dataloader
        self.rollout_collector = rollout_collector
        self.estimator = estimator
        self.replay_buffer = replay_buffer
        self.updater = updater

    def train(self) -> None:
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                traj = self.rollout_collector.collect(batch)
                targets = self.estimator.compute(traj)
                self.replay_buffer.add(traj, targets)

                if self.replay_buffer.ready():
                    train_batch = self.replay_buffer.sample()
                    self.updater.update(train_batch)
