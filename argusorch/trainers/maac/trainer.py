from typing import Any, Iterable

from argusorch.env.rollout_collector import RolloutCollector
from argusorch.trainers.common.replay_buffer import ReplayBuffer
from argusorch.trainers.common.target_estimation import TargetEstimator
from argusorch.trainers.maac.updater import MAACUpdater
from argusorch.trainers.common.loggers import BaseLogger


class MAACTrainer:
    def __init__(
        self,
        config: Any,
        dataloader: Iterable[Any],
        rollout_collector: RolloutCollector,
        estimator: TargetEstimator,
        replay_buffer: ReplayBuffer,
        updater: MAACUpdater,
        logger: BaseLogger,
    ) -> None:
        self.config = config
        self.dataloader = dataloader
        self.rollout_collector = rollout_collector
        self.estimator = estimator
        self.replay_buffer = replay_buffer
        self.updater = updater
        self.logger = logger
        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                traj, env_metrics = self.rollout_collector.collect(batch)
                targets = self.estimator.compute(traj)
                self.replay_buffer.add(traj, targets)

                metrics = env_metrics.copy()
                
                if self.replay_buffer.ready():
                    train_batch = self.replay_buffer.sample()
                    update_metrics = self.updater.update(train_batch)
                    metrics.update(update_metrics)

                self.global_step += 1
                self.logger.log(metrics, self.global_step)

