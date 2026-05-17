class MAACTrainer:
    def __init__(
        self,
        config,
        dataloader,
        rollout_collector,
        estimator,
        replay_buffer,
        updater,
    ):
        self.config = config
        self.dataloader = dataloader
        self.rollout_collector = rollout_collector
        self.estimator = estimator
        self.replay_buffer = replay_buffer
        self.updater = updater

    def train(self):
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                traj = self.rollout_collector.collect(batch)
                targets = self.estimator.compute(traj)
                self.replay_buffer.add(traj, targets)

                if self.replay_buffer.ready():
                    train_batch = self.replay_buffer.sample()
                    self.updater.update(train_batch)