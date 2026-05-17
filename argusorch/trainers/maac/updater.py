class MAACUpdater:
    def __init__(
        self, actors, critic, actor_optimizers, critic_optimizer, loss, ppo_epochs=4
    ):
        self.actors = actors
        self.critic = critic
        self.actor_optimizers = actor_optimizers
        self.critic_optimizer = critic_optimizer
        self.loss = loss
        self.ppo_epochs = ppo_epochs

    def update(self, batch: TrainingBatch):
        for _ in range(self.ppo_epochs):
            for agent_id in batch.agent_ids():
                actor_eval = self.actors[agent_id].evaluate_action(
                    batch.observations(agent_id),
                    batch.actions(agent_id),
                )

                loss_a = self.loss.actor_loss(
                    actor_eval.logprobs,
                    batch.old_logprobs(agent_id),
                    batch.advantages(agent_id),
                )

                self.actor_optimizers[agent_id].zero_grad()
                loss_a.backward()
                self.actor_optimizers[agent_id].step()

            critic_eval = self.critic.evaluate_states(batch.joint_states)
            loss_c = self.loss.critic_loss(
                critic_eval.values,
                batch.value_targets,
            )

            self.critic_optimizer.zero_grad()
            loss_c.backward()
            self.critic_optimizer.step()
