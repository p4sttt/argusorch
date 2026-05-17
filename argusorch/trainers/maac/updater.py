from typing import Any, Dict

import torch

from argusorch.trainers.common.types import TrainingBatch
from argusorch.agents.llm_actor import LLMActor
from argusorch.agents.critic import CentralizedCritic
from argusorch.trainers.maac.losses import MAACLoss


class MAACUpdater:
    def __init__(
        self,
        actors: Dict[str, LLMActor],
        critic: CentralizedCritic,
        actor_optimizers: Dict[str, torch.optim.Optimizer],
        critic_optimizer: torch.optim.Optimizer,
        loss: MAACLoss,
        ppo_epochs: int = 4,
    ) -> None:
        self.actors = actors
        self.critic = critic
        self.actor_optimizers = actor_optimizers
        self.critic_optimizer = critic_optimizer
        self.loss = loss
        self.ppo_epochs = ppo_epochs

    def update(self, batch: TrainingBatch) -> None:
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
                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_id].model.parameters(), max_norm=1.0
                )
                self.actor_optimizers[agent_id].step()

            critic_eval = self.critic.evaluate_states(batch.joint_states)
            loss_c = self.loss.critic_loss(
                critic_eval.values,
                batch.value_targets,
            )

            self.critic_optimizer.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
