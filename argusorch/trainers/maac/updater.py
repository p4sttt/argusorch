from typing import Dict, Optional

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
        device: Optional[torch.device] = None,
    ) -> None:
        self.actors = actors
        self.critic = critic
        self.actor_optimizers = actor_optimizers
        self.critic_optimizer = critic_optimizer
        self.loss = loss
        self.ppo_epochs = ppo_epochs
        self.device = device or torch.device("cpu")

    def update(self, batch: TrainingBatch) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        total_critic_loss = 0.0
        total_actor_losses = {agent_id: 0.0 for agent_id in batch.agent_ids()}

        for _ in range(self.ppo_epochs):
            for agent_id in batch.agent_ids():
                obs_list = batch.observations(agent_id)
                act_list = batch.actions(agent_id)

                # old_logprobs и advantages уже на self.device (созданы в ReplayBuffer)
                old_lp = batch.old_logprobs(agent_id)  # Tensor on device
                advs = batch.advantages(agent_id)  # Tensor on device

                # Пересчитываем log-prob для каждого шага (последовательно, т.к. LLM)
                new_lp_list = [
                    self.actors[agent_id].compute_logprob(obs.prompt, act.text, no_grad=False)
                    for obs, act in zip(obs_list, act_list)
                ]
                new_lp = torch.stack(new_lp_list)  # (T,) on model device

                # Выравниваем устройства перед loss
                loss_a = self.loss.actor_loss(
                    new_lp,
                    old_lp.to(new_lp.device),
                    advs.to(new_lp.device),
                )

                self.actor_optimizers[agent_id].zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_id].model.parameters(), max_norm=1.0
                )
                self.actor_optimizers[agent_id].step()
                total_actor_losses[agent_id] += loss_a.item()

            critic_eval = self.critic.evaluate_states(batch.joint_states)
            loss_c = self.loss.critic_loss(
                critic_eval.values,
                batch.value_targets.to(critic_eval.values.device),
            )

            self.critic_optimizer.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            total_critic_loss += loss_c.item()

        metrics["critic_loss"] = total_critic_loss / self.ppo_epochs
        for agent_id in batch.agent_ids():
            metrics[f"actor_loss_{agent_id}"] = (
                total_actor_losses[agent_id] / self.ppo_epochs
            )

        return metrics

