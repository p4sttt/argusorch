from typing import Dict, List, Optional

import torch

from argusorch.trainers.common.types import TrainingBatch
from argusorch.agents.llm_actor import LLMActor
from argusorch.agents.critic import CentralizedCritic
from argusorch.agents.types import ValuePrediction
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
        critic_chunk_size: int = 8,
    ) -> None:
        self.actors = actors
        self.critic = critic
        self.actor_optimizers = actor_optimizers
        self.critic_optimizer = critic_optimizer
        self.loss = loss
        self.ppo_epochs = ppo_epochs
        self.device = device or torch.device("cpu")
        self.critic_chunk_size = critic_chunk_size

    def _evaluate_critic_chunked(self, joint_states: list) -> ValuePrediction:
        all_values: List[torch.Tensor] = []
        first_prompt = ""
        for i in range(0, len(joint_states), self.critic_chunk_size):
            chunk = joint_states[i : i + self.critic_chunk_size]
            pred = self.critic.evaluate_states(chunk)
            all_values.append(pred.values)
            if i == 0:
                first_prompt = pred.critic_prompt

        values = torch.cat(all_values, dim=0) if len(all_values) > 1 else all_values[0]
        return ValuePrediction(values=values, critic_prompt=first_prompt)

    def update(self, batch: TrainingBatch) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        total_critic_loss = 0.0
        total_actor_losses = {agent_id: 0.0 for agent_id in batch.agent_ids()}

        for _ in range(self.ppo_epochs):
            for agent_id in batch.agent_ids():
                obs_list = batch.observations(agent_id)
                act_list = batch.actions(agent_id)

                old_lp = batch.old_logprobs(agent_id)
                advs = batch.advantages(agent_id)

                n_samples = len(obs_list)
                self.actor_optimizers[agent_id].zero_grad()

                for i in range(n_samples):
                    # forward pass per sample
                    single_new_lp = self.actors[agent_id].compute_logprob(
                        obs_list[i].prompt, act_list[i].text, no_grad=False
                    )

                    single_old_lp = old_lp[i : i + 1].to(single_new_lp.device)
                    single_adv = advs[i : i + 1].to(single_new_lp.device)

                    loss_a_item = (
                        self.loss.actor_loss(
                            single_new_lp.unsqueeze(0), single_old_lp, single_adv
                        )
                        / n_samples
                    )

                    loss_a_item.backward()
                    total_actor_losses[agent_id] += loss_a_item.item() * n_samples

                    # free up memory
                    del single_new_lp, loss_a_item

                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_id].model.parameters(), max_norm=1.0
                )
                self.actor_optimizers[agent_id].step()

            critic_eval = self._evaluate_critic_chunked(batch.joint_states)
            loss_c = self.loss.critic_loss(
                critic_eval.values,
                batch.value_targets.to(critic_eval.values.device),
            )

            self.critic_optimizer.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            total_critic_loss += loss_c.item()

            # clean cache to prevent OOM in next epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics["critic_loss"] = total_critic_loss / self.ppo_epochs
        for agent_id in batch.agent_ids():
            metrics[f"actor_loss_{agent_id}"] = (
                total_actor_losses[agent_id] / self.ppo_epochs
            )

        return metrics
