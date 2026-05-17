import torch

class MAACLoss:
    def __init__(self, clip_eps=0.2):
        self.clip_eps = clip_eps

    def actor_loss(self, logprobs, old_logprobs, advantages):
        # ratio = exp(log_pi_new - log_pi_old) 
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def critic_loss(self, values, targets):
        return ((values - targets.detach()) ** 2).mean()