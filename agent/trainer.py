# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Sequence

class PPOTrainer:
    def __init__(
        self,
        agent: nn.Module,
        input_list: Sequence[dict],
        device: torch.device = None,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
    ):
        self.agent = agent
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.input_list = input_list  # list of dicts mapping discrete idx -> binary actions

    def compute_returns(self, rewards, dones):
        R = 0.0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _actions_idx_to_binary(self, actions_idx):
        """
        Convert a list/array of discrete indices to binary action array shape (T,4)
        ordered as [left, right, accelerate, brake], dtype=float32
        """
        T = len(actions_idx)
        bin_actions = np.zeros((T, 4), dtype=np.float32)
        for i, idx in enumerate(actions_idx):
            cfg = self.input_list[idx]
            bin_actions[i, 0] = 1.0 if cfg["left"] else 0.0
            bin_actions[i, 1] = 1.0 if cfg["right"] else 0.0
            bin_actions[i, 2] = 1.0 if cfg["accelerate"] else 0.0
            bin_actions[i, 3] = 1.0 if cfg["brake"] else 0.0
        return bin_actions

    def update_from_episode(self, obs_seq, actions_idx_seq, rewards_seq):
        """
        Single PPO update from one episode trajectory.

        obs_seq: np.ndarray (T, state_dim) floats (these correspond to states used by the agent)
        actions_idx_seq: list/array length T of discrete indices (0..N-1)
        rewards_seq: list/array length T of scalar rewards (float)
        """
        if len(obs_seq) == 0:
            return

        # Convert to tensors
        obs = torch.tensor(np.asarray(obs_seq), dtype=torch.float32, device=self.device)  # [T, state_dim]
        actions_idx = np.asarray(actions_idx_seq, dtype=np.int64)
        rewards = np.asarray(rewards_seq, dtype=np.float32)

        # Convert discrete indices to binary action vectors (T,4)
        actions_bin = self._actions_idx_to_binary(actions_idx)  # numpy
        actions_bin_t = torch.tensor(actions_bin, dtype=torch.float32, device=self.device)  # [T, 4]

        # Compute returns
        dones = [False] * (len(rewards) - 1) + [True]
        returns = self.compute_returns(rewards.tolist(), dones).to(self.device)  # [T]

        # Evaluate current policy on obs (get logits and values)
        # We expect agent.evaluate(obs, actions_bin_t) to return:
        #   policy_logits (T,4), actions (T,4) (maybe same), log_probs (T), values (T)
        # But older agent.evaluate returns: policy_logits, actions, log_probs, values
        policy_logits, _, log_probs, values = self.agent.evaluate(obs, actions_bin_t)

        # Advantage
        values = values.squeeze(-1)
        advantage = returns - values.detach()

        # PPO ratio (log probs are for Bernoulli over 4 dims, summed as implemented earlier)
        old_logprobs = log_probs.detach()  # we treat rollout log-probs as "old"; in on-policy single-episode setting they are current
        # For stability, compute ratio with zeros where needed
        ratio = torch.exp(log_probs - old_logprobs)  # becomes 1.0 -> surr1 == surr2, safe

        # But in our simple single-episode update, old_logprobs == log_probs -> ratio==1
        # To still apply clipping we compute directly using current log_probs (this is a simplification)
        ratio = torch.exp(log_probs - old_logprobs)  # equals 1

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (returns - values).pow(2).mean()

        # entropy approximation for Bernoulli policy: sum over dims of -p*log p - (1-p)*log(1-p)
        probs = torch.sigmoid(policy_logits)
        eps = 1e-8
        entropy_per_dim = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        entropy = entropy_per_dim.sum(dim=-1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "episode_return": float(returns.sum().item()),
        }
