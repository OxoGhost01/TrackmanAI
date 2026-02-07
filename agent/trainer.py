import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Sequence, List

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
        entropy_coef: float = 0.01,
        ppo_epochs: int = 3,
        mini_batch_size: int = 512,
    ):
        self.agent = agent
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.input_list = input_list

    def compute_returns(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
        R = 0.0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _actions_idx_to_binary(self, actions_idx: np.ndarray) -> np.ndarray:
        T = len(actions_idx)
        bin_actions = np.zeros((T, 4), dtype=np.float32)
        for i, idx in enumerate(actions_idx):
            cfg = self.input_list[idx]
            bin_actions[i, 0] = float(cfg["left"])
            bin_actions[i, 1] = float(cfg["right"])
            bin_actions[i, 2] = float(cfg["accelerate"])
            bin_actions[i, 3] = float(cfg["brake"])
        return bin_actions

    def update_from_episodes(self, episodes: List[dict]) -> dict:
        if not episodes:
            return {}

        all_obs = []
        all_actions_bin = []
        all_rewards = []
        all_dones = []

        for ep in episodes:
            obs = ep["obs"]
            actions_idx = ep["actions"]
            rewards = ep["rewards"]

            if len(obs) == 0:
                continue

            all_obs.append(obs)
            all_actions_bin.append(self._actions_idx_to_binary(actions_idx))
            all_rewards.extend(rewards.tolist())

            ep_dones = [False] * (len(rewards) - 1) + [True]
            all_dones.extend(ep_dones)

        if not all_obs:
            return {}

        obs_t = torch.tensor(np.vstack(all_obs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.vstack(all_actions_bin), dtype=torch.float32, device=self.device)
        returns = self.compute_returns(all_rewards, all_dones)

        with torch.no_grad():
            _, _, old_log_probs, old_values = self.agent.evaluate(obs_t, actions_t)
            old_log_probs = old_log_probs.detach()
            advantages = returns - old_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = obs_t.shape[0]
        n_updates = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(N, device=self.device)

            for start in range(0, N, self.mini_batch_size):
                end = min(start + self.mini_batch_size, N)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                policy_logits, _, log_probs, values = self.agent.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()

                probs = torch.sigmoid(policy_logits)
                eps = 1e-8
                entropy_per_dim = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
                entropy = entropy_per_dim.sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "total_return": float(sum(all_rewards)),
            "n_steps": N,
        }
