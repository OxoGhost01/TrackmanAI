import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrackmaniaAgent(nn.Module):
    def __init__(self, state_dim=16, action_dim=3, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy(x), self.value(x)

    def act(self, obs):
        policy_logits, _ = self.forward(obs)
        steer = torch.tanh(policy_logits[:, 0])
        gas = (torch.sigmoid(policy_logits[:, 1]) > 0.5).float()
        brake = (torch.sigmoid(policy_logits[:, 2]) > 0.5).float()
        log_prob = -((1 - steer**2) + gas + brake).sum(dim=-1) * 0.01  # dummy entropy-like logprob
        return steer.item(), gas.item(), brake.item(), log_prob

    def evaluate(self, obs, actions):
        policy_logits, values = self.forward(obs)
        steer_logits = policy_logits[:, 0]
        log_probs = -((steer_logits - actions[:, 0]) ** 2).mean(dim=-1)
        return steer_logits, actions[:, 1], actions[:, 2], log_probs, values.squeeze(-1)
