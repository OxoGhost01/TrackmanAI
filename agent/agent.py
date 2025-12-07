import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Bernoulli

class TrackmaniaAgent(nn.Module):
    def __init__(self, state_dim=19384, action_dim=4, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)  # 4 logits for [left,right,gas,brake]
        self.value = nn.Linear(hidden_dim, 1)

        # optional: small weight init to avoid immediate saturation
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.normal_(self.policy.weight, std=1e-2)
        nn.init.constant_(self.policy.bias, 0.0)
        nn.init.normal_(self.value.weight, std=1e-2)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)   # shape [B,4]
        value = self.value(x)     # shape [B,1]
        return logits, value

    def act(self, obs):
        """
        obs: torch tensor shape [1, state_dim] (single sample)
        Returns: left, right, gas, brake (floats 0.0/1.0), summed_log_prob (torch scalar or float)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(obs)        # [1,4]
            probs = torch.sigmoid(logits)       # [1,4] in (0,1)
            # sample bernoulli (stochastic action)
            m = Bernoulli(probs)
            sample = m.sample()                 # [1,4] values 0 or 1 (float tensor)
            log_prob = m.log_prob(sample)       # [1,4] log prob per dim
            summed_log_prob = log_prob.sum(dim=-1).squeeze(0)  # scalar tensor

            # convert to python floats for the rest of pipeline
            left = float(sample[0, 0].item())
            right = float(sample[0, 1].item())
            gas = float(sample[0, 2].item())
            # brake = float(sample[0, 3].item())
            brake = 0

        return left, right, gas, brake, float(summed_log_prob.item())

    def evaluate(self, obs, actions):
        """
        obs: [B, state_dim]
        actions: [B,4] 0/1 floats (binary)
        returns: policy_logits, actions (same), log_probs (sum per sample), values [B]
        """
        logits, values = self.forward(obs)  # [B,4], [B,1]
        probs = torch.sigmoid(logits)
        eps = 1e-8
        # Bernoulli log-prob for each dimension:
        # log p(a) = a*log(p) + (1-a)*log(1-p)
        log_probs_per_dim = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
        log_probs = log_probs_per_dim.sum(dim=-1)  # [B]
        return logits, actions, log_probs, values.squeeze(-1)
