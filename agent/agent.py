import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Bernoulli

FLOAT_DIM = 184
IMG_H = 120
IMG_W = 160


class TrackmaniaAgent(nn.Module):
    def __init__(self, state_dim=19384, action_dim=3, hidden_dim=256):
        super().__init__()
        self.float_dim = FLOAT_DIM
        self.img_h = IMG_H
        self.img_w = IMG_W
        self.action_dim = action_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_H, IMG_W)
            conv_out_dim = self.conv(dummy).shape[1]

        self.float_fc = nn.Linear(FLOAT_DIM, 128)

        combined_dim = conv_out_dim + 128
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

        nn.init.kaiming_normal_(self.float_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.float_fc.bias, 0.0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.normal_(self.policy.weight, std=1e-2)
        nn.init.constant_(self.policy.bias, 0.0)
        nn.init.normal_(self.value.weight, std=1e-2)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x):
        floats = x[:, :self.float_dim]
        img_flat = x[:, self.float_dim:]
        img = img_flat.view(-1, 1, self.img_h, self.img_w)

        img_features = self.conv(img)
        float_features = F.relu(self.float_fc(floats))

        combined = torch.cat([img_features, float_features], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        logits = self.policy(x)
        value = self.value(x)
        return logits, value

    def act(self, obs):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(obs)
            probs = torch.sigmoid(logits)
            m = Bernoulli(probs)
            sample = m.sample()
            log_prob = m.log_prob(sample)
            summed_log_prob = log_prob.sum(dim=-1).squeeze(0)

            left = float(sample[0, 0].item())
            right = float(sample[0, 1].item())
            gas = float(sample[0, 2].item())
            brake = 0.0

        return left, right, gas, brake, float(summed_log_prob.item())

    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        actions_3d = actions[:, :3]

        probs = torch.sigmoid(logits)
        eps = 1e-8
        log_probs_per_dim = actions_3d * torch.log(probs + eps) + (1 - actions_3d) * torch.log(1 - probs + eps)
        log_probs = log_probs_per_dim.sum(dim=-1)
        return logits, actions_3d, log_probs, values.squeeze(-1)
