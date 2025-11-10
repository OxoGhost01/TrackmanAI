import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOTrainer:
    def __init__(self, agent, env_manager, device, gamma=0.99, clip_epsilon=0.2, lr=3e-4):
        self.agent = agent
        self.env = env_manager
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)

    def compute_returns(self, rewards, dones):
        R = 0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def train(self, num_episodes=100):
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0.0

            obs_buf, act_buf, rew_buf, done_buf, logprob_buf = [], [], [], [], []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                steer, gas, brake, log_prob = self.agent.act(obs_t)
                action = np.array([steer, gas, brake])
                next_obs, reward, done, _ = self.env.step(action)

                obs_buf.append(obs)
                act_buf.append(action)
                rew_buf.append(reward)
                done_buf.append(done)
                logprob_buf.append(log_prob.item())

                obs = next_obs
                total_reward += reward

            # Compute returns and update
            returns = self.compute_returns(rew_buf, done_buf)
            self.update(obs_buf, act_buf, logprob_buf, returns)

            print(f"Episode {ep+1}/{num_episodes} — total reward {total_reward:.2f}")

    def update(self, obs_buf, act_buf, old_logprobs, returns):
        obs = torch.tensor(obs_buf, dtype=torch.float32, device=self.device)
        actions = torch.tensor(act_buf, dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
        returns = returns.detach()

        steer, gas, brake, log_probs, values = self.agent.evaluate(obs, actions)
        advantage = returns - values.detach()

        ratio = torch.exp(log_probs - old_logprobs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
