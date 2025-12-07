import numpy as np
import torch

from config_files.input_list import inputs  # adjust path if needed


def make_exploration_policy(agent, net_lock=None):
    def exploration_policy(frame, floats):
        if net_lock is not None:
            net_lock.acquire()
        try:
            frame_norm = frame.astype(np.float32).flatten() / 255.0
            obs_np = np.concatenate((floats.astype(np.float32), frame_norm))
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

            left, right, gas, brake, log_prob = agent.act(obs_t)

            left = bool(round(left))
            right = bool(round(right))
            gas = bool(round(gas))
            brake = bool(round(brake))

            target = {
                "left": left,
                "right": right,
                "accelerate": gas,
                "brake": brake,
            }

            best_idx = 0
            best_dist = 1e9
            for i, cfg in enumerate(inputs):
                dist = (
                    (cfg["left"] != target["left"])
                    + (cfg["right"] != target["right"])
                    + (cfg["accelerate"] != target["accelerate"])
                    + (cfg["brake"] != target["brake"])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            action_idx = best_idx
            action_was_greedy = True
            q_value = 0.0
            q_values = np.zeros(len(inputs))

            return action_idx, action_was_greedy, q_value, q_values
        finally:
            if net_lock is not None:
                net_lock.release()

    return exploration_policy
