import numpy as np

def make_exploration_policy(agent):
    def exploration_policy(frame, floats):
        obs = np.concatenate((floats, frame.flatten() / 255.0))
        steer, gas, brake, logprob = agent.act(obs)
        # convert to nearest discrete config_copy action
        action_idx = int(gas * 1 + brake * 2 + (steer > 0) * 3)
        action_was_greedy = True  # placeholder
        q_value = 0.0
        q_values = np.zeros(5)
        return action_idx, action_was_greedy, q_value, q_values
    return exploration_policy
