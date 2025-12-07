import os
import torch

def save_agent(agent, save_dir):
    path = os.path.join(save_dir, "agent.pt")
    print(f"[SAVE] Saving agent to {path}")
    torch.save(agent.state_dict(), path)


def load_agent(agent, save_dir):
    path = os.path.join(save_dir, "agent.pt")
    if not os.path.exists(path):
        print(f"[LOAD] No checkpoint found at {path}")
        return False

    print(f"[LOAD] Loading agent from {path}")
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    return True
