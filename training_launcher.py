from agent.agent import TrackmaniaAgent
from TMI.game_instance_manager import GameInstanceManager
from agent.policy_adapter import make_exploration_policy
import numpy as np

def update_network():
    pass  # Will call PPOTrainer.update() later cuz now i'm checking if this works

if __name__ == "__main__":
    agent = TrackmaniaAgent(obs_dim=1000, action_dim=3)
    exploration_policy = make_exploration_policy(agent)

    gim = GameInstanceManager(game_spawning_lock=None, tmi_port=5400)
    results, stats = gim.rollout(
        exploration_policy=exploration_policy,
        map_path="A01-Race.Challenge.Gbx",
        zone_centers=np.zeros((10, 3)),  # dummy placeholder
        update_network=update_network
    )

    print("Rollout finished:", stats)
