
import time
from pathlib import Path
import torch
import numpy as np

from TMI.game_instance_manager import GameInstanceManager
from scripts.rewards import make_rewards_from_rollout
from agent.policy_adapter import make_exploration_policy


def collector_process_fn(
    rollout_queue,
    shared_network,       # TrackmaniaAgent in shared memory
    shared_network_lock,
    game_spawning_lock,
    shared_steps,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
    shared_best_time,
    best_time_lock,
):
    """
    Collector loop, similar spirit to Linesight's collector_process_fn.
    Each process:
        - Has its own GameInstanceManager with its own port
        - Uses the shared agent for inference (with lock)
        - Pushes rollouts into rollout_queue
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_network.to(device)
    shared_network.eval()

    exploration_policy = make_exploration_policy(shared_network, net_lock=shared_network_lock)

    gim = GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        tmi_port=tmi_port,
        max_minirace_duration_ms=2000,
        max_overall_duration_ms=100000,
    )

    print(f"[Collector {process_number}] Started on TMI port {tmi_port}")

    while True:
        try:
            # --- Ensure game is up (your ensure_game_launched handles locking / port) ---
            gim.ensure_game_launched()

            # --- Rollout ---
            results, stats = gim.rollout(
                exploration_policy=exploration_policy,
                map_path="Level-1.Challenge.Gbx",
                zone_centers_filename="level1_0.5m_cl.npy",
                update_network=lambda: None,
            )

            obs_seq, actions_idx_seq, rewards_seq = make_rewards_from_rollout(results, stats)

            if len(obs_seq) == 0:
                print(f"[Collector {process_number}] Empty episode, stats={stats}")
                continue

            # Optional: increase shared step counter
            with shared_steps.get_lock():
                shared_steps.value += len(obs_seq)

            # Package data for learner
            payload = {
                "obs": np.array(obs_seq, dtype=np.float32),
                "actions": np.array(actions_idx_seq, dtype=np.int64),
                "rewards": np.array(rewards_seq, dtype=np.float32),
                "stats": stats,
            }

            rollout_queue.put(payload)

        except Exception as e:
            print(f"[Collector {process_number}] Exception: {e}")
            # give TMI / game a little breathing time before trying again
            time.sleep(2.0)
