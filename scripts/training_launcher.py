# training_launcher.py
import os
from pathlib import Path
import time

import numpy as np
import torch

from agent.agent import TrackmaniaAgent
from agent.trainer import PPOTrainer
from TMI.game_instance_manager import GameInstanceManager
from agent.policy_adapter import make_exploration_policy
from config_files import config_copy
from config_files.input_list import inputs as INPUTS_LIST


def clear_tm_instances():
    if config_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        # kill by executable name (case-insensitive)
        os.system("taskkill /F /IM TmForever.exe >nul 2>&1")


def make_rewards_from_rollout(results, stats):
    """
    Build per-step rewards from rollout output.

    Reward components (all tunable below):

    r_t = r_progress + r_time + r_speed + r_stuck + r_terminal

    Where:
        - r_progress: proportional to delta meters along centerline
        - r_time: small constant negative per step
        - r_speed: small positive proportional to speed magnitude
        - r_stuck: penalty if no forward progress for many steps
        - r_terminal: big bonus if race finished, penalty otherwise

    Also reconstruct the same observation that policy_adapter sees:
        obs = concat(floats, frame.flatten()/255)

    Returns: (obs_seq, actions_idx_seq, rewards_seq)
        obs_seq: list of np.ndarray, each shape (state_dim == 19384,)
        actions_idx_seq: list of ints
        rewards_seq: list of floats
    """
    floats_seq = results.get("state_float", [])
    actions_seq = results.get("actions", [])
    prog_seq   = results.get("meters_advanced_along_centerline", [])
    frames_seq = results.get("frames", [])

    # Basic sanity checks
    if (
        len(floats_seq) == 0
        or len(actions_seq) == 0
        or len(prog_seq) == 0
        or len(frames_seq) == 0
    ):
        return [], [], []

    prog = np.array(prog_seq, dtype=np.float32)

    # -------------------------------
    # Hyperparameters for the reward
    # -------------------------------
    ms_per_step = config_copy.ms_per_action

    # 1) Progress reward
    k_progress = 1.0   # reward per meter advanced

    # 2) Time penalty (per second)
    k_time_per_s = -0.1
    k_time = k_time_per_s * (ms_per_step / 1000.0)

    # 3) Speed reward
    k_speed = 1e-3     # very small; speed is in m/s

    # 4) Stuck penalty
    no_progress_threshold_m = 0.001   # treat tiny movement as 0
    max_no_progress_steps   = 30      # after N steps without progress, punish
    stuck_penalty           = -1.0

    # 5) Terminal rewards
    reward_finish = 1000.0
    reward_fail   = -100.0

    # -------------------------------
    # Align sequences (skip index 0)
    # -------------------------------
    # We'll use steps 1..T; step t compares prog[t] with prog[t-1]
    T = min(
        len(prog) - 1,           # because we look at prog[t-1]
        len(floats_seq) - 1,
        len(actions_seq) - 1,
        len(frames_seq) - 1,
    )

    if T <= 0:
        return [], [], []

    obs_seq = []
    actions_idx_seq = []
    rewards_seq = []

    no_progress_counter = 0

    for t in range(1, 1 + T):
        # ---- Observation reconstruction ----
        floats = np.asarray(floats_seq[t], dtype=np.float32)  # (F,)

        frame = frames_seq[t]
        frame = np.asarray(frame, dtype=np.float32)           # (1, H, W)
        frame_norm = (frame / 255.0).flatten()                # (H*W,)

        obs = np.concatenate([floats, frame_norm])            # -> dim 19384
        obs_seq.append(obs)

        # ---- Action ----
        actions_idx_seq.append(int(actions_seq[t]))

        # ---- Progress ----
        dm = float(prog[t] - prog[t - 1])
        if abs(dm) < no_progress_threshold_m:
            dm = 0.0

        r_progress = k_progress * dm
        r_time = k_time

        # dm = delta progression along centerline that you already computed
        # prefer "fast progress", not just "fast speed"
        speed = float(floats[0])
        if dm > 0:
            # reward increases with progress and speed, but only if we're actually advancing
            r_speed = k_speed * speed * dm
        else:
            r_speed = 0.0

        """        # ---- Speed reward ----
        # rollout stored speed magnitude at floats[0]
        speed = float(floats[0])
        r_speed = k_speed * speed
        if speed < 1.0:
            r_speed += -0.05"""

        # ---- Stuck penalty ----
        if dm <= 0.0:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        r_stuck = 0.0
        if no_progress_counter >= max_no_progress_steps:
            r_stuck = stuck_penalty

        reward_t = r_progress + r_time + r_speed + r_stuck
        rewards_seq.append(float(reward_t))

    # ---- Terminal bonus / penalty ----
    if len(rewards_seq) > 0:
        if stats.get("race_finished", False):
            rewards_seq[-1] += reward_finish
        else:
            rewards_seq[-1] += reward_fail

    return obs_seq, actions_idx_seq, rewards_seq



if __name__ == "__main__":
    clear_tm_instances()
    time.sleep(0.2)

    # --- agent ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state_dim must match what policy_adapter builds (floats + flattened frame)
    agent = TrackmaniaAgent(state_dim=19384, action_dim=4).to(device)
    exploration_policy = make_exploration_policy(agent)

    # --- trainer ---
    trainer = PPOTrainer(
        agent=agent,
        input_list=INPUTS_LIST,
        device=device,
        gamma=0.995,
        clip_epsilon=0.2,
        lr=3e-5,
        value_coef=0.5,
        entropy_coef=0.05,
    )

    # --- Game Instance Manager ---
    gim = GameInstanceManager(game_spawning_lock=None, tmi_port=5400, max_minirace_duration_ms=2000, max_overall_duration_ms=100000)

    num_episodes = 10000
    for ep in range(num_episodes):
        print(f"=== Episode {ep+1}/{num_episodes} ===")
        # rollout returns full trajectory and stats
        results, stats = gim.rollout(
            exploration_policy=exploration_policy,
            map_path="Level-1.Challenge.Gbx",
            zone_centers_filename="level1_5_cl.npy",
            update_network=lambda: None,  # unused; we do offline update after rollout
        )

        obs_seq, actions_idx_seq, rewards_seq = make_rewards_from_rollout(results, stats)

        if len(obs_seq) == 0:
            print("Empty episode (no steps). Skipping update.")
            print("stats:", stats)
            continue

        info = trainer.update_from_episode(obs_seq, actions_idx_seq, rewards_seq)
        print("Episode stats:", stats)
        print("Trainer update:", info)

    # cleanup
    cfg = Path("config_files/config_copy.py")
    if cfg.exists():
        os.remove(cfg)

    clear_tm_instances()  
