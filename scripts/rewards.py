import numpy as np
from config_files import config_copy

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