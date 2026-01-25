import numpy as np
from config_files import config_copy

def make_rewards_from_rollout(results, stats, shared_best_time=None, best_time_lock=None):
    floats_seq = results.get("state_float", [])
    actions_seq = results.get("actions", [])
    prog_seq   = results.get("meters_advanced_along_centerline", [])
    frames_seq = results.get("frames", [])

    if (
        len(floats_seq) == 0
        or len(actions_seq) == 0
        or len(prog_seq) == 0
        or len(frames_seq) == 0
    ):
        return [], [], [],

    prog = np.array(prog_seq, dtype=np.float32)

    # --- Hyperparameters  ---
    k_progress = 1.0          # strong progress signal
    k_time = -0.0001           # tiny per-step time penalty
    stuck_penalty = -5.0       # only when really stuck
    no_progress_threshold_m = 0.0005
    max_no_progress_steps = 40
    reward_finish = 200.0

    T = min(
        len(prog) - 1,
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

    for t in range(1, T+1):
        floats = np.asarray(floats_seq[t], dtype=np.float32)
        frame = np.asarray(frames_seq[t], dtype=np.float32).flatten() / 255.0

        obs = np.concatenate([floats, frame])
        obs_seq.append(obs)
        actions_idx_seq.append(int(actions_seq[t]))

        # --- Progress ---
        dm = float(prog[t] - prog[t-1])
        if abs(dm) < no_progress_threshold_m:
            dm = 0.0

        r_progress = k_progress * dm

        # --- Small time penalty ---
        r_time = k_time

        # --- Stuck detection ---
        if dm <= 0:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        # --- Progress efficiency reward ---
        speed = max(float(floats[0]), 0.1)  # avoid div by zero
        efficiency = dm / speed             # meters per (m/s)

        k_efficiency = 0.05                 # small but decisive
        r_efficiency = k_efficiency * efficiency


        r_stuck = stuck_penalty if no_progress_counter >= max_no_progress_steps else 0.0

        reward_t = r_progress + r_time + r_stuck + r_efficiency
        rewards_seq.append(float(reward_t))

    

    # ---- TERMINAL ----
    race_done = stats.get("race_finished", False)
    race_time = float(stats.get("race_time", config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms))

    if race_done:
        # Base finish reward
        rewards_seq[-1] += reward_finish

        # Only bonus if beating PB
        if shared_best_time is not None and best_time_lock is not None:
            with best_time_lock:
                cur_best = float(shared_best_time.value)

                if cur_best > 1e11:  # first run ever
                    shared_best_time.value = race_time

                elif race_time < cur_best:
                    improvement = cur_best - race_time
                    bonus = 0.05 * improvement     # small, safe bonus
                    rewards_seq[-1] += bonus
                    shared_best_time.value = race_time
                    print(f"[INFO] PB improved -> {race_time} ms (bonus {bonus:.2f})")

    else:
        # No punishment — failure is already punished by very low total progress.
        pass

    return obs_seq, actions_idx_seq, rewards_seq
