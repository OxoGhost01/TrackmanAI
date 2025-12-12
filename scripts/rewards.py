import numpy as np
from config_files import config_copy

def make_rewards_from_rollout(results, stats, shared_best_time=None, best_time_lock=None):
    """
    Same as before, but accepts:
        shared_best_time: mp.Value(ctypes.c_double) in ms
        best_time_lock: mp.Lock() to synchronize read/write
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

    # hyperparams (same as your previous)
    ms_per_step = config_copy.ms_per_action
    k_progress = 0.001
    k_time_per_s = -0.1
    k_time = k_time_per_s * (ms_per_step / 1000.0)
    k_speed = 0.005
    no_progress_threshold_m = 0.001
    max_no_progress_steps = 30
    stuck_penalty = -1.0
    reward_finish = 1000.0
    reward_fail   = -100.0

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

    for t in range(1, 1 + T):
        floats = np.asarray(floats_seq[t], dtype=np.float32)
        frame = np.asarray(frames_seq[t], dtype=np.float32)
        frame_norm = (frame / 255.0).flatten()
        obs = np.concatenate([floats, frame_norm])
        obs_seq.append(obs)
        actions_idx_seq.append(int(actions_seq[t]))

        dm = float(prog[t] - prog[t - 1])
        if abs(dm) < no_progress_threshold_m:
            dm = 0.0

        r_progress = k_progress * dm
        r_time = k_time

        speed = float(floats[0])
        r_speed = k_speed * speed * dm if dm > 0 else 0.0

        if dm <= 0.0:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        r_stuck = stuck_penalty if no_progress_counter >= max_no_progress_steps else 0.0
        reward_t = r_progress + r_time + r_speed + r_stuck
        rewards_seq.append(float(reward_t))

    # ---- Terminal bonus / penalty ----
    race_done = stats.get("race_finished", False)
    race_time = float(stats.get("race_time", config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms))

    if race_done:
        # Base finish reward
        rewards_seq[-1] += reward_finish

        # Time performance reward using shared_best_time
        if shared_best_time is not None and best_time_lock is not None:
            with best_time_lock:
                cur_best = float(shared_best_time.value)
                if cur_best > 1e11:
                    shared_best_time.value = race_time
                    time_improvement = 0.0
                else:
                    time_improvement = cur_best - race_time

                # scale factor for improvement
                k_time_improve = 0.5
                rewards_seq[-1] += k_time_improve * time_improvement

                # update if new record
                if race_time < shared_best_time.value:
                    shared_best_time.value = race_time
        else:
            pass

    else:
        rewards_seq[-1] += reward_fail

    return obs_seq, actions_idx_seq, rewards_seq
