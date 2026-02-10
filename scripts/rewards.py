import numpy as np
from config_files import config_copy
from agent.normalize import normalize_floats


def make_rewards_from_rollout(results, stats, shared_best_time=None, best_time_lock=None):
    floats_seq = results.get("state_float", [])
    actions_seq = results.get("actions", [])
    prog_seq = results.get("meters_advanced_along_centerline", [])
    frames_seq = results.get("frames", [])

    if (
        len(floats_seq) == 0
        or len(actions_seq) == 0
        or len(prog_seq) == 0
        or len(frames_seq) == 0
    ):
        return [], [], []

    prog = np.array(prog_seq, dtype=np.float32)

    k_progress = 1.0
    k_time = -0.0001
    k_centering = 0.005
    max_lateral_dist = 30.0
    stuck_penalty = -5.0
    no_progress_threshold_m = 0.0005
    max_no_progress_steps = 40
    reward_finish = 1000.0

    T = min(len(prog), len(floats_seq), len(actions_seq), len(frames_seq)) - 1
    if T <= 0:
        return [], [], []

    obs_seq = []
    actions_idx_seq = []
    rewards_seq = []
    lateral_dists = []

    no_progress_counter = 0

    for t in range(T):
        floats_raw = np.asarray(floats_seq[t], dtype=np.float32)
        floats = normalize_floats(floats_raw)
        frame = np.asarray(frames_seq[t], dtype=np.float32).flatten() / 255.0

        obs = np.concatenate([floats, frame])
        obs_seq.append(obs)
        actions_idx_seq.append(int(actions_seq[t]))

        dm = float(prog[t + 1] - prog[t])
        if abs(dm) < no_progress_threshold_m:
            dm = 0.0

        r_progress = k_progress * dm

        r_time = k_time

        if dm <= 0:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        speed = max(float(floats_raw[0]), 0.1)
        r_speed = 0.001 * speed

        # Perpendicular distance from car (origin) to track centerline
        # using first two zone centers in car reference frame (indices 62-67)
        p0 = floats_raw[62:65]
        p1 = floats_raw[65:68]
        track_dir = p1 - p0
        track_dir_len = np.linalg.norm(track_dir)
        if track_dir_len > 0.01:
            lateral_dist = np.linalg.norm(np.cross(p0, track_dir)) / track_dir_len
        else:
            lateral_dist = np.linalg.norm(p0)
        lateral_dist = min(lateral_dist, max_lateral_dist)
        lateral_dists.append(lateral_dist)
        r_centering = -k_centering * lateral_dist

        r_stuck = stuck_penalty if no_progress_counter >= max_no_progress_steps else 0.0

        reward_t = r_progress + r_time + r_stuck + r_speed + r_centering
        rewards_seq.append(float(reward_t))

    stats["avg_lateral_dist"] = float(np.mean(lateral_dists)) if lateral_dists else 0.0

    race_done = stats.get("race_finished", False)
    race_time = float(stats.get("race_time", config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms))

    if race_done and len(rewards_seq) > 0:
        rewards_seq[-1] += reward_finish

        if shared_best_time is not None and best_time_lock is not None:
            with best_time_lock:
                cur_best = float(shared_best_time.value)

                if cur_best > 1e11:
                    shared_best_time.value = race_time
                elif race_time < cur_best:
                    improvement = cur_best - race_time
                    bonus = 0.05 * improvement
                    rewards_seq[-1] += bonus
                    shared_best_time.value = race_time
                    print(f"[INFO] PB improved -> {race_time} ms (bonus {bonus:.2f})")

    return obs_seq, actions_idx_seq, rewards_seq
