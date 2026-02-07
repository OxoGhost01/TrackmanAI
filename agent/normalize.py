import numpy as np

FLOAT_DIM = 184

NORM_SCALES = np.array([
    60.0,  # speed_mag (idx 0)
    *[1.0] * 20,  # prev_actions (idx 1-20) already 0/1
    *[1.0] * 4,  # is_sliding (idx 21-24) 0/1
    *[1.0] * 4,  # has_ground_contact (idx 25-28) 0/1
    *[0.15] * 4,  # damper_absorb (idx 29-32) range ~0-0.15
    1.0,  # gearbox_state (idx 33) 0/1/2
    5.0,  # gear (idx 34) 0-5
    10000.0,  # actual_rpm (idx 35) 0-10000
    30.0,  # counter_gearbox_state (idx 36) 0-30
    *[1.0] * 16,  # contact_materials (idx 37-52) 0/1
    *[5.0] * 3,  # angular_velocity (idx 53-55) ±5 rad/s
    *[60.0] * 3,  # velocity (idx 56-58) ±60 m/s
    *[1.0] * 3,  # y_map_vector (idx 59-61) unit vector
    *[100.0] * 120,  # zone_centers (idx 62-181) ±100m typical
    700.0,  # distance_to_finish (idx 182) 0-700
    1.0,  # is_freewheeling (idx 183) 0/1
], dtype=np.float32)


def normalize_floats(floats: np.ndarray) -> np.ndarray:
    """Normalize float features to roughly [-1, 1] or [0, 1] range."""
    return floats / NORM_SCALES


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Normalize full observation (floats + image). Image assumed already /255."""
    floats = obs[:FLOAT_DIM]
    image = obs[FLOAT_DIM:]
    normalized_floats = normalize_floats(floats)
    return np.concatenate([normalized_floats, image])
