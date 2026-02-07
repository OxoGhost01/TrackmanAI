import numpy as np
from sys import platform

# ==============================================
# GENERAL RUN INFO
# ==============================================
run_name = "test1 v2.0"  # name of the save
running_speed = 80  # runs 80x the speed of the game
is_linux = platform in ["linux", "linux2"]

# ==============================================
# SIMULATION SETTINGS
# ==============================================
tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action

# ==============================================
# INPUT CONFIGURATION
# ==============================================
n_zone_centers_in_inputs = 40
one_every_n_zone_centers_in_inputs = 20
n_zone_centers_extrapolate_after_end_of_map = 1000
n_zone_centers_extrapolate_before_start_of_map = 20
n_prev_actions_in_inputs = 5
n_contact_material_physics_behavior_types = 4

float_input_dim = (
    27
    + 3 * n_zone_centers_in_inputs
    + 4 * n_prev_actions_in_inputs
    + 4 * n_contact_material_physics_behavior_types
    + 1
)
W_downsized = 160
H_downsized = 120

# ==============================================
# EPISODE SETTINGS
# ==============================================
cutoff_rollout_if_race_not_finished_within_duration_ms = 60_000
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 3_000

# ==============================================
# TIMEOUTS
# ==============================================
timeout_during_run_ms = 10_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500
game_reboot_interval = 3600 * 12

# ==============================================
# PARALLELISM
# ==============================================
gpu_collectors_count = 4  # number of parallel game instances
update_inference_network_every_n_actions = 20  # number of actions between network updates
max_rollout_queue_size = 2

# ==============================================
# MAP GEOMETRY
# ==============================================
distance_between_checkpoints = 0.5  # set this in concordance with the map.npy you created
road_width = 90
max_allowable_distance_to_virtual_checkpoint = np.sqrt(
    (distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2
)
margin_to_announce_finish_meters = 700
deck_height = -np.inf
game_camera_number = 2
sync_virtual_and_real_checkpoints = True
