import time
from pathlib import Path
import numpy as np

from TMI.game_instance_manager import GameInstanceManager
from scripts.rewards import make_rewards_from_rollout
from agent.policy_adapter import make_exploration_policy
from config_files import config_copy


def collector_process_fn(
    rollout_queue,
    shared_network,
    shared_network_lock,
    shared_steps,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
    shared_best_time,
    best_time_lock,
    game_spawning_lock,
):
    print(f"[Collector {process_number}] Starting on TMI port {tmi_port}")

    shared_network.eval()

    exploration_policy = make_exploration_policy(
        shared_network,
        net_lock=shared_network_lock
    )

    gim = GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        tmi_port=tmi_port,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
    )

    consecutive_failures = 0
    max_consecutive_failures = 5

    while True:
        try:
            gim.ensure_game_launched()

            results, stats = gim.rollout(
                exploration_policy=exploration_policy,
                map_path="Level-1.Challenge.Gbx",
                zone_centers_filename="level1_0.5m_cl.npy",
                update_network=lambda: None,
            )

            obs_seq, actions_idx_seq, rewards_seq = make_rewards_from_rollout(
                results, stats, shared_best_time, best_time_lock
            )

            if len(obs_seq) == 0:
                print(f"[Collector {process_number}] Empty episode, skipping. Stats: {stats}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[Collector {process_number}] Too many empty episodes, restarting game...")
                    try:
                        gim.close_game()
                        gim.iface = None
                    except:
                        pass
                    consecutive_failures = 0
                    time.sleep(5.0)
                continue

            consecutive_failures = 0

            with shared_steps.get_lock():
                shared_steps.value += len(obs_seq)

            payload = {
                "obs": np.array(obs_seq, dtype=np.float32),
                "actions": np.array(actions_idx_seq, dtype=np.int64),
                "rewards": np.array(rewards_seq, dtype=np.float32),
                "stats": stats,
            }

            try:
                rollout_queue.put(payload, timeout=5.0)
                print(f"[Collector {process_number}] Pushed episode: "
                      f"len={len(obs_seq)}, "
                      f"total_reward={sum(rewards_seq):.2f}, "
                      f"finished={stats.get('race_finished', False)}")
            except:
                print(f"[Collector {process_number}] Queue full, skipping episode")

        except KeyboardInterrupt:
            print(f"[Collector {process_number}] Interrupted by user")
            break
        except Exception as e:
            print(f"[Collector {process_number}] Exception: {e}")
            import traceback
            traceback.print_exc()

            consecutive_failures += 1

            try:
                if gim.is_game_running():
                    print(f"[Collector {process_number}] Game still running, continuing...")
                else:
                    print(f"[Collector {process_number}] Game crashed, will restart")
                    gim.iface = None
                    gim.tm_process_id = None
            except:
                pass

            sleep_time = min(10.0, 2.0 * (1.5 ** consecutive_failures))
            print(f"[Collector {process_number}] Sleeping {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)

            if consecutive_failures >= max_consecutive_failures:
                print(f"[Collector {process_number}] Too many failures, attempting full restart...")
                try:
                    gim.close_game()
                except:
                    pass
                gim.iface = None
                gim.tm_process_id = None
                consecutive_failures = 0
                time.sleep(10.0)

    print(f"[Collector {process_number}] Shutting down...")
    try:
        if gim.is_game_running():
            gim.close_game()
    except:
        pass
