import time
from pathlib import Path
import torch
import numpy as np
from multiprocessing import Lock

from TMI.game_instance_manager import GameInstanceManager
from scripts.rewards import make_rewards_from_rollout
from agent.policy_adapter import make_exploration_policy

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
    game_spawning_lock=None,
):
    """
    Collector loop: runs game rollouts and pushes to queue.
    Each collector has a unique TMI port.
    """
    print(f"[Collector {process_number}] Starting on TMI port {tmi_port}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_network.to(device)
    shared_network.eval()
    
    exploration_policy = make_exploration_policy(
        shared_network, 
        net_lock=shared_network_lock
    )
    
    gim = GameInstanceManager(
        game_spawning_lock=None,
        tmi_port=tmi_port,
        max_minirace_duration_ms=2000,
        max_overall_duration_ms=100000,
    )
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            # Ensure game is running (handles locking internally)
            gim.ensure_game_launched()
            
            # Run rollout
            results, stats = gim.rollout(
                exploration_policy=exploration_policy,
                map_path="Level-1.Challenge.Gbx",
                zone_centers_filename="level1_0.5m_cl.npy",
                update_network=lambda: None,
            )
            
            # Process rollout into training data
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
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Update step counter
            with shared_steps.get_lock():
                shared_steps.value += len(obs_seq)
            
            # Package data for learner
            payload = {
                "obs": np.array(obs_seq, dtype=np.float32),
                "actions": np.array(actions_idx_seq, dtype=np.int64),
                "rewards": np.array(rewards_seq, dtype=np.float32),
                "stats": stats,
            }
            
            # Push to queue
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
            
            # Try to recover
            try:
                if gim.is_game_running():
                    print(f"[Collector {process_number}] Game still running, continuing...")
                else:
                    print(f"[Collector {process_number}] Game crashed, will restart on next iteration")
                    gim.iface = None
                    gim.tm_process_id = None
            except:
                pass
            
            # Exponential backoff
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
    
    # Cleanup
    print(f"[Collector {process_number}] Shutting down...")
    try:
        if gim.is_game_running():
            gim.close_game()
    except:
        pass