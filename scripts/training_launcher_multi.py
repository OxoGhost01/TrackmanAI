import ctypes
import os
import signal
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from art import tprint

from scripts.create_config import create_config_copy
from scripts.collector_process import collector_process_fn
from scripts.learner_process import learner_process_fn
from agent.agent import TrackmaniaAgent
from agent.save_utils import load_agent, save_agent

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")

def clear_tm_instances():
    """Kill all TrackMania instances."""
    if config_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")
    import time
    time.sleep(1.0)

def signal_handler(sig, frame):
    print("\n" + "="*60)
    print("Received SIGINT signal. Shutting down gracefully...")
    print("="*60)
    
    try:
        save_agent(shared_network, save_dir)
        print("[INFO] Agent saved successfully")
    except Exception as e:
        print(f"[ERROR] Could not save agent: {e}")
    
    clear_tm_instances()
    
    for child in mp.active_children():
        print(f"[INFO] Terminating child process: {child.name}")
        child.terminate()
    
    import time
    time.sleep(2.0)
    
    for child in mp.active_children():
        print(f"[WARN] Force killing: {child.name}")
        child.kill()
    
    tprint("Bye bye!", font="tarty1")
    sys.exit(0)

if __name__ == "__main__":
    create_config_copy()
    from config_files import config_copy
    
    signal.signal(signal.SIGINT, signal_handler)
    
    mp.set_start_method("spawn", force=True)
    
    clear_tm_instances()
    
    base_dir = Path(__file__).resolve().parents[1] 
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_base_dir = base_dir / "tensorboard"
    
    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("PPO TM Trainer", font="tarty1")
    print("\n" * 2)
    print("Training is starting!")
    
    if config_copy.is_linux:
        os.system(f"chmod +x {config_copy.linux_launch_game_path}")
    
    # --- Shared objects ---
    shared_steps = mp.Value(ctypes.c_int64, 0)
    
    gpu_collectors_count = config_copy.gpu_collectors_count
    rollout_queues = [
        mp.Queue(config_copy.max_rollout_queue_size)
        for _ in range(gpu_collectors_count)
    ]
    
    manager = mp.Manager()
    shared_network_lock = manager.Lock()
    shared_best_time = mp.Value(ctypes.c_double, 1e12)
    best_time_lock = manager.Lock()
    
    # CRITICAL: Single shared lock for game launching (prevents race conditions)
    game_spawning_lock = manager.Lock()
    
    shared_network = TrackmaniaAgent()
    
    if load_agent(shared_network, save_dir):
        print("[INFO] Loaded agent from checkpoint.")
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")
    
    shared_network.share_memory()
    
    # --- Start collector processes ---
    base_tmi_port = config_copy.base_tmi_port
    
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                rollout_queues[process_number],
                shared_network,
                shared_network_lock,
                shared_steps,
                base_dir,
                save_dir,
                base_tmi_port + process_number,
                process_number,
                shared_best_time,
                best_time_lock,
                game_spawning_lock,  # Pass the SAME lock to ALL collectors
            ),
            name=f"Collector-{process_number}",
        )
        for process_number in range(gpu_collectors_count)
    ]
    
    for collector_process in collector_processes:
        collector_process.start()
        print(f"[INFO] Started {collector_process.name}")
    
    import time
    time.sleep(2.0)
    
    # --- Start learner in main process ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Learner using device: {device}")
    
    try:
        learner_process_fn(
            rollout_queues,
            shared_network,
            shared_network_lock,
            shared_steps,
            base_dir,
            save_dir,
            tensorboard_base_dir,
            shared_best_time=shared_best_time,
            best_time_lock=best_time_lock,
            device=device,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Learner interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Learner crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Shutting down collectors...")
        for collector_process in collector_processes:
            collector_process.terminate()
        
        for collector_process in collector_processes:
            collector_process.join(timeout=5.0)
            if collector_process.is_alive():
                collector_process.kill()
        
        clear_tm_instances()
        print("[INFO] Shutdown complete")