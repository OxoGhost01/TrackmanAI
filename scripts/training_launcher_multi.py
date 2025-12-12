# train_ppo.py

import ctypes
import os
import signal
import sys
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Lock
from art import tprint

from config_files import config_copy

from scripts.collector_process import collector_process_fn
from scripts.learner_process import learner_process_fn
from agent.agent import TrackmaniaAgent
from agent.save_utils import load_agent, save_agent

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")


def clear_tm_instances():
    if config_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances, and saving the agent...")
    save_agent(shared_network, save_dir)
    clear_tm_instances()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


if __name__ == "__main__":
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
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0

    gpu_collectors_count = 4
    rollout_queues = [mp.Queue(config_copy.max_rollout_queue_size) for _ in range(gpu_collectors_count)]

    shared_network_lock = Lock()
    game_spawning_lock = Lock()

    shared_best_time = mp.Value(ctypes.c_int64, float(1e12))
    best_time_lock = Lock()

    # Create shared TrackmaniaAgent
    state_dim = 19384
    action_dim = 4
    shared_network = TrackmaniaAgent(state_dim=state_dim, action_dim=action_dim)

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
                rollout_queue,
                shared_network,
                shared_network_lock,
                game_spawning_lock,
                shared_steps,
                base_dir,
                save_dir,
                base_tmi_port + process_number,
                process_number,
                shared_best_time,
                best_time_lock,
            ),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(gpu_collectors_count))
    ]

    for collector_process in collector_processes:
        collector_process.start()

    # --- Start learner in main process (like Linesight) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner_process_fn(
        rollout_queues,
        shared_network,
        shared_network_lock,
        shared_steps,
        base_dir,
        save_dir,
        tensorboard_base_dir,
        shard_best_time = shared_best_time,
        device=device,
    )

    # If learner exits, wait for collectors
    for collector_process in collector_processes:
        collector_process.join()
