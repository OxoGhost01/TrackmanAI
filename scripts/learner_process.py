import time
import copy
import torch

from agent.trainer import PPOTrainer
from agent.save_utils import save_agent
from config_files.input_list import inputs as INPUTS_LIST
from tools.stats_logger import StatsLogger


def learner_process_fn(
    rollout_queues,
    shared_network,
    shared_network_lock,
    shared_steps,
    base_dir,
    save_dir,
    tensorboard_base_dir,
    shared_best_time=None,
    best_time_lock=None,
    device=None,
    gamma=0.995,
    clip_epsilon=0.2,
    lr=1e-4,
    value_coef=1.0,
    entropy_coef=0.15,
    batch_episodes=4,
    save_every_n_updates=50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Learner] Training device: {device}")

    stats_logger = StatsLogger(save_dir)

    local_network = copy.deepcopy(shared_network)
    local_network.to(device)

    trainer = PPOTrainer(
        agent=local_network,
        input_list=INPUTS_LIST,
        device=device,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        ppo_epochs=3,
    )

    update_count = 0

    while True:
        episodes = []
        batch_stats = []

        while len(episodes) < batch_episodes:
            for q in rollout_queues:
                if not q.empty():
                    payload = q.get()

                    episode = {
                        "obs": payload["obs"],
                        "actions": payload["actions"],
                        "rewards": payload["rewards"],
                    }
                    episodes.append(episode)
                    batch_stats.append(payload["stats"])

                    if len(episodes) >= batch_episodes:
                        break

            time.sleep(0.01)

        if not episodes:
            print("[Learner] No episodes collected, skipping.")
            continue

        info = trainer.update_from_episodes(episodes)

        with shared_network_lock:
            for shared_param, local_param in zip(
                shared_network.parameters(), local_network.parameters()
            ):
                shared_param.data.copy_(local_param.data.cpu())

        update_count += 1

        print(f"[Learner] Update #{update_count} | {info}")
        for i, s in enumerate(batch_stats):
            finished = s.get("race_finished", False)
            race_time = s.get("race_time", 0)
            print(f"  Ep {i}: finished={finished}, time={race_time}ms")

        stats_logger.log(update_count, info, batch_stats)

        if update_count % save_every_n_updates == 0:
            try:
                save_agent(shared_network, save_dir)
                print(f"[Learner] Checkpoint saved at update {update_count}")
            except Exception as e:
                print(f"[Learner] Failed to save: {e}")
