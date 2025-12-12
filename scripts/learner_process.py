
import time
import torch

from agent.trainer import PPOTrainer
from config_files.input_list import inputs as INPUTS_LIST


def learner_process_fn(
    rollout_queues,
    shared_network,       # TrackmaniaAgent in shared memory
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
    lr=3e-5,
    value_coef=0.5,
    entropy_coef=0.05,
    batch_episodes=4,     # episodes per PPO update
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_network.to(device)

    # Create PPO trainer around the shared network
    trainer = PPOTrainer(
        agent=shared_network,
        input_list=INPUTS_LIST,
        device=device,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )

    episode_idx = 0

    while True:
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_stats = []

        # Collect batch_episodes episodes total across all queues
        while len(batch_stats) < batch_episodes:
            for q in rollout_queues:
                if not q.empty():
                    payload = q.get()

                    obs = payload["obs"]
                    actions = payload["actions"]
                    rewards = payload["rewards"]
                    stats = payload["stats"]

                    batch_obs.extend(obs)
                    batch_actions.extend(actions)
                    batch_rewards.extend(rewards)
                    batch_stats.append(stats)

                    episode_idx += 1
                    # print(f"[Learner] Got episode {episode_idx}, len={len(obs)}, stats={stats}")

                    if len(batch_stats) >= batch_episodes:
                        break

            time.sleep(0.01)

        if len(batch_obs) == 0:
            print("[Learner] Empty batch, skipping update.")
            continue

        # Convert to flat episode for PPO
        obs_flat = batch_obs
        actions_flat = batch_actions
        rewards_flat = batch_rewards

        # Lock network while doing update (protects collectors if they lock during inference)
        with shared_network_lock:
            info = trainer.update_from_episode(obs_flat, actions_flat, rewards_flat)

        print(f"[Learner] PPO update done. Info: {info}")
        print(f"[Learner] Batch stats (size={len(batch_stats)}):")
        for i, s in enumerate(batch_stats):
            print(f"  Episode {i}: {s}")
