import json
import time
from pathlib import Path


class StatsLogger:
    """Appends one JSON line per learner update to a log file in the save directory."""

    def __init__(self, save_dir: Path):
        self.log_path = save_dir / "training_log.jsonl"
        self.start_time = time.time()

    def log(self, update_idx: int, learner_info: dict, episode_stats: list):
        n_finished = sum(1 for s in episode_stats if s.get("race_finished", False))
        race_times = [
            s["race_time"] for s in episode_stats
            if s.get("race_finished", False) and "race_time" in s
        ]
        all_race_times = [s.get("race_time", 0) for s in episode_stats]

        entry = {
            "update": update_idx,
            "wall_time": time.time() - self.start_time,
            "policy_loss": learner_info.get("policy_loss", 0),
            "value_loss": learner_info.get("value_loss", 0),
            "entropy": learner_info.get("entropy", 0),
            "total_return": learner_info.get("total_return", 0),
            "n_steps": learner_info.get("n_steps", 0),
            "avg_reward_per_step": (
                learner_info.get("total_return", 0) / max(learner_info.get("n_steps", 1), 1)
            ),
            "episodes_finished": n_finished,
            "episodes_total": len(episode_stats),
            "best_race_time_ms": min(race_times) if race_times else None,
            "avg_race_time_ms": sum(all_race_times) / max(len(all_race_times), 1),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
