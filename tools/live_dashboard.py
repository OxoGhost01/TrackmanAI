"""
Live training dashboard.

Usage:
    python -m tools.live_dashboard              (auto-finds latest log)
    python -m tools.live_dashboard path/to/training_log.jsonl

The window refreshes every 5 seconds while training is running.
Close the window or Ctrl+C to stop.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def find_latest_log() -> Path:
    save_root = Path(__file__).resolve().parents[1] / "save"
    logs = list(save_root.glob("*/training_log.jsonl"))
    if not logs:
        print("No training_log.jsonl found in save/*/")
        sys.exit(1)
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    print(f"Using: {latest}")
    return latest


def load_entries(path: Path) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def draw(axes, entries):
    if len(entries) < 2:
        return

    updates = [e["update"] for e in entries]
    wall_min = [e["wall_time"] / 60 for e in entries]

    # --- 1. Total return ---
    ax = axes[0, 0]
    ax.clear()
    ax.plot(updates, [e["total_return"] for e in entries], color="#2196F3", linewidth=1)
    ax.set_title("Total Return (per batch)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Sum of rewards")
    ax.grid(True, alpha=0.3)

    # --- 2. Avg reward per step ---
    ax = axes[0, 1]
    ax.clear()
    ax.plot(updates, [e["avg_reward_per_step"] for e in entries], color="#4CAF50", linewidth=1)
    ax.set_title("Avg Reward / Step")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # --- 3. Policy loss ---
    ax = axes[1, 0]
    ax.clear()
    ax.plot(updates, [e["policy_loss"] for e in entries], color="#FF9800", linewidth=1)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # --- 4. Value loss ---
    ax = axes[1, 1]
    ax.clear()
    ax.plot(updates, [e["value_loss"] for e in entries], color="#F44336", linewidth=1)
    ax.set_title("Value Loss")
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # --- 5. Entropy ---
    ax = axes[2, 0]
    ax.clear()
    ax.plot(updates, [e["entropy"] for e in entries], color="#9C27B0", linewidth=1)
    ax.axhline(y=2.08, color="gray", linestyle="--", alpha=0.5, label="max (3×ln2)")
    ax.set_title("Entropy")
    ax.set_xlabel("Update")
    ax.set_ylim(bottom=0, top=2.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 6. Race finishes + best time ---
    ax = axes[2, 1]
    ax.clear()
    finish_pct = [
        100 * e["episodes_finished"] / max(e["episodes_total"], 1) for e in entries
    ]
    ax.plot(updates, finish_pct, color="#009688", linewidth=1, label="finish %")
    ax.set_title("Race Finish %")
    ax.set_xlabel("Update")
    ax.set_ylabel("%")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    best_times = [e["best_race_time_ms"] for e in entries if e["best_race_time_ms"] is not None]
    best_updates = [e["update"] for e in entries if e["best_race_time_ms"] is not None]
    if best_times:
        ax2 = ax.twinx()
        ax2.plot(best_updates, [t / 1000 for t in best_times], color="#E91E63",
                 linewidth=1, linestyle="--", label="best time (s)")
        ax2.set_ylabel("Best time (s)", color="#E91E63")
        ax2.tick_params(axis="y", labelcolor="#E91E63")

    for row in axes:
        for a in row:
            a.xaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_log()

    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"TrackmanAI Training Dashboard\n{log_path.parent.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    print("Dashboard running. Close window or Ctrl+C to stop.")

    try:
        while True:
            entries = load_entries(log_path)
            if entries:
                draw(axes, entries)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            plt.pause(5)
    except KeyboardInterrupt:
        print("\nDashboard closed.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
