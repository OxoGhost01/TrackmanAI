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

import numpy as np
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


def smooth(values, window=20):
    """Simple moving average for noisy data."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def draw(axes, entries):
    if len(entries) < 2:
        return

    updates = [e["update"] for e in entries]

    # --- 1. Total return ---
    ax = axes[0, 0]
    ax.clear()
    raw = [e["total_return"] for e in entries]
    ax.plot(updates, raw, color="#2196F3", linewidth=0.5, alpha=0.3)
    ax.plot(updates, smooth(raw), color="#2196F3", linewidth=2)
    ax.set_title("Total Return (per batch)")
    ax.set_ylabel("Sum of rewards")
    ax.grid(True, alpha=0.3)

    # --- 2. Avg reward per step ---
    ax = axes[0, 1]
    ax.clear()
    raw = [e["avg_reward_per_step"] for e in entries]
    ax.plot(updates, raw, color="#4CAF50", linewidth=0.5, alpha=0.3)
    ax.plot(updates, smooth(raw), color="#4CAF50", linewidth=2)
    ax.set_title("Avg Reward / Step")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # --- 3. Policy loss ---
    ax = axes[1, 0]
    ax.clear()
    raw = [e["policy_loss"] for e in entries]
    ax.plot(updates, raw, color="#FF9800", linewidth=0.5, alpha=0.3)
    ax.plot(updates, smooth(raw), color="#FF9800", linewidth=2)
    ax.set_title("Policy Loss")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # --- 4. Value loss ---
    ax = axes[1, 1]
    ax.clear()
    raw = [e["value_loss"] for e in entries]
    ax.plot(updates, raw, color="#F44336", linewidth=0.5, alpha=0.3)
    ax.plot(updates, smooth(raw), color="#F44336", linewidth=2)
    ax.set_title("Value Loss")
    ax.grid(True, alpha=0.3)

    # --- 5. Entropy ---
    ax = axes[2, 0]
    ax.clear()
    raw = [e["entropy"] for e in entries]
    ax.plot(updates, raw, color="#9C27B0", linewidth=0.5, alpha=0.3)
    ax.plot(updates, smooth(raw), color="#9C27B0", linewidth=2)
    ax.axhline(y=2.08, color="gray", linestyle="--", alpha=0.5, label="max (3*ln2)")
    ax.set_title("Entropy")
    ax.set_ylim(bottom=0, top=2.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 6. Race finish % ---
    ax = axes[2, 1]
    ax.clear()
    finish_pct = [
        100 * e["episodes_finished"] / max(e["episodes_total"], 1) for e in entries
    ]
    ax.bar(updates, finish_pct, color="#009688", alpha=0.4, width=max(1, len(updates) // 200))
    if len(finish_pct) >= 20:
        ax.plot(updates, smooth(finish_pct), color="#009688", linewidth=2)
    ax.set_title("Race Finish %")
    ax.set_ylabel("%")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Add best time as text annotation
    best_times = [e["best_race_time_ms"] for e in entries if e["best_race_time_ms"] is not None]
    if best_times:
        overall_best = min(best_times) / 1000
        recent_best = min(e["best_race_time_ms"] for e in entries[-50:] if e["best_race_time_ms"] is not None) / 1000 if any(e["best_race_time_ms"] is not None for e in entries[-50:]) else None
        text = f"Best: {overall_best:.1f}s"
        if recent_best and recent_best != overall_best:
            text += f"\nRecent: {recent_best:.1f}s"
        ax.text(0.98, 0.95, text, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color="#E91E63", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E91E63", alpha=0.9))

    total_finishes = sum(e["episodes_finished"] for e in entries)
    total_eps = sum(e["episodes_total"] for e in entries)
    ax.text(0.02, 0.95, f"Total: {total_finishes}/{total_eps}", transform=ax.transAxes,
            fontsize=9, color="#666", ha="left", va="top")

    for row in axes:
        for a in row:
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
            a.set_xlabel("Update", fontsize=8)


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
