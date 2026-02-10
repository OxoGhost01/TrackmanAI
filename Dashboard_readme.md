# How to read the dashboard
1. Total Return (top-left, blue)
What it is: Sum of all rewards across all episodes in a single PPO batch.

What to look for: Upward trend = agent is collecting more reward over time.

1. Avg Reward / Step (top-right, green)
What it is: Total return divided by number of steps. This is the best single metric for "is the agent improving?"

What to look for: Steady upward trend. Should go from ~0.1-0.2 (random) towards 0.5+ (good driving).

1. Policy Loss (middle-left, orange)
What it is: How much the PPO clipped surrogate objective is changing the policy.

What to look for: Small values (0.01-0.05) with occasional spikes is healthy. Consistently near 0 means the policy isn't updating at all. Large sustained values (>0.1) mean updates are too aggressive.

1. Value Loss (middle-right, red)
What it is: How far off the value network's predictions are from actual returns (MSE).

What to look for: Should decrease over time as the value network learns to predict returns accurately. High values = the critic has no idea what states are worth.

1. Entropy (bottom-left, purple)
What it is: How random/exploratory the policy is. Max for 3 binary actions = 3 * ln(2) ≈ 2.08 (the dashed line). At max, actions are pure coin flips. At 0, the policy is fully deterministic.

What to look for: Gradual decline from ~2.0 → ~1.0 over hundreds of updates. If it drops too fast (below 1.0 early), the agent committed to a bad strategy. If it stays too high, the agent isn't learning.

1. Race Finish % (bottom-right, teal)
What it is: Percentage of episodes where the agent crossed the finish line.

What to look for: Should start at 0% and gradually climb. Once you see >50%, the agent knows how to complete the track.

## Quick diagnostic checklist for future training runs

Symptom	                                   |                       Likely Cause                      |                    Fix  

Entropy drops fast (<1.0 in <100 updates)  |      Entropy coef too low, or learning rate too high	 |    Increase entropy_coef or decrease lr


Value loss stays huge (>500)               |  	Value network underfitting, or targets too noisy	 |    Decrease lr slightly, or increase batch_episodes


Total return declining                     |	    Policy collapse — agent converged to bad strategy	 |    Fresh start, increase entropy, reduce lr


Avg reward/step flat near 0	               |     Reward function issue or agent not learning at all   | 	 Check reward computation


Finish % stuck at 0                        |  	Agent can't navigate — needs more training time	     |   Be patient; if >500 updates and still 0%, something is broken


Queue full spam	                           |     Learner too slow vs collectors	                     |Install CUDA PyTorch (your main issue), or reduce gpu_collectors_count


## What "trained enough" looks like

Finish % > 80% — the agent reliably completes the track
Best time converging to a plateau — no more improvement in lap time
Entropy stabilized around 0.5-0.8 — confident but still slightly exploratory
Value loss low and stable (<50)
Avg reward/step plateaued at a high value