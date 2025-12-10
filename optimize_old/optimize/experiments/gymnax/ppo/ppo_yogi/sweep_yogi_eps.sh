#!/bin/bash
# Simple sweep over eps values for Yogi optimizer.
# Add hydra overrides as needed.

set -e

eps_values=(2e-3 1e-2 5e-2)

for eps in "${eps_values[@]}"; do
  echo "Running Yogi PPO with eps=${eps}"
  python ppo_yogi.py eps=${eps}
  echo "Completed eps=${eps}"
  echo "----------------------------------------"
  sleep 5
done

echo "All Yogi eps sweeps completed."