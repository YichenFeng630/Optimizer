#!/bin/bash
# Simple sweep over learning rate values for Yogi optimizer.
# Add hydra overrides as needed.

set -e

lr_values=(5e-5 1e-4 3e-4 1e-3 3e-3 6e-3 1e-2)

for lr in "${lr_values[@]}"; do
  echo "Running Yogi PPO with lr=${lr}"
  python ppo_yogi.py lr=${lr}
  echo "Completed lr=${lr}"
  echo "----------------------------------------"
  sleep 5
done

echo "All Yogi lr sweeps completed."