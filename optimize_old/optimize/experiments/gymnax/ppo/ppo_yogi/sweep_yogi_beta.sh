#!/bin/bash
# Sweep over beta_1 and beta_2 values for Yogi optimizer.
# Fixed eps=1e-4
# 4x4 = 16 experiments

set -e

beta1_values=(0.9 0.7 0.5 0.3)
beta2_values=(0.5 0.8 0.99 0.995)
eps=1e-4

total_runs=$((${#beta1_values[@]} * ${#beta2_values[@]}))
current_run=0

for beta1 in "${beta1_values[@]}"; do
  for beta2 in "${beta2_values[@]}"; do
    current_run=$((current_run + 1))
    echo "[$current_run/$total_runs] Running Yogi PPO with beta_1=${beta1}, beta_2=${beta2}, eps=${eps}"
    python ppo_yogi.py beta_1=${beta1} beta_2=${beta2} eps=${eps}
    echo "Completed beta_1=${beta1}, beta_2=${beta2}"
    echo "----------------------------------------"
    sleep 5
  done
done

echo "All Yogi beta sweeps completed: ${total_runs} runs."