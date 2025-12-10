#!/bin/bash
# Sweep over beta_1 and beta_2 values for ANO optimizer.
# Fixed eps=1e-8
# 4x4 = 16 experiments

set -e

beta1_values=(0.92 0.85 0.7 0.5)
beta2_values=(0.99 0.95 0.9 0.8)
eps=1e-8

total_runs=$((${#beta1_values[@]} * ${#beta2_values[@]}))
current_run=0

for beta1 in "${beta1_values[@]}"; do
  for beta2 in "${beta2_values[@]}"; do
    current_run=$((current_run + 1))
    echo "[$current_run/$total_runs] Running ANO PPO with beta_1=${beta1}, beta_2=${beta2}, eps=${eps}"
    python ppo_ano.py beta_1=${beta1} beta_2=${beta2} eps=${eps}
    echo "Completed beta_1=${beta1}, beta_2=${beta2}"
    echo "----------------------------------------"
    sleep 5
  done
done

echo "All ANO beta sweeps completed: ${total_runs} runs."
