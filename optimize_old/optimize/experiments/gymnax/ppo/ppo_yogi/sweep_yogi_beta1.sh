#!/bin/bash
# Sweep over beta_1 values for Yogi optimizer.
# Fixed eps=1e-4 and beta_2=0.999

set -e

beta1_values=(0.8 0.9 0.95 0.98)
eps=1e-4
beta2=0.999

for beta1 in "${beta1_values[@]}"; do
  echo "Running Yogi PPO with beta_1=${beta1}, eps=${eps}, beta_2=${beta2}"
  python ppo_yogi.py beta_1=${beta1} eps=${eps} beta_2=${beta2}
  echo "Completed beta_1=${beta1}"
  echo "----------------------------------------"
  sleep 5
done

echo "All Yogi beta_1 sweeps completed."