#!/bin/bash
# Simple sweep over num_minibatches to change batch_size.
# num_steps=128, num_envs=16 are fixed in your YAML.

set -e

minibatches=(2 4 8 16 32)

for mb in "${minibatches[@]}"; do
  echo "Running Yogi PPO with num_minibatches=${mb}"
  python ppo_yogi.py num_minibatches=${mb} +batch_size_label=${mb}
  echo "Completed num_minibatches=${mb}"
  echo "----------------------------------------"
  sleep 3
done

echo "All Yogi batch-size sweeps completed."
