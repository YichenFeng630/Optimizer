#!/bin/bash

# Activate conda environment
# conda activate adam

# Array of beta_1 values to test
#beta_1_values=(0.2 0.9 0.999)
beta_2_values=(0.90  0.95  0.97  0.99  0.999)

# # Loop through each beta_1 value
# for beta_1 in "${beta_1_values[@]}"; do
#     echo "Running PPO with beta_1 = $beta_1"
    
#     # Run the PPO training script with the current beta_1 value
#     python ppo_discrete.py beta_1=$beta_1
    
#     echo "Completed run with beta_1 = $beta_1"
#     echo "----------------------------------------"
#     sleep 10  # sleep for 1 minute
# done

# # Loop through each beta_2 value
for beta_2 in "${beta_2_values[@]}"; do
    echo "Running PPO with beta_2 = $beta_2"
    
    # Run the PPO training script with the current beta_2 value
    python ppo_discrete.py beta_2=$beta_2
    
    echo "Completed run with beta_2 = $beta_2"
    echo "----------------------------------------"
    sleep 10  # sleep for 1 minute
done

echo "All beta_2 sweeps completed!"
