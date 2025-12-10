# PPO Adam Baseline Test

## ppo_yogi_adam_v0 
PPO_Adam_Mountaincarv0 Baseline

``` yaml
total_timesteps: 4e7        # Total training timesteps
num_envs: 16                # Number of parallel environments
num_steps: 4000              # Steps per rollout
num_minibatches: 4          # Number of minibatches per epoch
update_epochs: 64           # Number of epochs per update (reduced from 64 to 16)


optimizer: adam             # Optimizer choice: yogi, adam, rmsprop, sgd
lr: 4e-3                  # Learning rate (tested range: 4e-3 to 1e-3)
anneal_lr: true             # Whether to use learning rate annealing
max_grad_norm: 1.0          # Gradient clipping threshold

beta_1: 0.9                 # First moment decay coefficient (momentum)
beta_2: 0.999               # Second moment decay coefficient
eps: 1e-8                   # Epsilon value (outside sqrt for Yogi, 1e-6 best from sweep)

gamma: 0.99                 # Discount factor
gae_lambda: 0.95            # GAE lambda parameter
clip_eps: 0.2               # PPO clipping parameter
scale_clip_eps: false       # Whether to dynamically scale clip_eps
ent_coef: 0.003             # Entropy regularization coefficient 0.003 # 0.01 for Pendulum-v1
vf_coef: 0.5                # Value function loss coefficient

activation: ReLU             # Activation function
fc_dim_size: 64           # Fully connected layer dimension

env_name: MountainCar-v0  # Gymnax environment name (Pendulum-v1/MountainCar-v0)
seed: 0                     # Random seed
num_seeds: 10               # Number of parallel seeds to run
```



## ppo_yogi_adam_v1 
update_epochs（64 → 16）
PPO works! 

## ppo_yogi_adam_v2 
num_steps（4000 → 2048）
PPO also works, and even works better!

## ppo_yogi_adam_v3 
num_steps（2048 → 1024）
also works, but more noisy than v2

## ppo_yogi_adam_v4
num_steps（2048）
lr = 4e-3 → 3e-4


# PPO_yogi_test

## ppo_yogi_v0

parameters:
``` yaml
optimizer: adam --> yogi
```
Current Yogi runs suffer from an effective learning-rate mismatch.
According to the Yogi paper, Yogi typically requires a **learning rate 5–10× larger than Adam**, and the Adaptive-Inertia paper shows that stability depends mainly on the ratio lr / sqrt(v).


But We cannot simply set lr_yogi = lr_adam × 5.
The “×5–10” rule from the Yogi paper applies to large-scale low-noise CV tasks.

In PPO + MountainCar-v0, the effective learning rate depends heavily on the variance of v_t, as described in the Adaptive Inertia paper.

Therefore lr scaling must be empirically tuned.
We will sweep lr in the range [4e-3, 8e-3, 1e-2, 2e-2, 4e-2], which is the paper-guided but task-adjusted search space.


## ppo_yogi_sweep_v1 
lr sweeps for yogi

parameters:
``` yaml
lr = [4e-3, 8e-3, 1e-2, 2e-2, 4e-2]
```

Adam (lr=0.004) is the only optimizer that fully solves the task, reaching ~–120 return.

All Yogi variants remain around –190 ~ –200, which means:
they have not learned the task
only a few learning rates (e.g., 0.008, 0.02) show slight improvement, but still far from solving it

Increasing the Yogi learning rate does not produce a smooth improvement.
Instead, performance jumps irregularly → strongly non-monotonic behavior.



## ppo_yogi_sweep_v2 
Fix LR = 4e-3 

parameters:
``` yaml
eps_values=(1e-4 5e-4 1e-3 2e-3)
```

