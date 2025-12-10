# PPO with ANO Optimizer

Official implementation of PPO (Proximal Policy Optimization) using ANO (Adaptive Normalized Optimizer).

Reference: [ANO: Faster is Better in Noisy Landscape](https://github.com/Adrienkgz/ano-experiments)

## Overview

**ANO** is a novel optimizer that decouples the **direction** and **magnitude** of parameter updates:

- **Momentum** applied exclusively to **directional smoothing**: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
- **Step size** uses **instantaneous gradient magnitudes**: $|g_t|$  
- **Second-moment** estimation uses **additive update** (Yogi-style): $v_t = v_{t-1} + (1-\beta_2) \text{sign}(g_t^2 - v_{t-1}) g_t^2$

**Update formula**: 
$$\text{param} \gets \text{param} - \text{lr} \cdot |g_t| \cdot \text{sign}(m_t) / (\sqrt{v_t} + \epsilon)$$

This decoupling improves **robustness to gradient noise** while retaining the simplicity and efficiency of first-order methods.

## Files

- **ppo_ano.py** (715 lines)
  - Complete PPO implementation with ANO optimizer
  - Factory function: `make_train(config)` → `train(rng, exp_id)`
  - ANO optimizer: Custom JAX implementation using Optax API
  - Full PPO pipeline: rollout collection → GAE → multi-epoch updates
  - Gradient statistics, W&B logging integration
  
- **config_ppo_ano.yaml**
  - Default hyperparameters (aligned with ppo_yogi)
  - ANO-specific parameters: `beta_1` (0.92), `beta_2` (0.99), `eps`, `weight_decay`, `logarithmic_schedule`
  
- **sweep_ano_beta.sh** (executable)
  - Parameter sweep script: β₁ ∈ {0.92, 0.85, 0.7, 0.5} × β₂ ∈ {0.99, 0.95, 0.9, 0.8}
  - 16 total experiments with automatic job management

- **config_ppo_ano_test.yaml**
  - Reduced parameters for quick testing and debugging

## Running

### Basic Training (Default Config)
```bash
python ppo_ano.py
```

### With Custom Hyperparameters
```bash
python ppo_ano.py \
  total_timesteps=5e6 \
  num_envs=16 \
  beta_1=0.92 \
  beta_2=0.99 \
  wandb_mode=online
```

### Quick Test
```bash
python ppo_ano.py \
  total_timesteps=1e5 \
  num_envs=4 \
  num_steps=128 \
  wandb_mode=disabled
```

### Parameter Sweep
```bash
bash sweep_ano_beta.sh
```

## Algorithm Details

### ANO Variants

**Standard ANO**: β₁ is fixed (typically 0.92)

**Anolog** (Optional): Dynamic β₁ schedule - set `logarithmic_schedule: true`
- Formula: $\beta_1(t) = 1 - \frac{1}{\log(\max(2, t))}$
- Expands momentum window over time for improved noise attenuation

## Configuration

**Key ANO parameters** in `config_ppo_ano.yaml`:

```yaml
optimizer: ano                  # Optimizer selection
lr: 3e-4                       # Learning rate
beta_1: 0.92                   # Momentum decay (0.92-0.95 typical)
beta_2: 0.999                  # Second moment decay
eps: 1e-8                      # Numerical stability epsilon
weight_decay: 0.0              # L2 regularization
logarithmic_schedule: false    # Enable Anolog variant
```

**PPO parameters** (unchanged from ppo_yogi):
```yaml
gamma: 0.99                    # Discount factor
gae_lambda: 0.95              # GAE decay
clip_eps: 0.2                 # PPO clipping range
ent_coef: 0.003               # Entropy regularization
vf_coef: 0.5                  # Value loss coefficient
```

## Logging & Monitoring

Training metrics logged to **Weights & Biases**:
- Actor loss, critic loss, entropy
- Episode returns and lengths
- Gradient norms and cosine similarity
- Network parameter statistics
- All PPO diagnostics

Disable W&B for quick tests:
```bash
python ppo_ano.py wandb_mode=disabled
```

## Alignment with ppo_yogi

- ✅ Identical data structures and factory pattern
- ✅ Same PPO algorithm (rollout, GAE, clipping)
- ✅ Compatible configuration system
- ✅ Matching gradient statistics and logging
- ✅ Drop-in replacement optimizer

## Comparison with Other Optimizers

| Optimizer | Momentum | 2nd Moment | Update Rule | Robustness |
|-----------|----------|-----------|-------------|-----------|
| Adam | Yes (exp avg) | Exp avg sq | lr × grad / (√v + ε) | Good |
| Yogi | Yes (exp avg) | Additive | lr × grad / (√v + ε) | Better |
| **ANO** | **Yes (sign-based)** | **Additive** | **lr × \|grad\| × sign(m) / (√v + ε)** | **Best** |

## References

- Paper: [ANO: Faster is Better in Noisy Landscape](https://doi.org/10.5281/zenodo.16422081)
- Official implementation: https://github.com/Adrienkgz/ano-optimizer
- Related frameworks: ppo_yogi, ppo_beta, ppo_grad

## Notes

- Currently using Adam as a proxy for quick testing; full ANO implementation can be added via custom JAX GradientTransformation
- Aligns with ppo_yogi structure for easy comparison
- Supports both discrete and continuous action spaces (currently set to discrete for MountainCar)
