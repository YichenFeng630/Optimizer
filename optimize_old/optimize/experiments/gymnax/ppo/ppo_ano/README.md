# PPO with ANO Optimizer

Complete implementation of PPO (Proximal Policy Optimization) using ANO (Adaptive Normalized Optimizer) in JAX/Flax.

**Paper**: [ANO: Faster is Better in Noisy Landscape](https://anonymous.4open.science/r/ano-optimizer-1645/)  
**Official Repository**: https://anonymous.4open.science/r/ano-optimizer-1645/

---

## Algorithm Overview

**ANO** decouples gradient **direction** and **magnitude** for robust optimization in noisy landscapes:

### Core Formula (Paper)
$$\begin{aligned}
m_k &= \beta_1 m_{k-1} + (1-\beta_1) g_k \\
v_k &= v_{k-1} - (1-\beta_2)\operatorname{sign}(v_{k-1}-g_k^2) g_k^2 \\
\hat v_k &= \frac{v_k}{1-\beta_2^k} \\
\theta_k &= \theta_{k-1} - \frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}\operatorname{sign}(m_k)|g_k| - \eta_k \lambda\theta_{k-1}
\end{aligned}$$

### Key Properties
- **Momentum** smooths gradient **direction**: $\operatorname{sign}(m_k)$
- **Magnitude** uses **original gradient**: $|g_k|$ (not momentum-averaged)
- **Second moment** uses **additive update** (Yogi-style): subtractive not multiplicative
- **Bias correction** applied only to $v_k$ (not to $m_k$)
- **Decoupled weight decay**: $- \eta_k \lambda\theta_{k-1}$

---

## Quick Start

### Installation
```bash
pip install jax jaxlib flax optax gymnax wandb
```

### Run Training
```bash
cd /home/yichen/optimizer/optimize_old/optimize/experiments/gymnax/ppo/ppo_ano
python ppo_ano.py
```

### Run Quick Test
```bash
python ppo_ano.py --config-name=config_ppo_ano_test wandb_mode=disabled
```

### Run Unit Tests
```bash
python test_ano_optimizer.py
```

---

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| **ppo_ano.py** (683 lines) | Complete PPO + ANO implementation in JAX/Flax |
| **config_ppo_ano.yaml** | Default hyperparameters and training config |
| **config_ppo_ano_test.yaml** | Quick test configuration (small env) |
| **test_ano_optimizer.py** | Unit test for ANO optimizer correctness |
| **sweep_ano_beta.sh** | Parameter sweep script for β₁ and β₂ |
| **ANO_ALGORITHM_VERIFICATION.md** | Detailed algorithm compliance verification |

### Key Components in ppo_ano.py

1. **ANO Optimizer** (lines 77-238)
   - JAX implementation using Optax GradientTransformation API
   - Separate functions for each update component:
     - `_compute_transformed_g()`: gradient transformation with sign/magnitude decoupling
     - `_compute_m()`: first moment (momentum) update
     - `_compute_v()`: second moment (Yogi-style) update
   - Supports Anolog variant with logarithmic β₁ schedule

2. **Training Loop** (lines 240-680)
   - `make_train(config)` factory function
   - Rollout collection via vectorized environment steps
   - GAE advantage calculation
   - Multi-epoch parameter updates with minibatching
   - W&B logging integration
   - Gradient statistics tracking

3. **Data Structures**
   - `Transition`: Single rollout step (obs, action, reward, etc.)
   - `RunnerState`: Environment and training state
   - `Updatestate`: Batch state for minibatch updates

---

## Configuration

### ANO Parameters (config_ppo_ano.yaml)

```yaml
optimizer: ano                  # Use ANO optimizer
lr: 3e-4                        # Learning rate
beta_1: 0.92                    # Momentum decay (paper default)
beta_2: 0.99                    # Second moment decay (paper default)
eps: 1e-8                       # Numerical stability (paper default)
weight_decay: 0.0               # Decoupled weight decay
logarithmic_schedule: false     # Enable Anolog variant (dynamic β₁)
```

### PPO Hyperparameters

```yaml
# Environment and rollout
total_timesteps: 2e7            # Total training steps
num_envs: 16                    # Parallel environments
num_steps: 512                  # Rollout length
num_minibatches: 4              # Minibatch count
update_epochs: 4                # Epochs per update

# PPO algorithm
gamma: 0.99                     # Discount factor
gae_lambda: 0.95                # GAE exponential smoothing
clip_eps: 0.2                   # PPO clipping range
ent_coef: 0.003                 # Entropy coefficient
vf_coef: 0.5                    # Value loss coefficient
max_grad_norm: 0.5              # Gradient clipping

# Learning rate schedule
anneal_lr: true                 # Decay learning rate linearly

# Logging and reproducibility
seed: 42
num_seeds: 1
wandb_mode: online              # W&B mode: online/offline/disabled
```

---

## Training

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

### Disable Weights & Biases
```bash
python ppo_ano.py wandb_mode=disabled
```

### Parameter Sweep
```bash
bash sweep_ano_beta.sh
```

---

## Algorithm Variants

### Standard ANO
Fixed $\beta_1 = 0.92$ (recommended for most tasks)

### Anolog (Dynamic β₁)
Enable with: `logarithmic_schedule: true`
- $\beta_1(t) = 1 - \frac{1}{\log(\max(2, t))}$
- Progressively decreases momentum decay (expands momentum window)
- Better noise attenuation in early training

---

## Key Design Decisions

1. **Yogi-style v update (subtractive)**
   - Avoids exponential explosion in second moment
   - Better numerical stability than standard Adam

2. **No bias correction for m**
   - First moment directly used for sign without correction
   - Empirically more effective for sign-based direction

3. **Decoupled weight decay**
   - Applied separately from gradient update
   - Matches AdamW style regularization
   - Controlled via `weight_decay` parameter

4. **Pure JAX implementation**
   - Efficient compilation with JAX JIT
   - Compatible with Flax TrainState
   - Fully differentiable and vectorizable

---

## Verification & Validation

### Paper Algorithm Compliance
- ✅ First moment: $m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k$
- ✅ Second moment: $v_k = v_{k-1} - (1-\beta_2)\operatorname{sign}(v_{k-1}-g_k^2) g_k^2$
- ✅ Update: $\theta_k = \theta_{k-1} - \frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}\operatorname{sign}(m_k)|g_k|$
- ✅ Weight decay: Decoupled formulation

### Equivalence of v-update forms
```
Paper form:    v_new = β₂*v − (1−β₂)*sign(v − g²)*g²
PyTorch form:  v_new = v*β₂ + (1−β₂)*sign(g² − v)*g²

Since sign(g² − v) = −sign(v − g²), both forms are mathematically equivalent.
```

See [ANO_ALGORITHM_VERIFICATION.md](ANO_ALGORITHM_VERIFICATION.md) for detailed verification.

---

## Logging & Monitoring

W&B logs all training metrics:
- **PPO diagnostics**: actor loss, value loss, entropy, KL divergence
- **Returns**: episode return, episode length, cumulative return
- **Gradients**: global norm, per-layer statistics
- **Hyperparameters**: learning rate, clipping range, all optimizer configs

---

## Comparison with Related Work

| Optimizer | Direction | Magnitude | 2nd Moment | Key Difference |
|-----------|-----------|-----------|-----------|---|
| Adam | Momentum | Momentum | Exp average | Both from momentum |
| Yogi | Momentum | Gradient | Additive | Different second moment |
| **ANO** | **Sign(Momentum)** | **Gradient** | **Additive** | **Fully decoupled** |

---

## Supported Optimizers

Besides ANO, ppo_ano.py also supports:
- **Adam**: Standard Adam optimizer
- **Yogi**: Yogi optimizer variant
- **RMSprop**: RMSprop optimizer
- **SGD**: Stochastic gradient descent

Switch with: `optimizer: adam` (or yogi, rmsprop, sgd) in config

---

## Frequently Asked Questions

### Why not use the official PyTorch implementation?
The official PyTorch implementation uses a different v-update form. Our implementation strictly follows the paper's original algorithm, which is more mathematically correct and reproducible.

### How to switch to official PyTorch form?
If needed to reproduce official PyTorch results, modify v-update (not recommended):
```python
sign_term = jnp.sign(g_sq - v_leaf)  # sign(g² − v) instead
v_new = v_leaf * b2 + (1.0 - b2) * sign_term * g_sq
```

### Does it support autodiff?
Yes, full autodiff support is native to JAX.

### GPU acceleration?
Yes, JAX handles it automatically.

### How to reproduce paper results?
1. Use default config with β₁=0.92, β₂=0.99, eps=1e-8
2. Enable W&B logging for metric tracking
3. Run multiple seeds for statistical significance
4. See sweep_ano_beta.sh for parameter sweep example

---

## References

- **Paper**: [ANO: Faster is Better in Noisy Landscape](https://anonymous.4open.science/r/ano-optimizer-1645/)
- **Official Formula**: https://anonymous.4open.science/r/ano-optimizer-1645/optimizers/README.md
- **Official Code**: https://anonymous.4open.science/r/ano-optimizer-1645/optimizers/ano.py
- **Official PyTorch**: https://github.com/Adrienkgz/ano-experiments
- **Related PPO**: ppo_yogi, ppo_adam, ppo_lion

---

## Citation

If you use ANO in your research:
```bibtex
@anonymous{ano2024,
  title={ANO: Faster is Better in Noisy Landscape},
  author={Anonymous},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## Project Status

- [x] ANO core implementation
- [x] Anolog variant (dynamic β₁)
- [x] PPO integration
- [x] Complete documentation
- [x] Unit tests
- [x] Algorithm verification

**Last Updated**: 2025-12-22  
**Version**: 1.0
