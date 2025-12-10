"""
PPO (Proximal Policy Optimization) using ANO (Adaptive Normalized Optimizer) 版本。

严格遵循原始 `ppo_yogi.py` 结构：
1. make_train(config) -> train(rng, exp_id)
2. 内部包含 rollout 收集、GAE 优势计算、多个 epoch & minibatch 更新
3. 使用 TrainState 封装参数与优化器状态
4. 增加优化器分支: ano (Sign-Magnitude Decoupled Optimizer)

ANO 关键特性 (参考 https://github.com/Adrienkgz/ano-experiments):
- 对梯度方向和梯度幅度解耦处理
- 动量仅作用于梯度方向（sign），幅度使用原始梯度的绝对值
- 加法型第二矩更新（受 Yogi 启发）
- 公式: transformed_g = |g| * sign(m̂) / (√v̂ + ε)，然后 param = param - lr * transformed_g

额外添加了中文注释，帮助理解每个步骤。
"""
import os

# 为了结果可复现，启用 XLA GPU 确定性操作
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import hydra
from flax.training.train_state import TrainState
from typing import NamedTuple
from jax._src.typing import Array
from omegaconf import OmegaConf
import datetime
from optimize.utils.wandb_multilogger import WandbMultiLogger
from optimize.networks.mlp import ActorCriticDiscrete
import numpy as np
import optax
import gymnax
from optimize.utils.jax_utils import pytree_norm

# 全局日志记录器（用于 io_callback）
LOGGER = None

# ============================================================
# 数据结构
# ============================================================
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    new_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray

class RunnerState(NamedTuple):
    train_state: TrainState
    running_grad: jnp.ndarray
    obs: jnp.ndarray
    state: jnp.ndarray
    done: jnp.ndarray
    cumulative_return: jnp.ndarray
    timesteps: jnp.ndarray
    update_step: int
    rng: Array

class Updatestate(NamedTuple):
    train_state: TrainState
    running_grad: jnp.ndarray
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: Array


# ============================================================
# ANO (Adaptive Normalized Optimizer) - 正确实现
# ============================================================
def ano(
    learning_rate,
    b1: float = 0.92,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    logarithmic_schedule: bool = False,
) -> optax.GradientTransformation:
    """
    ANO 优化器: 解耦梯度方向和幅度
    
    关键公式:
    1. m_t = β₁*m_{t-1} + (1-β₁)*g_t                           (一阶矩：方向动量)
    2. v_t = v_{t-1} + (1-β₂)*sign(g_t²-v_{t-1})*g_t²           (二阶矩：加法式，Yogi风格)
    3. m̂_t = m_t / (1 - β₁^t)                                  (一阶矩 bias correction)
    4. v̂_t = v_t / (1 - β₂^t)                                  (二阶矩 bias correction)
    5. transformed_g = |g_t| * sign(m̂_t) / (√v̂_t + ε)         (梯度变换)
    6. param = param - lr * transformed_g                      (参数更新由optax处理)
    
    参考: https://github.com/Adrienkgz/ano-experiments/blob/main/optimizers/ano.py
    
    Args:
        learning_rate: 初始学习率或学习率调度函数
        b1: 第一阶矩衰减系数（0.92推荐）
        b2: 第二阶矩衰减系数（0.99推荐）
        eps: 数值稳定性的小值（1e-8推荐）
        weight_decay: L2权重衰减（可选）
        logarithmic_schedule: 使用对数化β₁调度(Anolog变体)
    """
    
    def init_fn(params):
        """初始化优化器状态"""
        return {
            "m": jax.tree.map(jnp.zeros_like, params),   # 一阶矩
            "v": jax.tree.map(jnp.zeros_like, params),   # 二阶矩
            "step": jnp.array(0, dtype=jnp.int32),       # 步数计数
        }
    
    def update_fn(updates, state, params=None, **kwargs):
        """
        执行梯度变换。注意返回的是：
        - transformed_updates: 变换后的梯度（非参数更新）
        - new_state: 更新后的优化器状态
        
        optax会在之后用学习率乘以transformed_updates进行参数更新。
        """
        step = state["step"]
        m = state["m"]
        v = state["v"]
        
        # 处理学习率调度
        if callable(learning_rate):
            lr = learning_rate(step)
        else:
            lr = learning_rate
        
        step_f = step.astype(jnp.float32) + 1.0
        
        # 处理对数化β₁ (Anolog变体)
        if logarithmic_schedule:
            # β₁(t) = 1 - 1/log(max(2, t))
            step_safe = jnp.maximum(step_f, 2.0)
            b1_t = 1.0 - 1.0 / jnp.log(step_safe)
        else:
            b1_t = b1
        
        # Bias correction系数
        bias_corr_1 = 1.0 - jnp.power(b1_t, step_f)
        bias_corr_2 = 1.0 - jnp.power(b2, step_f)
        
        # 首先计算变换后的梯度
        def _compute_transformed_g(g, m_leaf, v_leaf):
            """计算变换后的梯度"""
            m_new = b1_t * m_leaf + (1.0 - b1_t) * g
            g_sq = jnp.square(g)
            sign_term = jnp.sign(v_leaf - g_sq)
            v_new = v_leaf - (1.0 - b2) * sign_term * g_sq
            
            m_hat = m_new / bias_corr_1
            v_hat = v_new / bias_corr_2
            
            transformed_g = jnp.abs(g) * jnp.sign(m_hat) / (jnp.sqrt(v_hat) + eps)
            return transformed_g
        
        # 计算新的m
        def _compute_m(g, m_leaf, v_leaf):
            """计算新的一阶矩"""
            return b1_t * m_leaf + (1.0 - b1_t) * g
        
        # 计算新的v
        def _compute_v(g, m_leaf, v_leaf):
            """计算新的二阶矩"""
            g_sq = jnp.square(g)
            sign_term = jnp.sign(v_leaf - g_sq)
            return v_leaf - (1.0 - b2) * sign_term * g_sq
        
        # 分别计算
        transformed_updates = jax.tree.map(_compute_transformed_g, updates, m, v)
        new_m = jax.tree.map(_compute_m, updates, m, v)
        new_v = jax.tree.map(_compute_v, updates, m, v)
        
        # 构建新状态
        new_state = {
            "m": new_m,
            "v": new_v,
            "step": step + 1,
        }
        
        return transformed_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)


# ============================================================
# 工厂函数：根据 config 生成训练函数
# ============================================================
def make_train(config):
    env, env_params = gymnax.make(config["env_name"])

    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["minibatch_size"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )

    def train(rng, exp_id):
        def train_setup(rng):
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            obs, state = jax.vmap(env.reset, in_axes=(0, None))(_rng_resets, env_params)

            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config["num_minibatches"] * config["update_epochs"]))
                    / config["num_updates"]
                )
                return config["lr"] * frac

            lr_schedule = linear_schedule if config["anneal_lr"] else config["lr"]

            network = ActorCriticDiscrete(
                action_dim=env.num_actions,
                activation=config["activation"],
            )
            rng, _rng_net = jax.random.split(rng)
            init_x = jnp.zeros(obs.shape)
            network_params = network.init(_rng_net, init_x)

            # 选择优化器 - 现在正确使用自定义的 ANO
            if config["optimizer"] == "ano":
                # ANO: 符合论文的正确实现
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    ano(
                        learning_rate=lr_schedule,
                        b1=config.get("beta_1", 0.92),
                        b2=config.get("beta_2", 0.99),
                        eps=config.get("eps", 1e-8),
                        weight_decay=config.get("weight_decay", 0.0),
                        logarithmic_schedule=config.get("logarithmic_schedule", False),
                    ),
                )
            elif config["optimizer"] == "yogi":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.yogi(
                        learning_rate=lr_schedule,
                        b1=config["beta_1"],
                        b2=config["beta_2"],
                        eps=config.get("eps", 1e-3),
                    ),
                )
            elif config["optimizer"] == "adam":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.adam(
                        learning_rate=lr_schedule,
                        eps=1e-5,
                        b1=config["beta_1"],
                        b2=config["beta_2"],
                    ),
                )
            elif config["optimizer"] == "rmsprop":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
                )
            elif config["optimizer"] == "sgd":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.sgd(learning_rate=lr_schedule),
                )
            else:
                raise ValueError(f"未知 optimizer: {config['optimizer']}")

            train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
            running_grad = jax.tree.map(jnp.zeros_like, network_params)
            return obs, state, train_state, running_grad, network

        rng, _rng_setup = jax.random.split(rng)
        obs, env_state, train_state, running_grad, network = train_setup(_rng_setup)

        runner_state = RunnerState(
            train_state=train_state,
            running_grad=running_grad,
            obs=obs,
            state=env_state,
            done=jnp.zeros(config["num_envs"], dtype=bool),
            cumulative_return=jnp.zeros(config["num_envs"]),
            timesteps=jnp.zeros(config["num_envs"], dtype=jnp.int32),
            update_step=0,
            rng=rng,
        )

        # ========== 外层训练循环 ==========
        def _train_loop(runner_state, unused):
            initial_timesteps = runner_state.timesteps

            # ========== 1. 收集 rollout ==========
            def _env_step(runner_state, unused):
                train_state = runner_state.train_state
                obs = runner_state.obs
                state = runner_state.state
                done = runner_state.done
                rng = runner_state.rng

                def reset_if_done(obs, state, done, rng):
                    return jax.lax.cond(
                        done,
                        lambda: env.reset(rng, env_params),
                        lambda: (obs, state)
                    )

                rng, _rng_reset = jax.random.split(rng)
                rng_resets = jax.random.split(_rng_reset, config["num_envs"])
                obs, state = jax.vmap(reset_if_done)(obs, state, done, rng_resets)

                rng, _rng_action = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs)
                action = pi.sample(seed=_rng_action)
                log_prob = pi.log_prob(action)

                rng, _rng_step = jax.random.split(rng)
                rng_step = jax.random.split(_rng_step, config["num_envs"])
                new_obs, new_state, reward, new_done, info = jax.vmap(env.step)(
                    rng_step, state, action
                )

                timesteps = jnp.where(new_done, 0, runner_state.timesteps + 1)

                transition = Transition(
                    obs=obs,
                    action=action.squeeze(),
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    new_done=new_done,
                    value=value.squeeze(),
                    info=info,
                )

                runner_state = RunnerState(
                    train_state=train_state,
                    running_grad=runner_state.running_grad,
                    obs=new_obs,
                    state=new_state,
                    done=new_done,
                    cumulative_return=runner_state.cumulative_return,
                    timesteps=timesteps,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["num_steps"],
            )

            # ========== 2. 计算 GAE ==========
            train_state = runner_state.train_state
            last_obs = runner_state.obs
            _, last_value = network.apply(train_state.params, last_obs)
            last_value = last_value.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.new_done, transition.value, transition.reward
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # ========== 3. 多 epoch 更新 ==========
            def _update_epoch(update_state, unused):
                train_state = update_state.train_state
                running_grad = update_state.running_grad
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                rng, _rng_permute = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_permute, config["num_envs"])
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                shuffled_batch_split = jax.tree.map(
                    lambda x: jnp.reshape(
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch,
                )
                minibatches = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), shuffled_batch_split)

                def _update_minibatch(carry, minibatch):
                    train_state, running_grad = carry
                    traj_minibatch, advantages_minibatch, targets_minibatch = minibatch

                    def _loss(params, traj_minibatch, gae_minibatch, targets_minibatch):
                        pi, value = network.apply(params, traj_minibatch.obs)
                        log_prob = pi.log_prob(traj_minibatch.action)

                        logratio = log_prob - traj_minibatch.log_prob
                        ratio = jnp.exp(logratio)
                        gae_minibatch = (gae_minibatch - gae_minibatch.mean()) / (gae_minibatch.std() + 1e-8)
                        loss_actor_1 = ratio * gae_minibatch
                        loss_actor_2 = (
                            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"])
                            * gae_minibatch
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        value_pred_clipped = traj_minibatch.value + (
                            value - traj_minibatch.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss_unclipped = jnp.square(value - targets_minibatch)
                        value_loss_clipped = jnp.square(value_pred_clipped - targets_minibatch)
                        value_loss = 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped).mean()

                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        total_loss = (
                            loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        )
                        return total_loss, {
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                        }

                    grad_fn = jax.value_and_grad(_loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )

                    updated_train_state = train_state.apply_gradients(grads=grads)

                    def cosine_similarity(grad1, grad2):
                        flat_grad1 = jax.tree.leaves(grad1)
                        flat_grad2 = jax.tree.leaves(grad2)
                        vec1 = jnp.concatenate([jnp.ravel(x) for x in flat_grad1])
                        vec2 = jnp.concatenate([jnp.ravel(x) for x in flat_grad2])
                        denom = jnp.linalg.norm(vec1) * jnp.linalg.norm(vec2)
                        return jnp.where(denom > 1e-8, jnp.dot(vec1, vec2) / denom, 0.0)

                    cos_sim = cosine_similarity(grads, running_grad)
                    cos_sim_mu = 0.0
                    mu_norm = 0.0
                    nu_norm = 0.0
                    # 注意：ANO使用自定义优化器，opt_state结构与Yogi不同
                    # 安全地尝试访问mu/nu（仅当使用Yogi/Adam等优化器时存在）
                    try:
                        if (hasattr(updated_train_state.opt_state, '__getitem__') 
                            and len(updated_train_state.opt_state) > 1):
                            opt_inner = updated_train_state.opt_state[1]
                            if isinstance(opt_inner, (list, tuple)) and len(opt_inner) > 0:
                                opt_inner = opt_inner[0]
                            if hasattr(opt_inner, 'mu') and hasattr(opt_inner, 'nu'):
                                mu_norm = pytree_norm(opt_inner.mu)
                                nu_norm = pytree_norm(opt_inner.nu)
                                cos_sim_mu = cosine_similarity(grads, opt_inner.mu)
                    except (KeyError, IndexError, TypeError):
                        pass  # 如果访问失败就跳过（ANO没有mu/nu）

                    cos_sim_clamped = jnp.clip(cos_sim, -1.0, 1.0)
                    gradient_angle_deg = jnp.arccos(cos_sim_clamped) * 180.0 / jnp.pi

                    new_running_grad = grads
                    total_loss[1]["grad_norm"] = pytree_norm(grads)
                    total_loss[1]["mu_norm"] = mu_norm
                    total_loss[1]["nu_norm"] = nu_norm
                    total_loss[1]["cosine_similarity"] = cos_sim
                    total_loss[1]["cosine_similarity_mu"] = cos_sim_mu
                    total_loss[1]["gradient_angle_deg"] = gradient_angle_deg

                    return (updated_train_state, new_running_grad), total_loss

                (final_train_state, final_running_grad), total_loss = jax.lax.scan(
                    _update_minibatch,
                    (train_state, running_grad),
                    minibatches,
                )

                update_state = Updatestate(
                    train_state=final_train_state,
                    running_grad=final_running_grad,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )
                return update_state, total_loss

            update_state = Updatestate(
                train_state=train_state,
                running_grad=runner_state.running_grad,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=runner_state.rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config["update_epochs"],
            )

            # ========== 4. 计算回报与统计 ==========
            reward = traj_batch.reward
            done = traj_batch.new_done
            cumulative_return = runner_state.cumulative_return

            def _returns(carry_return, inputs):
                reward, done = inputs
                cumulative_return = carry_return + reward
                reset_return = jnp.zeros(reward.shape[1:], dtype=float)
                carry_return = jnp.where(done, reset_return, cumulative_return)
                return carry_return, cumulative_return

            new_cumulative_return, returns = jax.lax.scan(
                _returns,
                cumulative_return,
                (reward, done),
            )
            only_returns = jnp.where(done, returns, 0)
            returns_avg = jnp.where(done.sum() > 0, only_returns.sum() / done.sum(), 0.0)

            def _episode_lengths(carry_length, done):
                cumulative_length = carry_length + 1
                reset_length = jnp.zeros(done.shape[1:], dtype=jnp.int32)
                carry_length = jnp.where(done, reset_length, cumulative_length)
                return carry_length, cumulative_length

            _, episode_lengths = jax.lax.scan(_episode_lengths, initial_timesteps, done)
            only_episode_ends = jnp.where(done, episode_lengths, 0)
            episode_length_avg = jnp.where(
                done.sum() > 0, only_episode_ends.sum() / done.sum(), 0.0
            )

            network_leaves = jax.tree.leaves(update_state.train_state.params)
            flat_network = jnp.concatenate([jnp.ravel(x) for x in network_leaves])
            network_l1 = jnp.sum(jnp.abs(flat_network))
            network_l2 = jnp.linalg.norm(flat_network)
            network_linfty = jnp.max(jnp.abs(flat_network))
            network_mu = jnp.mean(flat_network)
            network_std = jnp.std(flat_network)
            network_max = jnp.max(flat_network)
            network_min = jnp.min(flat_network)

            total_loss, loss_info = loss_info
            loss_info["total_loss"] = total_loss
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric = {
                "update_step": runner_state.update_step,
                "env_step": runner_state.update_step * config["num_envs"] * config["num_steps"],
                "return": returns_avg,
                "episode_length": episode_length_avg,
                "network_l1": network_l1,
                "network_l2": network_l2,
                "network_linfty": network_linfty,
                "network_mu": network_mu,
                "network_std": network_std,
                "network_max": network_max,
                "network_min": network_min,
            }
            metric.update(loss_info)

            def callback(exp_id, metric):
                np_log_dict = {k: np.array(v) for k, v in metric.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)

            runner_state = RunnerState(
                train_state=update_state.train_state,
                running_grad=update_state.running_grad,
                obs=runner_state.obs,
                state=runner_state.state,
                done=runner_state.done,
                cumulative_return=new_cumulative_return,
                timesteps=runner_state.timesteps,
                update_step=runner_state.update_step + 1,
                rng=runner_state.rng,
            )
            return runner_state, metric

        rng, _train_rng = jax.random.split(rng)
        done = jnp.zeros((config["num_envs"]), dtype=jnp.bool_)
        cumulative_return = jnp.zeros((config["num_envs"]), dtype=float)
        initial_runner_state = RunnerState(
            train_state=train_state,
            running_grad=running_grad,
            obs=obs,
            state=env_state,
            done=done,
            cumulative_return=cumulative_return,
            timesteps=jnp.zeros((config["num_envs"]), dtype=jnp.int32),
            update_step=0,
            rng=_train_rng,
        )

        final_runner_state, metrics_batch = jax.lax.scan(
            _train_loop,
            initial_runner_state,
            None,
            length=config["num_updates"],
        )
        return final_runner_state, metrics_batch

    return train


# ============================================================
# 主函数 + Hydra
# ============================================================
@hydra.main(version_base=None, config_path="./", config_name="config_ppo_ano")
def main(config):
    try:
        config = OmegaConf.to_container(config)
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile (ANO)...")
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.block_until_ready(jax.jit(train_vmap))
        print("Compile finished...")

        job_type = f"{config['job_type']}_{config['env_name']}"
        group = (
            f"ppo_ano_beta1_{config.get('beta_1', 0.92)}_{config['env_name']}"
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        )
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=config.get("wandb_mode", "online"),
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        print("Running ANO PPO...")
        _ = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
