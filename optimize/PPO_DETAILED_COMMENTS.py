"""
PPO (Proximal Policy Optimization) 强化学习算法的完整实现（带详细中文注释）

这个脚本是整个项目的核心。它实现了：
1. Actor-Critic 神经网络的训练循环
2. PPO 算法的损失函数和梯度更新
3. 多种子并行训练
4. Wandb 日志记录

执行方式：
    python ppo_discrete.py                          # 使用默认配置
    python ppo_discrete.py beta_1=0.8               # 覆盖 beta_1
    python ppo_discrete.py total_timesteps=100000   # 快速测试
    PYTHONPATH=/path/to/project python ppo_discrete.py  # 指定包路径
"""

# ============================================================================
# 第一部分：导入库
# ============================================================================

import os
# 禁用 XLA 随机性，使实验可重复
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import hydra  # 配置管理库
from flax.training.train_state import TrainState  # 训练状态容器
from typing import NamedTuple, Dict  # 类型提示
from jax._src.typing import Array
from omegaconf import OmegaConf  # Omega Conf 配置转换
import datetime
from optimize.utils.wandb_multilogger import WandbMultiLogger  # 多进程 wandb
from optimize.networks.mlp import ActorCriticDiscrete  # Actor-Critic 网络
import numpy as np
import optax  # 优化器库（包含 Adam）
import gymnax  # JAX 版本的 Gym 环境
from optimize.utils.jax_utils import pytree_norm, jprint  # 工具函数

# ============================================================================
# 第二部分：数据结构定义（NamedTuple）
# ============================================================================

class Transition(NamedTuple):
    """
    存储一个时间步的经验数据
    
    用途：在与环境交互时，记录所有必要的信息
    形状：(time_steps, num_envs, ...)
    """
    obs: jnp.ndarray              # 观测（状态），如游戏画面
    action: jnp.ndarray           # 采取的动作
    log_prob: jnp.ndarray         # 该动作在当前策略下的对数概率
    reward: jnp.ndarray           # 环境给予的奖励
    done: jnp.ndarray             # 游戏是否结束（当前步之前）
    new_done: jnp.ndarray         # 游戏是否结束（当前步之后）
    value: jnp.ndarray            # Critic 对当前状态的估计价值
    info: jnp.ndarray             # 额外信息（通常未使用）


class RunnerState(NamedTuple):
    """
    训练循环的完整状态（jax.lax.scan 的 carry）
    
    在 jax.lax.scan 中，这个状态会在每个时间步被更新一次
    """
    train_state: TrainState        # Flax 的训练状态（参数 + 优化器状态）
    running_grad: jnp.ndarray      # 运行中的梯度（用于梯度相似度分析）
    obs: jnp.ndarray               # 当前观测
    state: jnp.ndarray             # 环境内部状态
    done: jnp.ndarray              # 每个环境是否已结束
    cumulative_return: jnp.ndarray # 累积回报（用于计算每集的总奖励）
    timesteps: jnp.ndarray         # 每个环境的时间步计数
    update_step: int               # 总的更新步数
    rng: Array                     # 随机数生成器


class Updatestate(NamedTuple):
    """
    网络更新阶段的状态（用于 _update_epoch 的扫描）
    """
    train_state: TrainState        # 当前训练状态
    running_grad: jnp.ndarray      # 运行梯度
    traj_batch: Transition         # 一批轨迹
    advantages: jnp.ndarray        # 优势估计
    targets: jnp.ndarray           # 回报目标
    rng: Array                     # 随机数生成器

# ============================================================================
# 第三部分：make_train 函数（这是算法的核心！）
# ============================================================================

def make_train(config):
    """
    创建一个可训练的函数。这个函数被 jax.vmap 并行化。
    
    参数：
        config (dict): 所有配置参数，来自 config_ppo.yaml
    
    返回：
        train (function): 一个接受 (rng, exp_id) 的训练函数
    
    为什么要包装成函数？
        1. 配置被"冻结"在函数内部，这样 JAX 编译时能生成最优代码
        2. 便于用 jax.vmap 并行化多个种子的训练
    """
    
    # 第一步：创建游戏环境
    env, env_params = gymnax.make(config["env_name"])
    # env: 环境对象，有 .reset() 和 .step() 方法
    # env_params: 环境的固定参数（重力、摩擦力等）
    
    # 第二步：计算派生的配置参数
    # 这些是根据基础参数计算出来的
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    # 例如：2,000,000 / 128 / 16 = 977 次大循环
    
    config["minibatch_size"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )
    # 例如：16 * 128 / 4 = 512 条数据/minibatch

    def train(rng, exp_id):
        """
        实际的训练函数。这个函数会被 jax.vmap 调用多次（一次/种子）
        
        参数：
            rng: 这个种子的随机数生成器
            exp_id: 实验 ID（0, 1, 2, ...）
        
        返回：
            final_runner_state: 训练结束时的状态
            metrics_batch: 所有 update 步骤的指标
        """
        
        # ====================================================================
        # 第四部分：训练准备（train_setup）
        # ====================================================================
        
        def train_setup(rng):
            """
            初始化环境、网络和优化器
            
            返回：
                obs: 初始观测
                state: 环境初始状态
                train_state: 初始化的训练状态
                running_grad: 初始化的运行梯度（全零）
                network: 网络对象
            """
            
            # 1. 重置环境
            rng, _rng_reset = jax.random.split(rng)  # 分出一个新的随机数种子
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            # 对 16 个并行环境各分一个随机种子
            
            obs, state = jax.vmap(env.reset, in_axes=(0, None))(_rng_resets, env_params)
            # vmap：对 16 个环境并行调用 reset
            # in_axes=(0, None)：第一个参数按第 0 轴（16 个种子）并行，第二个参数广播（不并行）
            # obs 和 state 的形状：(16, obs_dim) 和 (16, state_dim)
            
            # 2. 创建学习率调度器
            def linear_schedule(count):
                """
                学习率随训练进度线性衰减
                
                刚开始：学习率很高，学得快
                快结束：学习率很低，学得慢，微调
                """
                frac = (
                    1.0
                    - (count // (config["num_minibatches"] * config["update_epochs"]))
                    / config["num_updates"]
                )
                return config["lr"] * frac

            if config["anneal_lr"]:
                lr_schedule = linear_schedule
            else:
                lr_schedule = config["lr"]  # 固定学习率
            
            # 3. 创建网络
            network = ActorCriticDiscrete(
                action_dim=env.num_actions,
                activation=config["activation"],
            )
            # env.num_actions：MountainCar 有 3 个动作
            
            # 4. 初始化网络参数
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(obs.shape)  # 假输入，用来推断网络形状
            network_params = network.init(_rng, init_x)
            # network_params 是一个嵌套字典，包含所有权重和偏差
            
            # 5. 创建优化器
            # optax.chain：依次应用多个操作
            if config["optimizer"] == "adam":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    # 梯度裁剪：如果梯度范数超过 max_grad_norm，就缩放
                    # 作用：防止梯度爆炸导致训练崩溃
                    
                    optax.adam(
                        learning_rate=lr_schedule,
                        eps=1e-5,
                        b1=config["beta_1"],  # 一阶矩的指数衰减率
                        b2=config["beta_2"],  # 二阶矩的指数衰减率
                    ),
                    # Adam 优化器
                    # b1=0.9：记住 90% 的历史梯度方向
                    # b2=0.999：记住 99.9% 的历史梯度平方
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
            
            # 6. 创建训练状态
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            # train_state.params：网络参数
            # train_state.opt_state：优化器状态（Adam 的 m 和 v）
            
            # 7. 初始化运行梯度（全零）
            running_grad = jax.tree.map(jnp.zeros_like, network_params)
            # 这个会用来计算与当前梯度的相似度
            
            return obs, state, train_state, running_grad, network

        # 调用 train_setup 进行初始化
        rng, _rng_setup = jax.random.split(rng)
        obs, state, train_state, running_grad, network = train_setup(_rng_setup)

        # ====================================================================
        # 第五部分：主训练循环
        # ====================================================================
        
        def _train_loop(runner_state, unused):
            """
            一个 update 周期（大循环）
            
            这里进行：
            1. 与环境交互，收集 num_steps 步的经验
            2. 计算 GAE（优势估计）
            3. 更新网络参数 num_epochs 次
            4. 记录日志
            
            使用 jax.lax.scan，会被调用 num_updates 次（977 次）
            """
            
            initial_timesteps = runner_state.timesteps
            # 记录初始时间步（用于计算 episode length）

            # ────────────────────────────────────────────────────────────────
            # 第一阶段：与环境交互，收集经验
            # ────────────────────────────────────────────────────────────────
            
            def _env_step(runner_state, unused):
                """
                与环境交互一步
                
                这个函数会被 jax.lax.scan 调用 num_steps 次（128 次）
                """
                
                train_state = runner_state.train_state
                obs = runner_state.obs
                state = runner_state.state
                done = runner_state.done
                rng = runner_state.rng

                # 1. 如果游戏已结束，重置环境
                def reset_if_done(obs, state, done, rng):
                    """条件函数：如果 done，就重置；否则保持不变"""
                    return jax.lax.cond(
                        done,
                        lambda: env.reset(rng, env_params),  # True: 重置
                        lambda: (obs, state)                 # False: 保持
                    )

                rng, _rng_reset = jax.random.split(rng)
                rng_resets = jax.random.split(_rng_reset, config["num_envs"])
                obs, state = jax.vmap(reset_if_done)(obs, state, done, rng_resets)
                # 对 16 个环境并行调用 reset_if_done

                # 2. 从网络获取策略和价值
                rng, _rng_action = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs)
                # pi: Categorical 分布，形状 (16,)
                # value: 价值估计，形状 (16,)

                # 3. 采样动作
                action = pi.sample(seed=_rng_action)
                # action: 0, 1, 或 2，形状 (16,)
                
                log_prob = pi.log_prob(action)
                # log_prob: 该动作在策略下的对数概率，形状 (16,)

                # 4. 在环境中执行动作
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                new_obs, new_state, reward, new_done, info = jax.vmap(env.step)(
                    rng_step, state, action
                )
                # 对 16 个环境并行执行 step

                # 5. 更新时间步计数
                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)
                # 如果游戏结束，重置为 0；否则 +1

                # 6. 打包成 Transition
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

                # 7. 更新 runner_state
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
            
            # 使用 jax.lax.scan 调用 _env_step 共 num_steps 次
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["num_steps"],  # 128 次
            )
            # traj_batch: Transition，形状 (128, 16, ...)

            # ────────────────────────────────────────────────────────────────
            # 第二阶段：计算 GAE（优势估计）
            # ────────────────────────────────────────────────────────────────
            
            train_state = runner_state.train_state
            last_obs = runner_state.obs
            rng = runner_state.rng

            # 计算最后一步的价值（用于 GAE 计算）
            _, last_value = network.apply(train_state.params, last_obs)
            last_value = last_value.squeeze()

            def _calculate_gae(traj_batch, last_val):
                """
                计算 GAE（广义优势估计）
                
                基本公式：
                    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                    A_t = delta_t + gamma * lambda * A_{t+1}
                
                返回：
                    advantages: 优势估计
                    targets: 回报目标（用来训练 Critic）
                """
                
                def _get_advantages(gae_and_next_value, transition):
                    """
                    一步的 GAE 计算（从后向前）
                    
                    使用 jax.lax.scan 的 reverse=True 从后往前遍历
                    """
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.new_done,
                        transition.value,
                        transition.reward,
                    )
                    
                    # delta = r + gamma * V(s') * (1 - done) - V(s)
                    # 如果游戏结束（done=True），V(s')=0（因为没有后续状态）
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    
                    # gae = delta + gamma * lambda * (1 - done) * gae_prev
                    # lambda 权衡估计的偏差和方差
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    )
                    
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,  # 从后向前
                    unroll=16,
                )
                
                # targets = advantages + values = 回报估计
                targets = advantages + traj_batch.value
                
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # ────────────────────────────────────────────────────────────────
            # 第三阶段：更新网络参数
            # ────────────────────────────────────────────────────────────────
            
            def _update_epoch(update_state, unused):
                """
                一个 epoch 的网络更新（会被调用 update_epochs 次）
                
                过程：
                1. 随机打乱数据
                2. 分成 minibatch
                3. 对每个 minibatch 计算损失和梯度
                4. 用 Adam 更新参数
                """
                
                train_state = update_state.train_state
                running_grad = update_state.running_grad
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                # 1. 随机打乱环境维度（第 1 轴）
                rng, _rng_permute = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_permute, config["num_envs"])
                # 生成 0-15 的随机排列
                
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1),  # axis=1 是环境维度
                    batch
                )
                # 按 permutation 重新排列

                # 2. 分成 minibatch
                shuffled_batch_split = jax.tree.map(
                    lambda x: jnp.reshape(
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch,
                )
                # 从 (128, 16, ...) 变成 (128, 4, 512, ...)
                
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1),  # 交换第 0 和 1 轴
                    shuffled_batch_split,
                )
                # 从 (128, 4, 512, ...) 变成 (4, 128, 512, ...)
                # 现在第 0 轴是 minibatch 索引

                def _update_minibatch(carry, minibatch):
                    """
                    更新一个 minibatch 的网络参数
                    """
                    train_state, running_grad = carry
                    traj_minibatch, advantages_minibatch, targets_minibatch = minibatch

                    def _loss(params, traj_minibatch, gae_minibatch, targets_minibatch):
                        """
                        计算 PPO 损失函数（这是算法的数学核心！）
                        """
                        
                        # 1. 网络前向推理
                        pi, value = network.apply(params, traj_minibatch.obs)
                        log_prob = pi.log_prob(traj_minibatch.action)

                        # ──────────────────────────────────────
                        # Actor 损失（策略损失）
                        # ──────────────────────────────────────
                        
                        # 计算策略比率 r = pi_new / pi_old
                        logratio = log_prob - traj_minibatch.log_prob
                        ratio = jnp.exp(logratio)
                        # 用 log 的差避免数值不稳定
                        
                        # 标准化优势（重要：稳定学习）
                        gae_minibatch = (gae_minibatch - gae_minibatch.mean()) / (
                            gae_minibatch.std() + 1e-8
                        )
                        
                        # PPO 的核心：裁剪替代目标
                        loss_actor_1 = ratio * gae_minibatch  # 无裁剪版本
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],  # 下界：0.8
                                1.0 + config["clip_eps"],  # 上界：1.2
                            )
                            * gae_minibatch
                        )  # 裁剪版本
                        
                        # 取较小值（保守地选择）
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        # 取负号是因为我们要最大化目标，但优化器最小化损失
                        
                        entropy = pi.entropy().mean()
                        # 熵：策略的随机性。高熵=随机，低熵=确定

                        # ──────────────────────────────────────
                        # Critic 损失（价值损失）
                        # ──────────────────────────────────────
                        
                        # 计算价值预测误差
                        value_pred_clipped = traj_minibatch.value + (
                            value - traj_minibatch.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        # 限制价值变化，防止过度更新
                        
                        value_loss = jnp.square(value - targets_minibatch)
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )
                        # 系数 0.5 是一个常见选择

                        # ──────────────────────────────────────
                        # 诊断统计（用于监控训练）
                        # ──────────────────────────────────────
                        
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        # KL 散度的两个方向估计
                        
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])
                        # 被裁剪的比例（应该在 10-20%）

                        # ──────────────────────────────────────
                        # 总损失
                        # ──────────────────────────────────────
                        
                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss  # vf_coef=0.5
                            - config["ent_coef"] * entropy     # ent_coef=0.003
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

                    # 计算梯度
                    grad_fn = jax.value_and_grad(_loss, has_aux=True)
                    (total_loss, aux_info), grads = grad_fn(
                        train_state.params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )

                    # 更新参数（使用 Adam 优化器）
                    updated_train_state = train_state.apply_gradients(
                        grads=grads,
                    )
                    # 这里 Adam 优化器在幕后计算：
                    # m = 0.9 * m_prev + 0.1 * grads
                    # v = 0.999 * v_prev + 0.001 * grads^2
                    # params = params - lr * m / (sqrt(v) + eps)

                    # ──────────────────────────────────────
                    # 梯度分析（可选：用于研究训练动态）
                    # ──────────────────────────────────────
                    
                    def cosine_similarity(grad1, grad2):
                        """计算两个梯度树的余弦相似度"""
                        flat_grad1 = jax.tree.leaves(grad1)
                        flat_grad2 = jax.tree.leaves(grad2)

                        vec1 = jnp.concatenate([jnp.ravel(x) for x in flat_grad1])
                        vec2 = jnp.concatenate([jnp.ravel(x) for x in flat_grad2])

                        dot_product = jnp.dot(vec1, vec2)
                        norm1 = jnp.linalg.norm(vec1)
                        norm2 = jnp.linalg.norm(vec2)

                        denominator = norm1 * norm2
                        cosine_sim = jnp.where(
                            denominator > 1e-8, dot_product / denominator, 0.0
                        )
                        return cosine_sim

                    cos_sim = cosine_similarity(grads, running_grad)
                    # 当前梯度与历史梯度的相似度
                    
                    cos_sim_mu_prev = cosine_similarity(
                        grads, train_state.opt_state[1][0].mu
                    )
                    # 与 Adam 的前一个一阶矩的相似度
                    
                    cos_sim_mu = cosine_similarity(
                        grads, updated_train_state.opt_state[1][0].mu
                    )
                    # 与 Adam 的当前一阶矩的相似度

                    # 计算梯度方向的变化角度
                    cos_sim_clamped = jnp.clip(cos_sim, -1.0, 1.0)
                    gradient_angle_rad = jnp.arccos(cos_sim_clamped)
                    gradient_angle_deg = gradient_angle_rad * 180.0 / jnp.pi
                    # 从弧度转换为度数

                    # 更新运行梯度
                    new_running_grad = grads
                    # （可以改进为 EMA：new_running_grad = 0.99*running_grad + 0.01*grads）

                    # 记录诊断统计
                    aux_info["grad_norm"] = pytree_norm(grads)
                    aux_info["mu_norm"] = pytree_norm(
                        updated_train_state.opt_state[1][0].mu
                    )
                    aux_info["nu_norm"] = pytree_norm(
                        updated_train_state.opt_state[1][0].nu
                    )
                    aux_info["cosine_similarity"] = cos_sim
                    aux_info["gradient_angle_deg"] = gradient_angle_deg
                    aux_info["cosine_similarity_mu"] = cos_sim_mu

                    return (updated_train_state, new_running_grad), (total_loss, aux_info)

                # 对所有 minibatch 更新参数
                (final_train_state, final_running_grad), loss_info = jax.lax.scan(
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

                return update_state, loss_info

            # 对所有 epoch 更新参数
            update_state = Updatestate(
                train_state=train_state,
                running_grad=runner_state.running_grad,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=rng,
            )

            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config["update_epochs"],  # 2 次
            )

            # ────────────────────────────────────────────────────────────────
            # 第四阶段：计算日志指标
            # ────────────────────────────────────────────────────────────────
            
            reward = traj_batch.reward
            done = traj_batch.new_done
            cumulative_return = runner_state.cumulative_return

            # 计算每集的总奖励
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
            returns_avg = jnp.where(
                done.sum() > 0, only_returns.sum() / done.sum(), 0.0
            )
            # 平均每集的总奖励

            # 计算每集的长度
            def _episode_lengths(carry_length, done):
                cumulative_length = carry_length + 1
                reset_length = jnp.zeros(done.shape[1:], dtype=jnp.int32)
                carry_length = jnp.where(done, reset_length, cumulative_length)
                return carry_length, cumulative_length

            _, episode_lengths = jax.lax.scan(
                _episode_lengths, initial_timesteps, done
            )

            only_episode_ends = jnp.where(done, episode_lengths, 0)
            episode_length_avg = jnp.where(
                done.sum() > 0, only_episode_ends.sum() / done.sum(), 0.0
            )

            # 计算网络参数的统计信息
            network_leaves = jax.tree.leaves(update_state.train_state.params)
            flat_network = jnp.concatenate([jnp.ravel(x) for x in network_leaves])
            network_l1 = jnp.sum(jnp.abs(flat_network))
            network_l2 = jnp.linalg.norm(flat_network)
            network_linfty = jnp.max(jnp.abs(flat_network))
            network_mu = jnp.mean(flat_network)
            network_std = jnp.std(flat_network)
            network_max = jnp.max(flat_network)
            network_min = jnp.min(flat_network)

            # 汇总所有指标
            total_loss, loss_info = loss_info
            loss_info["total_loss"] = total_loss
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric = {}
            metric["update_step"] = runner_state.update_step
            metric["env_step"] = (
                runner_state.update_step * config["num_envs"] * config["num_steps"]
            )
            metric["return"] = returns_avg
            metric["episode_length"] = episode_length_avg
            metric["network_l1"] = network_l1
            metric["network_l2"] = network_l2
            metric["network_linfty"] = network_linfty
            metric["network_mu"] = network_mu
            metric["network_std"] = network_std
            metric["network_max"] = network_max
            metric["network_min"] = network_min
            metric.update(loss_info)

            # ────────────────────────────────────────────────────────────────
            # 第五阶段：日志回调（发送到 Wandb）
            # ────────────────────────────────────────────────────────────────
            
            def callback(exp_id, metric):
                """JAX 回调函数：发送 numpy 数组到 Wandb"""
                np_log_dict = {k: np.array(v) for k, v in metric.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)
            # JAX 的回调机制，用来在 JIT 编译的代码中执行 Python 代码

            # ────────────────────────────────────────────────────────────────
            # 更新最终的 runner_state
            # ────────────────────────────────────────────────────────────────
            
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

        # 初始化最初的 runner_state
        rng, _train_rng = jax.random.split(rng)
        done = jnp.zeros((config["num_envs"]), dtype=jnp.bool_)
        cumulative_return = jnp.zeros((config["num_envs"]), dtype=float)
        initial_runner_state = RunnerState(
            train_state=train_state,
            running_grad=running_grad,
            obs=obs,
            state=state,
            done=done,
            cumulative_return=cumulative_return,
            timesteps=jnp.zeros((config["num_envs"]), dtype=jnp.int32),
            update_step=0,
            rng=_train_rng,
        )

        # 运行训练循环 num_updates 次
        final_runner_state, metrics_batch = jax.lax.scan(
            _train_loop,
            initial_runner_state,
            None,
            length=config["num_updates"],  # 977 次
        )
        
        return final_runner_state, metrics_batch

    return train

# ============================================================================
# 第六部分：入口函数（main）和配置加载
# ============================================================================

@hydra.main(version_base=None, config_path="./", config_name="config_ppo")
def main(config):
    """
    主函数：加载配置，编译训练函数，启动训练
    
    参数：
        config: Hydra 加载的配置对象
    
    执行过程：
    1. 将配置转换为 Python 字典
    2. 创建训练函数
    3. 用 jax.vmap 并行化（多个种子）
    4. 用 jax.jit 编译优化
    5. 启动训练
    6. 完成后关闭 Wandb
    """
    
    try:
        # 1. 配置转换
        config = OmegaConf.to_container(config)
        # 从 Hydra 的 OmegaConf 转换为标准 Python 字典
        
        # 2. 创建随机数生成器
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        # 为每个种子生成不同的随机数
        
        exp_ids = jnp.arange(config["num_seeds"])
        # 实验 ID：0, 1, 2, ...

        print("Starting compile...")
        # 3. 创建训练函数
        train_vmap = jax.vmap(make_train(config))
        # jax.vmap：将 make_train(config) 返回的函数在第 0 轴（种子维度）并行化
        
        # 4. JIT 编译
        train_vjit = jax.block_until_ready(jax.jit(train_vmap))
        # jax.jit：编译成机器码
        # jax.block_until_ready：等待编译完成
        print("Compile finished...")

        # 5. 初始化 Wandb 多进程 logger
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = (
            f"ppo_beta1_{config['beta_1']}_{config['env_name']}"
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        )
        # group 名字会包含 beta_1 值，便于在 Wandb 上分组对比
        
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=config["wandb_mode"],
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        # 6. 开始训练
        print("Running...")
        out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        # train_vjit 被调用，输入是 num_seeds 个随机数和 num_seeds 个 ID
        # 10 个种子会并行运行（使用 jax.vmap）
        
    finally:
        # 7. 清理：关闭 Wandb
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
