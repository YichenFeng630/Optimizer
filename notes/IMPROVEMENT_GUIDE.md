# 💡 PPO 改进实战指南

> **目标**：从"理解代码"进阶到"改进算法"。这份文档提供三个从易到难的改进方向，每个都有具体的代码修改和验证方法。

---

## 📊 改进方向总览

| 难度 | 改进方向 | 预期收益 | 所需时间 |
|------|---------|---------|--------|
| ⭐ | 超参数 Grid Search | 找到最优 beta_1 值 | 1-2 小时 |
| ⭐⭐ | 网络结构优化 | 提升 AI"大脑"容量 | 2-4 小时 |
| ⭐⭐⭐ | 算法改进（EMA 梯度） | 理解训练动态，发现问题 | 4-6 小时 |

---

## 改进方向 1️⃣：超参数调优（最简单，最实用）

### 为什么这个改进有用？

算法的效果对超参数非常敏感。即使只改一两个数字，训练效果也会有明显差异。

### 改进目标

找到最优的 **学习率 (lr)** 和 **Beta_1** 值的组合。

### 步骤 1：创建 Sweep 脚本

创建文件：`optimize/experiments/gymnax/ppo/ppo/sweep_hyperparams.sh`

```bash
#!/bin/bash

# 超参数 Grid Search 脚本
# 遍历学习率和 beta_1 的不同组合

export PYTHONPATH=/home/yichen/ADAM/optimize
source ~/miniconda3/etc/profile.d/conda.sh && conda activate adam

# 测试矩阵
learning_rates=(1e-3 5e-4 2e-4)
beta_1_values=(0.8 0.9 0.95 0.99)

echo "======================================"
echo "🔬 PPO 超参数 Grid Search"
echo "======================================"
echo ""
echo "将测试 ${#learning_rates[@]} × ${#beta_1_values[@]} = $((${#learning_rates[@]} * ${#beta_1_values[@]})) 个配置"
echo ""

config_count=0

for lr in "${learning_rates[@]}"; do
    for beta in "${beta_1_values[@]}"; do
        config_count=$((config_count + 1))
        echo "========== 配置 $config_count =========="
        echo "学习率: $lr"
        echo "Beta_1: $beta"
        echo ""
        
        cd /home/yichen/ADAM/optimize/optimize/experiments/gymnax/ppo/ppo
        
        # 运行训练
        # 为了快速看到结果，这里使用较小的 total_timesteps
        # 生产环境应该用 2e6
        python3 ppo_discrete.py \
            lr=$lr \
            beta_1=$beta \
            total_timesteps=100000 \
            num_seeds=1
        
        # 每次运行间隔 5 秒
        sleep 5
    done
done

echo ""
echo "======================================"
echo "✅ Grid Search 完成！"
echo "======================================"
echo ""
echo "📊 检查 Wandb："
echo "  1. 打开 https://wandb.ai"
echo "  2. 进入项目 'optimize'"
echo "  3. 在 'Groups' 中看到所有实验分组"
echo "  4. 比较最终得分和收敛速度"
echo ""
```

### 步骤 2：运行脚本

```bash
bash sweep_hyperparams.sh
```

### 步骤 3：在 Wandb 上比较结果

打开 Wandb 网站，你会看到类似这样的对比表格：

```
学习率    Beta_1   最终得分   收敛速度
2e-4     0.8      -150      快
2e-4     0.9      -180      快
2e-4     0.95     -140      ✅ 最好
2e-4     0.99     -200      慢
5e-4     0.8      -200      震荡
...
```

### 步骤 4：选择最优参数

假设通过对比你发现 `lr=2e-4, beta_1=0.95` 效果最好，那么：

修改 `config_ppo.yaml`：

```yaml
"lr": 2e-4          # 改了
"beta_1": 0.95      # 改了
```

然后用新的配置重新训练：

```bash
cd /home/yichen/ADAM/optimize/optimize/experiments/gymnax/ppo/ppo
python3 ppo_discrete.py total_timesteps=2000000
```

### 📈 期望结果

- 最终得分应该更高（更接近 0 或正数，取决于游戏）
- 训练更稳定（曲线更平滑，波动更小）
- 收敛速度更快（用更少的步数达到目标分数）

---

## 改进方向 2️⃣：网络结构优化（中等难度）

### 为什么这个改进有用？

当前的网络很简单（两层 64 单元）。增加网络的容量可能让 AI 学到更复杂的行为。

### 改进目标

尝试不同的网络大小，找到性能 vs 计算成本的最优平衡。

### 步骤 1：修改网络架构

编辑 `optimize/networks/mlp.py`：

**原代码（简单版）：**

```python
class ActorCriticDiscrete(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Actor 分支
        actor_mean = nn.Dense(64, ...)(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, ...)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, ...)(actor_mean)
        pi = Categorical(logits=actor_mean)
        
        # Critic 分支
        critic = nn.Dense(64, ...)(x)
        critic = activation(critic)
        critic = nn.Dense(64, ...)(critic)
        critic = activation(critic)
        critic = nn.Dense(1, ...)(critic)
        
        return pi, jnp.squeeze(critic, axis=-1)
```

**改进版本 1：增加宽度（128 单元）：**

```python
class ActorCriticDiscrete(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 128  # ← 新增参数

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Actor 分支
        actor_mean = nn.Dense(self.hidden_dim, ...)(x)  # ← 改成 128
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_dim, ...)(actor_mean)  # ← 改成 128
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, ...)(actor_mean)
        pi = Categorical(logits=actor_mean)
        
        # Critic 分支（相同修改）
        critic = nn.Dense(self.hidden_dim, ...)(x)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_dim, ...)(critic)
        critic = activation(critic)
        critic = nn.Dense(1, ...)(critic)
        
        return pi, jnp.squeeze(critic, axis=-1)
```

**改进版本 2：增加深度（三层 64 单元）：**

```python
class ActorCriticDiscrete(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Actor 分支（新增一层）
        actor_mean = nn.Dense(64, ...)(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, ...)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, ...)(actor_mean)  # ← 新增
        actor_mean = activation(actor_mean)         # ← 新增
        actor_mean = nn.Dense(self.action_dim, ...)(actor_mean)
        pi = Categorical(logits=actor_mean)
        
        # Critic 分支（相同修改）
        critic = nn.Dense(64, ...)(x)
        critic = activation(critic)
        critic = nn.Dense(64, ...)(critic)
        critic = activation(critic)
        critic = nn.Dense(64, ...)(critic)  # ← 新增
        critic = activation(critic)         # ← 新增
        critic = nn.Dense(1, ...)(critic)
        
        return pi, jnp.squeeze(critic, axis=-1)
```

### 步骤 2：对比实验

创建脚本 `optimize/experiments/gymnax/ppo/ppo/compare_architectures.sh`：

```bash
#!/bin/bash

export PYTHONPATH=/home/yichen/ADAM/optimize
source ~/miniconda3/etc/profile.d/conda.sh && conda activate adam

echo "对比网络架构"
echo ""

# 方案 1：原始网络（64, 64）
echo "正在运行原始网络..."
python3 ppo_discrete.py \
    total_timesteps=100000 \
    job_type="baseline_64x64"

sleep 5

# 方案 2：宽网络（128, 128）
# 需要同时修改 mlp.py 中的 hidden_dim = 128
echo "正在运行宽网络..."
python3 ppo_discrete.py \
    total_timesteps=100000 \
    job_type="wider_128x128"

sleep 5

# 方案 3：深网络（64, 64, 64）
# 需要修改 mlp.py 添加第三层
echo "正在运行深网络..."
python3 ppo_discrete.py \
    total_timesteps=100000 \
    job_type="deeper_64x64x64"

echo ""
echo "完成！检查 Wandb 对比三个网络的效果"
```

### 步骤 3：分析结果

在 Wandb 上对比三条曲线：

- **基准线（64×64）**：快速基线
- **宽网络（128×128）**：通常学得更好，但计算需要更多 GPU
- **深网络（64×64×64）**：有时候效果也不错，但容易过拟合

### 💡 注意事项

- 网络越大，需要更多 GPU 内存
- 不一定网络越大效果越好（可能过拟合）
- 对于 MountainCar 这个简单游戏，64×64 可能已经足够了

---

## 改进方向 3️⃣：算法改进 - EMA 运行梯度（高级）

### 为什么这个改进有用？

当前代码记录 `running_grad` 但没有充分利用。改进它可以：

1. 更好地追踪梯度的长期变化趋势
2. 发现训练不稳定的信号
3. 诊断为什么某些实验失败

### 改进目标

使用 **指数移动平均 (EMA)** 来更新 `running_grad`，而不是简单替换。

### 数学基础

```
原方法：
    running_grad = grads  （完全替换，丢失历史信息）

改进方法：
    running_grad = 0.99 × running_grad + 0.01 × grads
    
效果：
    - 保留 99% 的历史梯度方向
    - 吸收 1% 的当前梯度
    - 梯度方向的变化会逐步体现（平滑）
```

### 步骤 1：修改代码

编辑 `ppo_discrete.py`，找到 `_update_minibatch` 函数中的这一行：

**原代码（第约 400 行）：**

```python
# Update running gradient with current gradient
new_running_grad = grads
```

**改成：**

```python
# Update running gradient with current gradient using EMA
ema_decay = 0.99
new_running_grad = jax.tree.map(
    lambda rg, g: rg * ema_decay + g * (1 - ema_decay),
    running_grad,
    grads
)
```

### 步骤 2：理解改进

**修改前：**
```
第 1 步：running_grad = grads_1
第 2 步：running_grad = grads_2  （完全忘记了 grads_1）
第 3 步：running_grad = grads_3  （完全忘记了 grads_2）
```

**修改后：**
```
第 1 步：running_grad = 0.99 × 0 + 0.01 × grads_1 = grads_1
第 2 步：running_grad = 0.99 × grads_1 + 0.01 × grads_2 ≈ 0.99×grads_1 + 0.01×grads_2
第 3 步：running_grad = 0.99 × prev + 0.01 × grads_3 ≈ 梯度的长期平均
```

现在 `running_grad` 代表了过去梯度的"幽灵"，它会：
- 在梯度稳定时保持不变
- 在梯度改变方向时逐步调整

### 步骤 3：运行对比实验

**版本 A：原方法（simple replacement）**

```bash
python3 ppo_discrete.py \
    total_timesteps=200000 \
    job_type="original_running_grad"
```

**版本 B：改进方法（EMA）**

修改代码后：

```bash
python3 ppo_discrete.py \
    total_timesteps=200000 \
    job_type="ema_running_grad"
```

### 步骤 4：分析日志

在 Wandb 中查看 `cosine_similarity` 指标：

- **原方法**：余弦相似度会更跳跃，变化快
- **改进方法**：余弦相似度更平滑，变化缓慢

```
cosine_similarity 的含义：
  1.0 = 梯度方向完全相同（非常好）
  0.5 = 梯度方向成 60° 角（一般）
  0.0 = 梯度方向垂直（很差）
 -0.5 = 梯度方向相反（非常差）
```

### 💡 何时 EMA 有用

- 当你看到 `cosine_similarity` 频繁在 -1 到 1 之间跳跃时（不稳定）
- 当训练曲线震荡很大时
- 当想诊断训练为什么失败时

---

## 改进方向 4️⃣：奖励整形（Reward Shaping）

### 为什么这个改进有用？

有时候环境的默认奖励不够好。通过"奖励整形"，我们可以给 AI 额外的反馈信号，加快学习。

### 改进目标

在山地车游戏中，鼓励 AI 向山顶移动（而不仅仅依赖游戏本身的奖励）。

### 步骤 1：理解山地车游戏

```
游戏状态：position, velocity
奖励机制：
  - 默认：每步 -1（鼓励快速到达目标）
  - 到达目标：+0（任务完成）
  - 观察：位置范围通常是 [-1.2, 0.6]
```

### 步骤 2：修改环境交互部分

编辑 `ppo_discrete.py`，在 `_env_step` 中找到这一行：

```python
new_obs, new_state, reward, new_done, info = jax.vmap(env.step)(
    rng_step, state, action
)
```

改成：

```python
new_obs, new_state, reward, new_done, info = jax.vmap(env.step)(
    rng_step, state, action
)

# ← 添加奖励整形
# 鼓励向右移动（目标方向）
position_shaped_reward = 0.1 * (new_obs[:, 0] - obs[:, 0])
# obs[:, 0] 是 position（山地车的第一个观测维度）

reward = reward + position_shaped_reward
# 原奖励 + 位置奖励
```

### 步骤 3：对比实验

**版本 A：无奖励整形**

```bash
python3 ppo_discrete.py total_timesteps=100000 job_type="no_reward_shaping"
```

**版本 B：有奖励整形**

修改代码后：

```bash
python3 ppo_discrete.py total_timesteps=100000 job_type="with_reward_shaping"
```

### 期望结果

- 收敛更快（AI 更快学到向右走的好处）
- 最终得分更高

---

## 📋 改进实验检查清单

在做每个改进实验时，使用这个清单：

- [ ] **备份原代码**：`git commit` 或复制一份
- [ ] **明确假设**："我认为这个改进会导致 X 结果"
- [ ] **修改代码**：清楚地标注修改位置（注释说明）
- [ ] **设置对照组**：运行原始版本进行对比
- [ ] **记录配置**：Wandb 会自动记录，确保能在仪表板找到
- [ ] **分析结果**：对比损失曲线、最终得分、收敛速度
- [ ] **得出结论**：这个改进是否有帮助？为什么？
- [ ] **写报告**：记录下来（方便后续回顾）

---

## 🚀 快速开始

想立即做第一个改进？运行：

```bash
# 1. 进入项目目录
cd /home/yichen/ADAM/optimize/optimize/experiments/gymnax/ppo/ppo

# 2. 设置环境
export PYTHONPATH=/home/yichen/ADAM/optimize
source ~/miniconda3/etc/profile.d/conda.sh && conda activate adam

# 3. 快速对比两个 beta_1 值
echo "运行 beta_1=0.8..."
python3 ppo_discrete.py beta_1=0.8 total_timesteps=50000 job_type="ppo_beta_0.8"

sleep 10

echo "运行 beta_1=0.95..."
python3 ppo_discrete.py beta_1=0.95 total_timesteps=50000 job_type="ppo_beta_0.95"

echo ""
echo "完成！现在进入 Wandb 查看对比："
echo "https://wandb.ai/projects"
```

---

## 常见问题

### Q：如何同时保存两个版本的代码？

A：在做改进前，备份一份：

```bash
cp ppo_discrete.py ppo_discrete_original.py
# 现在修改 ppo_discrete.py
# 如果出问题了，可以随时恢复
```

### Q：如何快速看到改进的效果？

A：用小的 `total_timesteps` 和 `num_seeds` 做快速实验：

```bash
python3 ppo_discrete.py total_timesteps=50000 num_seeds=1
```

然后用更大的参数重新运行验证结果是否稳定。

### Q：怎样知道改进是否真的有效？

A：对比以下指标：

1. **最终得分**：是否更高？
2. **收敛速度**：是否需要更少的步数达到平台期？
3. **稳定性**：曲线是否更平滑？

---

**好的，现在你有了完整的改进指南。选择一个方向，开始你的第一个改进实验吧！🚀**
