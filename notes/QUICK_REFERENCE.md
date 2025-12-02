# 📋 快速参考卡片

> 快速查阅关键信息，无需翻阅整份文档

---

## 🚀 快速命令

```bash
# 快速测试（5 分钟）
cd /home/yichen/ADAM/optimize
bash quick_test.sh

# 基础运行（需要 20-30 分钟）
export PYTHONPATH=/home/yichen/ADAM/optimize
cd optimize/experiments/gymnax/ppo/ppo
python3 ppo_discrete.py

# 修改参数运行
python3 ppo_discrete.py beta_1=0.95 lr=2e-4 num_seeds=1

# 禁用 Wandb 本地测试
python3 ppo_discrete.py wandb_mode=disabled total_timesteps=50000
```

---

## 📁 文件导航

```
/home/yichen/ADAM/optimize/
├── README_LEARNING.md            ← 你在这里，开始读这个
├── LEARNING_GUIDE.md             ← 完整学习指南
├── IMPROVEMENT_GUIDE.md          ← 改进方向和实验指导
├── PPO_DETAILED_COMMENTS.py      ← 带注释的源代码
├── quick_test.sh                 ← 快速验证脚本
│
├── optimize/
│   ├── networks/mlp.py           ← 神经网络定义
│   ├── utils/
│   │   ├── wandb_multilogger.py  ← Wandb 日志
│   │   ├── jax_utils.py          ← 辅助函数
│   │   └── spaces.py             ← 空间定义（少用）
│   └── experiments/gymnax/ppo/ppo/
│       ├── config_ppo.yaml       ← 配置文件（关键！）
│       ├── ppo_discrete.py       ← 主训练脚本
│       ├── sweep_betas.sh        ← Beta 扫描脚本
│       └── __pycache__/
│
└── [其他文件]
```

---

## 🔑 关键配置参数

| 参数 | 默认值 | 范围 | 何时改 |
|------|--------|------|--------|
| `total_timesteps` | 2e6 | 1e4-1e7 | 测试时改小 |
| `lr` | 4e-3 | 1e-5-1e-2 | 调优首选 |
| `beta_1` | 0.9 | 0.8-0.99 | **这是研究重点** |
| `beta_2` | 0.999 | 0.9-0.999 | 很少改 |
| `num_envs` | 16 | 4-64 | 内存不足时改 |
| `num_steps` | 128 | 32-512 | 很少改 |
| `ent_coef` | 0.003 | 0-0.1 | 探索不足时增大 |
| `clip_eps` | 0.2 | 0.1-0.5 | 不稳定时改大 |
| `gamma` | 0.99 | 0.95-0.99 | 很少改 |
| `gae_lambda` | 0.95 | 0.9-0.99 | 很少改 |

**改参数的优先级**：lr > beta_1 > ent_coef > clip_eps > 其他

---

## 📊 三层嵌套循环简化图

```
NumUpdates 循环（977 次）
  │
  ├─ 环境交互（_env_step）：128 次
  │  └─ 每次：obs → network → action → env.step → reward
  │
  ├─ GAE 计算
  │  └─ advantages, targets = _calculate_gae(...)
  │
  └─ 网络更新（_update_epoch）：2 次
     └─ 对每个 Minibatch 更新参数（4 个 minibatch）
        └─ 计算损失 → 梯度 → Adam 优化器 → 参数更新
```

**重要**：
- 最外层（num_updates）被 jax.lax.scan 执行
- 最内层（minibatch）被 GPU JIT 优化，所以非常快
- 中间层（env_step）是数据收集的关键

---

## 🎯 PPO 算法核心三行代码

```python
# 1. 计算比率
ratio = exp(log_prob_new - log_prob_old)

# 2. PPO 裁剪目标（核心）
loss = -min(ratio × A, clip(ratio, 1±eps) × A)

# 3. 完整损失
total_loss = loss_actor + 0.5 × loss_critic - 0.003 × entropy
```

---

## 🔍 调试技巧

### 我想看中间变量怎么办？

```python
# 在 ppo_discrete.py 的 _loss 函数中加入：
jax.debug.print("ratio: {x}", x=ratio)
jax.debug.print("advantage: {x}", x=gae_minibatch)
jax.debug.print("entropy: {x}", x=entropy)

# 然后运行：
python3 ppo_discrete.py wandb_mode=disabled total_timesteps=10000
```

### 代码运行太慢？

```bash
# 方式 1：减少数据量
python3 ppo_discrete.py total_timesteps=10000 num_seeds=1

# 方式 2：在本地 CPU 上快速测试（不用 GPU）
XLA_PLATFORMS=cpu python3 ppo_discrete.py total_timesteps=10000

# 方式 3：减少环境数量
python3 ppo_discrete.py num_envs=4 total_timesteps=50000
```

### 如何查看 Wandb 离线日志？

```bash
# 查看本地保存的日志
ls -la ./wandb/
cat ./wandb/run-*/files/config.yaml
```

---

## 📈 Wandb 关键指标速查表

```
好的训练样子：
  return        ↗️  （持续上升）
  actor_loss    ↘️  （逐步下降）
  entropy       ↘️  （缓慢下降）
  clip_frac     ≈ 0.15  （10-20%）
  grad_norm     ≈ 1.0   （接近 1）

坏的训练样子：
  return        ↘️  （下降或停滞）
  actor_loss    ↗️  （快速增加）
  entropy       📉  （快速掉到 0）
  clip_frac     > 0.5   （过度裁剪）
  grad_norm     >> 1    （梯度爆炸）
```

---

## 🧠 函数速查表

| 函数 | 位置 | 作用 |
|------|------|------|
| `make_train` | ppo_discrete.py | 打包训练函数 |
| `train_setup` | make_train 内 | 初始化网络 |
| `_train_loop` | train 内 | 主循环（977 次） |
| `_env_step` | _train_loop 内 | 与环境交互一步 |
| `_calculate_gae` | _train_loop 内 | 计算优势估计 |
| `_update_epoch` | _train_loop 内 | 更新网络（2 次） |
| `_update_minibatch` | _update_epoch 内 | 更新一个 minibatch |
| `_loss` | _update_minibatch 内 | **计算损失**（这是核心！） |
| `ActorCriticDiscrete` | mlp.py | 神经网络定义 |

---

## 💾 快速实验模板

保存这个，每次做新实验时复制：

```bash
#!/bin/bash

export PYTHONPATH=/home/yichen/ADAM/optimize
source ~/miniconda3/etc/profile.d/conda.sh && conda activate adam

cd /home/yichen/ADAM/optimize/optimize/experiments/gymnax/ppo/ppo

# 改这里
EXPERIMENT_NAME="my_test"
LR=5e-4
BETA_1=0.9
TIMESTEPS=100000

echo "运行实验：$EXPERIMENT_NAME"
echo "配置：lr=$LR, beta_1=$BETA_1"
echo ""

python3 ppo_discrete.py \
    lr=$LR \
    beta_1=$BETA_1 \
    total_timesteps=$TIMESTEPS \
    num_seeds=1 \
    job_type="$EXPERIMENT_NAME"

echo ""
echo "✅ 完成！查看 Wandb 仪表板"
```

保存为 `my_experiment.sh`，然后运行：`bash my_experiment.sh`

---

## 🎓 理解等级自检

**Level 1（初级）**
- [ ] 能运行 quick_test.sh
- [ ] 能在命令行改参数
- [ ] 能在 Wandb 上看到结果

**Level 2（中级）**
- [ ] 能用 jax.debug.print 调试
- [ ] 能理解 Actor-Critic 的概念
- [ ] 能对比两个实验的结果

**Level 3（高级）**
- [ ] 能修改网络代码（mlp.py）
- [ ] 能指出 PPO 的核心公式在代码哪里
- [ ] 能做改进实验并分析结果

**Level 4（专家）**
- [ ] 能从论文推导改进想法
- [ ] 能设计新的实验来验证假设
- [ ] 能提出原创性的改进方案

---

## 🚨 常见错误速查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ModuleNotFoundError: No module named 'optimize'` | PYTHONPATH 未设置 | `export PYTHONPATH=/...` |
| 运行特别慢 (>1h) | 无 GPU 或参数太大 | 改小 total_timesteps |
| Wandb 无日志 | 未登录或网络问题 | `wandb login` 或 `wandb_mode=disabled` |
| 代码崩溃 + OOM | 内存不足 | 减小 num_envs 或 num_seeds |
| return 一直是 -500 | 训练完全失败 | 调大 lr 或改小 clip_eps |

---

## 📞 求助流程

遇到问题时按以下顺序尝试：

1. 查看这个快速参考卡片的"常见错误速查表"
2. 查看 README_LEARNING.md 的 FAQ
3. 查看 PPO_DETAILED_COMMENTS.py 中相关代码的注释
4. 减小 total_timesteps，快速重现问题
5. 用 jax.debug.print 找到出问题的位置

---

**最后提示**：  
保存这个文件在你容易访问的地方，作为快速查阅工具！

祝你学习顺利！ 🚀
