#!/bin/bash

# 🚀 快速测试脚本 - 让你在 5 分钟内验证整个代码流程
# 
# 这个脚本会：
# 1. 用最小化的配置运行训练
# 2. 只进行足够的迭代来验证代码逻辑
# 3. 快速完成，便于调试
#
# 执行方式：
#   bash quick_test.sh
#
# 预期运行时间：5-10 分钟（取决于 GPU）
# 预期输出：在 Wandb 上会创建一个新的实验

cd "$(dirname "$0")"

# 确保 PYTHONPATH 正确
export PYTHONPATH=/home/yichen/ADAM/optimize

# 初始化 conda（根据实际路径调整）
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh
conda activate adam

echo "================================================"
echo "🚀 开始快速测试"
echo "================================================"
echo ""
echo "配置说明："
echo "  - total_timesteps: 20000    (超小，快速完成)"
echo "  - num_envs: 4               (只用 4 个环境，省内存)"
echo "  - num_steps: 64             (每次收集 64 步)"
echo "  - num_seeds: 1              (只跑 1 个种子)"
echo ""
echo "这些设置能快速通过代码，但不会产生好的训练结果。"
echo "目的是验证代码逻辑是否正确工作。"
echo ""

# 运行训练
python3 optimize/experiments/gymnax/ppo/ppo/ppo_discrete.py \
    total_timesteps=20000 \
    num_envs=4 \
    num_steps=64 \
    num_seeds=1 \
    seed=42

echo ""
echo "================================================"
echo "✅ 快速测试完成！"
echo "================================================"
echo ""
echo "检查项：你应该看到"
echo "  1. 代码没有崩溃"
echo "  2. 在 Wandb 上生成了图表"
echo "  3. 输出的 'return' 大概在 -200 到 0 之间（山地车游戏）"
echo ""
echo "下一步："
echo "  1. 查看 Wandb 仪表板：https://wandb.ai"
echo "  2. 尝试修改参数（如 lr 或 beta_1）"
echo "  3. 对比两次运行的结果"
echo ""
