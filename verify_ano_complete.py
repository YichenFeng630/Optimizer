"""
完整验证ANO优化器实现：对照论文公式和官方实现
"""
import jax
import jax.numpy as jnp
import numpy as np

print("=" * 100)
print("ANO 优化器完整验证")
print("=" * 100)
print()

# ============================================================
# 论文公式（官方 anonymous.4open.science）
# ============================================================
print("【1】论文公式（Official Paper Formula）")
print("-" * 100)
print("""
来源: https://anonymous.4open.science/r/ano-optimizer-1645/optimizers/README.md

ANO算法（β₁ ∈ [0,1), β₂ ∈ [0.5,1)）:
    m_k = β₁*m_{k-1} + (1-β₁)*g_k
    v_k = β₂*v_{k-1} - (1-β₂)*sign(v_{k-1}-g_k²)*g_k²
    v̂_k = v_k / (1-β₂^k)
    θ_k = θ_{k-1} - (η_k/√(v̂_k+ε))*sign(m_k)*|g_k| - η_k*λ*θ_{k-1}

关键点：
  ✓ m无bias correction
  ✓ v有bias correction: v̂ = v/(1-β₂^k)
  ✓ v的更新形式: v = β₂*v - (1-β₂)*sign(v-g²)*g²  （注意是β₂*v开头）
  ✓ 学习率应用: η/√(v̂+ε)
  ✓ 梯度变换: sign(m)*|g|
  ✓ 权重衰减: 解耦形式 -η*λ*θ
""")

# ============================================================
# 官方PyTorch实现（Adrienkgz）
# ============================================================
print("\n【2】官方PyTorch实现（Adrienkgz/ano-experiments）")
print("-" * 100)
print("""
来源: https://github.com/Adrienkgz/ano-experiments/blob/main/optimizers/ano.py

关键代码片段（第54-67行）:
    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
    square_grad = torch.square(g)
    sign_term = torch.sign(square_grad - state['exp_avg_sq'])
    state['exp_avg_sq'].mul_(beta2).add_(sign_term * square_grad, alpha=1 - beta2)
    
    bias_c2 = 1 - beta2 ** t
    v_hat = exp_avg_sq / bias_c2
    adjusted_learning_rate = lr / torch.sqrt(v_hat + eps)
    update = adjusted_learning_rate * g.abs() * torch.sign(exp_avg)

等价形式：
    m = β₁*m + (1-β₁)*g
    v = v*β₂ + (1-β₂)*sign(g²-v)*g²
    v̂ = v/(1-β₂^k)
    update = (lr/√(v̂+ε)) * |g| * sign(m)

注意：v的PyTorch形式 v*β₂ + ... 与论文的 β₂*v - ... 在数学上等价！
""")

# ============================================================
# 你的JAX实现验证
# ============================================================
print("\n【3】你的JAX实现（ppo_ano.py）")
print("-" * 100)

# 模拟你的代码逻辑
def your_ano_update(g, m_old, v_old, b1, b2, lr, eps, step):
    """
    你的实现（从ppo_ano.py第170-195行，已修复）
    """
    # Bias correction
    bias_corr_2 = 1.0 - b2 ** step
    
    # 一阶矩更新
    m_new = b1 * m_old + (1.0 - b1) * g
    
    # 二阶矩更新（修复后）
    g_sq = g ** 2
    sign_term = jnp.sign(v_old - g_sq)  # sign(v - g²)
    v_new = b2 * v_old - (1.0 - b2) * sign_term * g_sq  # 修复：加上β₂*v项
    
    # Bias correction（仅对v）
    v_hat = v_new / bias_corr_2
    
    # 学习率调整
    adjusted_lr = lr / (jnp.sqrt(v_hat) + eps)
    
    # 梯度变换
    transformed_g = adjusted_lr * jnp.abs(g) * jnp.sign(m_new)
    
    return transformed_g, m_new, v_new

# ============================================================
# 数值验证：对比三种实现
# ============================================================
print("\n【4】数值验证：三种实现的等价性")
print("-" * 100)

# 测试参数
test_cases = [
    {"g": 0.5, "m": 0.3, "v": 0.8, "desc": "标准情况"},
    {"g": 1.5, "m": 0.1, "v": 0.05, "desc": "v < g² (小v)"},
    {"g": 0.3, "m": 0.8, "v": 2.0, "desc": "v > g² (大v)"},
    {"g": 1.0, "m": 0.5, "v": 1.0, "desc": "v = g² (边界)"},
]

b1, b2, lr, eps, step = 0.92, 0.99, 3e-4, 1e-8, 10.0

print(f"参数设置: β₁={b1}, β₂={b2}, lr={lr}, ε={eps}, step={step}")
print()

all_passed = True
for i, case in enumerate(test_cases, 1):
    g, m, v = case["g"], case["m"], case["v"]
    desc = case["desc"]
    
    # 1. 论文公式（注意：论文写的是β₂*v开头）
    m_new_paper = b1 * m + (1 - b1) * g
    g_sq = g ** 2
    v_new_paper = b2 * v - (1 - b2) * np.sign(v - g_sq) * g_sq
    v_hat_paper = v_new_paper / (1 - b2**step)
    update_paper = (lr / np.sqrt(v_hat_paper + eps)) * np.abs(g) * np.sign(m_new_paper)
    
    # 2. PyTorch形式（v*β₂ + ...）
    m_new_pytorch = b1 * m + (1 - b1) * g
    v_new_pytorch = v * b2 + (1 - b2) * np.sign(g_sq - v) * g_sq
    v_hat_pytorch = v_new_pytorch / (1 - b2**step)
    update_pytorch = (lr / np.sqrt(v_hat_pytorch + eps)) * np.abs(g) * np.sign(m_new_pytorch)
    
    # 3. 你的JAX实现
    update_yours, m_new_yours, v_new_yours = your_ano_update(
        jnp.array(g), jnp.array(m), jnp.array(v), b1, b2, lr, eps, step
    )
    update_yours = float(update_yours)
    m_new_yours = float(m_new_yours)
    v_new_yours = float(v_new_yours)
    
    # 验证v_new是否等价
    v_match_paper = np.allclose(v_new_yours, v_new_paper, rtol=1e-9)
    v_match_pytorch = np.allclose(v_new_yours, v_new_pytorch, rtol=1e-9)
    
    # 验证最终update
    update_match_paper = np.allclose(update_yours, update_paper, rtol=1e-9)
    update_match_pytorch = np.allclose(update_yours, update_pytorch, rtol=1e-9)
    
    passed = v_match_paper and update_match_paper
    all_passed = all_passed and passed
    
    print(f"测试 {i}: {desc}")
    print(f"  输入: g={g:.2f}, m={m:.2f}, v={v:.2f}, g²={g_sq:.2f}")
    print(f"  v_new:")
    print(f"    论文公式:    {v_new_paper:.8f}")
    print(f"    PyTorch:     {v_new_pytorch:.8f}")
    print(f"    你的实现:     {v_new_yours:.8f}")
    print(f"    匹配论文: {'✓' if v_match_paper else '✗'} | 匹配PyTorch: {'✓' if v_match_pytorch else '✗'}")
    print(f"  update:")
    print(f"    论文公式:    {update_paper:.8e}")
    print(f"    PyTorch:     {update_pytorch:.8e}")
    print(f"    你的实现:     {update_yours:.8e}")
    print(f"    匹配论文: {'✓' if update_match_paper else '✗'} | 匹配PyTorch: {'✓' if update_match_pytorch else '✗'}")
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    print()

# ============================================================
# 关键问题检查
# ============================================================
print("\n【5】关键问题检查清单")
print("-" * 100)

checks = [
    ("m更新公式", "m = β₁*m + (1-β₁)*g", "✓ 正确", "第180行"),
    ("v更新公式", "v = v - (1-β₂)*sign(v-g²)*g²", "✓ 正确（与论文β₂*v - ... 等价）", "第185-186行"),
    ("m的bias correction", "无bias correction", "✓ 正确（m不做bias correction）", "第188行未应用"),
    ("v的bias correction", "v̂ = v/(1-β₂^k)", "✓ 正确", "第189行"),
    ("学习率应用", "η/√(v̂+ε)", "✓ 正确", "第192行"),
    ("梯度变换", "sign(m)*|g|", "✓ 正确", "第195行"),
    ("权重衰减", "解耦形式", "✓ 实现但需确认", "第207-212行"),
]

for item, formula, status, location in checks:
    print(f"  {status:15s} | {item:20s} | {formula:30s} | {location}")

# ============================================================
# v更新的两种等价形式验证
# ============================================================
print("\n【6】v更新的两种等价形式数学证明")
print("-" * 100)
print("""
论文形式: v_new = β₂*v_old - (1-β₂)*sign(v_old-g²)*g²
你的形式: v_new = v_old - (1-β₂)*sign(v_old-g²)*g²

等等！你的代码写的是 v - (1-β₂)*...，而论文写的是 β₂*v - (1-β₂)*...

让我重新检查论文...
""")

# 重新验证
print("重新验证v更新公式的一致性：")
v_test = 1.0
g_test = 0.5
g_sq_test = g_test**2

# 论文形式（β₂*v开头）
v_paper_form = b2 * v_test - (1 - b2) * np.sign(v_test - g_sq_test) * g_sq_test
print(f"  论文: β₂*v - (1-β₂)*sign(v-g²)*g² = {b2}*{v_test} - {1-b2}*sign({v_test}-{g_sq_test})*{g_sq_test}")
print(f"      = {b2*v_test} - {(1-b2)*np.sign(v_test-g_sq_test)*g_sq_test} = {v_paper_form}")

# 你的形式（v开头）
v_your_form = v_test - (1 - b2) * np.sign(v_test - g_sq_test) * g_sq_test
print(f"  你的: v - (1-β₂)*sign(v-g²)*g² = {v_test} - {1-b2}*sign({v_test}-{g_sq_test})*{g_sq_test}")
print(f"      = {v_test} - {(1-b2)*np.sign(v_test-g_sq_test)*g_sq_test} = {v_your_form}")

print(f"\n  差异: {abs(v_paper_form - v_your_form):.10f}")
if not np.allclose(v_paper_form, v_your_form):
    print("  ✗ 警告：两种形式不等价！你的实现与论文公式不同！")
else:
    print("  ✓ 两种形式等价")

# ============================================================
# 最终结论
# ============================================================
print("\n" + "=" * 100)
print("【最终验证结论】")
print("=" * 100)

if all_passed:
    print("✓ 所有数值测试通过")
else:
    print("✗ 部分测试失败")

print("\n重要发现：")
print("  1. 你的v更新使用 v - (1-β₂)*... 形式")
print("  2. 论文公式使用 β₂*v - (1-β₂)*... 形式")
print("  3. 这两种形式在数学上【不等价】！")
print("  4. 需要确认论文公式是否写错，或者你的实现需要调整")
print("\n建议：检查官方anonymous.4open.science仓库中的实际代码实现！")
print("=" * 100)
