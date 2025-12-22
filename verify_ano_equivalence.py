"""
验证论文公式和Adrienkgz PyTorch实现的数学等价性
"""
import numpy as np

def paper_v_update(v_old, g, beta2):
    """
    论文公式: v_k = β₂*v_{k-1} - (1-β₂)*sign(v_{k-1} - g_k²)*g_k²
    
    注意：论文写的是 v_k = β₂*v_{k-1} - ...，不是 v_k = v_{k-1} - ...
    """
    g_sq = g**2
    sign_term = np.sign(v_old - g_sq)
    v_new = beta2 * v_old - (1 - beta2) * sign_term * g_sq
    return v_new

def pytorch_v_update(v_old, g, beta2):
    """
    PyTorch实现: v = v*β₂ + (1-β₂)*sign(g²-v)*g²
    """
    g_sq = g**2
    sign_term = np.sign(g_sq - v_old)
    v_new = v_old * beta2 + (1 - beta2) * sign_term * g_sq
    return v_new

# 测试多组数据
test_cases = [
    (1.0, 0.5, 0.99),   # v_old=1.0, g=0.5, beta2=0.99
    (0.1, 1.5, 0.99),   # v_old < g²
    (2.0, 1.0, 0.99),   # v_old > g²
    (1.0, 1.0, 0.99),   # v_old = g²
    (0.5, 0.3, 0.92),
]

print("=" * 80)
print("ANO v-update 公式等价性验证")
print("=" * 80)
print()

all_match = True
for v_old, g, beta2 in test_cases:
    paper_result = paper_v_update(v_old, g, beta2)
    pytorch_result = pytorch_v_update(v_old, g, beta2)
    
    match = np.allclose(paper_result, pytorch_result, rtol=1e-10)
    all_match = all_match and match
    
    print(f"v_old={v_old:.2f}, g={g:.2f}, g²={g**2:.2f}, β₂={beta2}")
    print(f"  论文公式:    v_new = {paper_result:.6f}")
    print(f"  PyTorch实现: v_new = {pytorch_result:.6f}")
    print(f"  差异: {abs(paper_result - pytorch_result):.2e}")
    print(f"  匹配: {'✓ YES' if match else '✗ NO'}")
    print()

print("=" * 80)
if all_match:
    print("结论: 两个公式在数学上 **完全等价** ✓")
else:
    print("结论: 两个公式 **不等价** ✗")
print("=" * 80)
