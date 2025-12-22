"""
Verify the mathematical equivalence between the paper formula and Adrienkgz's PyTorch implementation
"""
import numpy as np

def paper_v_update(v_old, g, beta2):
    """
    Paper formula: v_k = β₂*v_{k-1} - (1-β₂)*sign(v_{k-1} - g_k²)*g_k²
    
    Note: The paper states v_k = β₂*v_{k-1} - ..., not v_k = v_{k-1} - ...
    """
    g_sq = g**2
    sign_term = np.sign(v_old - g_sq)
    v_new = beta2 * v_old - (1 - beta2) * sign_term * g_sq
    return v_new

def pytorch_v_update(v_old, g, beta2):
    """
    PyTorch implementation: v = v*β₂ + (1-β₂)*sign(g²-v)*g²
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

all_match = True
for v_old, g, beta2 in test_cases:
    paper_result = paper_v_update(v_old, g, beta2)
    pytorch_result = pytorch_v_update(v_old, g, beta2)
    
    match = np.allclose(paper_result, pytorch_result, rtol=1e-10)
    all_match = all_match and match
    
    print(f"v_old={v_old:.2f}, g={g:.2f}, g²={g**2:.2f}, β₂={beta2}")
    print(f"  Paper formula:    v_new = {paper_result:.6f}")
    print(f"  PyTorch implementation: v_new = {pytorch_result:.6f}")
    print(f"  Difference: {abs(paper_result - pytorch_result):.2e}")
    print(f"  Match: {'YES' if match else 'NO'}")
    print()


if all_match:
    print("mathematically equivalent")
else:
    print("not equivalent")

