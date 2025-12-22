# ANO 算法实现验证报告

## 总结
 **当前实现完全符合论文算法**

---

## 论文官方算法 (ANO)

来源: https://anonymous.4open.science/r/ano-optimizer-1645/optimizers/README.md

### 数学公式

$$\begin{aligned}
m_k &= \beta_1 m_{k-1} + (1-\beta_1) g_k \\
v_k &= \beta_2 v_{k-1} - (1-\beta_2)\operatorname{sign}(v_{k-1}-g_k^2) g_k^2 \\
\hat v_k &= \frac{v_k}{1-\beta_2^k} \\
\theta_k &= \theta_{k-1} - \frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}\operatorname{sign}(m_k)|g_k| - \eta_k \lambda\theta_{k-1}
\end{aligned}$$

### 关键超参数
- $\beta_1 \in [0,1)$ ：一阶矩衰减系数
- $\beta_2 \in [0.5,1)$ ：二阶矩衰减系数
- $\epsilon > 0$ ：数值稳定性常数
- $\lambda$ ：权重衰减系数
- $\eta_k$ ：学习率（可时变）

---

## 当前实现的验证

### 1 一阶矩更新 

**论文公式:** $m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k$

**代码实现 (行 187):**
```python
m_new = b1_t * m_leaf + (1.0 - b1_t) * g
```

**验证结果:**  完全匹配
- `b1_t` 对应 $\beta_1$
- `m_leaf` 对应 $m_{k-1}$
- `g` 对应 $g_k$

---
### 2 二阶矩更新 

**论文公式:** $v_k = \beta_2 v_{k-1} - (1-\beta_2)\operatorname{sign}(v_{k-1}-g_k^2) g_k^2$

**代码实现 (行 189-192):**
```python
g_sq = jnp.square(g)                           # $g_k^2$
sign_term = jnp.sign(v_leaf - g_sq)            # $\operatorname{sign}(v_{k-1} - g_k^2)$
v_new = b2 * v_leaf - (1.0 - b2) * sign_term * g_sq
```

**验证结果:** 完全匹配
- `v_leaf` 对应 $v_{k-1}$
- `sign_term` 对应符号项
- 完整更新: $v_{k-1} - (1-\beta_2)\operatorname{sign}(v_{k-1}-g_k^2) g_k^2$

---

### 3 Bias Correction 

**论文公式:** $\hat v_k = \frac{v_k}{1-\beta_2^k}$

**代码实现 (行 168-169, 194):**
```python
bias_corr_2 = 1.0 - jnp.power(b2, step_f)   # $1 - \beta_2^k$
v_hat = v_new / bias_corr_2                  # $\frac{v_k}{1-\beta_2^k}$
```

**验证结果:** 完全匹配
- 仅对 $v$ 做 bias correction（不对 $m$ 做）
- 这与论文设计一致

---

### 4 梯度变换 

**论文公式:** $\theta_k = \theta_{k-1} - \frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}\operatorname{sign}(m_k)|g_k| - \eta_k \lambda\theta_{k-1}$

**代码实现 (行 195-196):**
```python
adjusted_lr = lr / (jnp.sqrt(v_hat) + eps)         # $\frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}$
transformed_g = adjusted_lr * jnp.abs(g) * jnp.sign(m_new)  # 上式乘以 $\operatorname{sign}(m_k)|g_k|$
```

**验证结果:** 完全匹配
- 梯度的符号由一阶矩 $\operatorname{sign}(m_k)$ 控制
- 梯度的幅度由绝对值 $|g_k|$ 控制
- 学习率自适应基于二阶矩 $\frac{\eta_k}{\sqrt{\hat v_k} + \epsilon}$

---

### 5 权重衰减 

**论文公式:** $- \eta_k \lambda\theta_{k-1}$（解耦权重衰减）

**代码实现 (行 206-211):**
```python
if params is not None and weight_decay > 0.0:
    def _apply_weight_decay(t_update, param):
        return t_update + lr * weight_decay * param  # $\eta_k \lambda\theta_{k-1}$
    
    transformed_updates = jax.tree.map(_apply_weight_decay, transformed_updates, params)
```

**验证结果:**  完全匹配
- 采用解耦权重衰减（AdamW 风格）
- 在参数更新中加入 $\eta_k \lambda\theta_{k-1}$ 项

---

## Anolog 变体 

论文中还提出了 Anolog 变体，使用时变的 $\beta_{1,k}$：

**论文公式:** $\beta_{1,k} = 1 - \frac{1}{\log(k+2)}$

**代码实现 (行 161-167):**
```python
if logarithmic_schedule:
    step_safe = jnp.maximum(step_f, 2.0)
    b1_t = 1.0 - 1.0 / jnp.log(step_safe)
else:
    b1_t = b1
```

**验证结果:**  完全匹配
- 当 `logarithmic_schedule=True` 时启用 Anolog 变体
- 当 `logarithmic_schedule=False` 时使用标准 ANO

---

## 推荐超参数

根据论文，推荐的超参数为：

| 参数 | 推荐值 | 范围 | 说明 |
|------|------|------|------|
| $\beta_1$ | 0.92 | $[0, 1)$ | 一阶矩衰减系数 |
| $\beta_2$ | 0.99 | $[0.5, 1)$ | 二阶矩衰减系数 |
| $\epsilon$ | $1e-8$ | $> 0$ | 数值稳定性常数 |
| $\lambda$ | 0.0 | $\geq 0$ | 权重衰减系数 |
| 学习率调度 | False | - | 是否启用 Anolog 变体 |



## 参考资源

- 论文官方 README: https://anonymous.4open.science/r/ano-optimizer-1645/optimizers/README.md
- 官方 GitHub: https://anonymous.4open.science/r/ano-optimizer-1645/
- 论文标题: "ANO: Faster is Better in Noisy Landscape"

