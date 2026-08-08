# differentiable_identify 函数详解

## 概述

`differentiable_identify` 用**梯度下降**替代经典最小二乘来辨识摩擦参数。核心链路：

```
定义可微模型 → 自动求梯度 → Adam 迭代优化 → 输出参数
```

和经典 LS 的本质区别：不需要手推回归矩阵 W，只需要写一个 forward 函数 `friction_model(phi, dq, ddv)`，梯度由库自动完成。

---

## 整体数据流

```
输入 dq, ddv, friction_target  — 都是 (T, 12) 的 NumPy 数组

    ↓  jnp.array(..., dtype=float32)

JAX 数组 dq_jax, ddv_jax, target_jax

    ↓  定义 loss_fn(phi)
    │    ├─ jax.vmap: 沿时间轴 T 并行调用 friction_model
    │    └─ MSE: 预测值 vs target_jax
    ↓
jax.grad(loss_fn) → grad_fn     ← 自动求梯度
jax.jit(loss_fn)   → loss_jit   ← JIT 编译加速

    ↓  Adam 循环 n_steps 次

phi_final → 转为 NumPy 数组返回
```

---

## JAX 库函数详解

### 1. `jax.grad` — 自动微分

```python
grad_fn = jax.grad(loss_fn)
```

**原理**：不依赖解析求导或数值差分。JAX 跟踪 `loss_fn` 内部所有 Python 操作，构建计算图，然后沿图反向传播算出 loss 对 phi 的精确梯度。

**自动微分 vs 其他方式**：

| 方式 | 做法 | 问题 |
|------|------|------|
| 解析推导 | 人手推 d(loss)/d(phi) = 2 W^T (W phi - Y) / n | 换模型就要重推，易出错 |
| 数值差分 | (f(x+h) - f(x)) / h | 精度只有 ~1e-8，慢（每个参数要算两次 f） |
| **自动微分** | JAX/grad 自动算 | 机器精度（~1e-16），速度接近解析解 |

在这个摩擦问题中，`loss_fn` 里所有操作——乘法、`tanh`、减法、平方——JAX 都知道怎么求导，所以 `grad_fn(phi)` 直接返回长度为 24（或 36）的精确梯度向量。

**调用示例**：
```python
g = grad_fn(phi)    # g 的形状 = phi 的形状 = (24,)
                     # g[i] = d(loss)/d(phi[i])，精确到机器精度
```

---

### 2. `jax.jit` — JIT 编译

```python
grad_fn = jax.jit(jax.grad(loss_fn))
loss_jit = jax.jit(loss_fn)
```

**原理**：JIT（Just-In-Time）将 Python 函数编译为 XLA（加速线性代数）计算图。第一次调用时 JAX 追踪函数的所有运算、编译成优化后的机器码，之后调用直接执行编译好的代码。

**为什么需要**：

- **第一次调用**：追踪 + 编译，耗时 ~100~500ms
- **后续调用**：跳过 Python 解释器，直接跑底层 kernel，快 10~100 倍

对于 2000 步的循环，第一次编译开销完全被后续加速覆盖，总体远快于纯 Python 循环。

**注意事项**：
- JIT 要求函数内部所有数组形状在编译后不变——数据维度必须固定
- Python 控制流（if/for）在追踪时展开为静态图，条件判断依赖的值会被当作常量固化

---

### 3. `jax.vmap` — 自动向量化

```python
pred = jax.vmap(lambda dq, ddv: friction_model(phi, dq, ddv))(dq_jax, ddv_jax)
```

**原理**：`vmap` 将单样本函数自动转换为批量函数。输入是 (T, 12) 的三维张量，`vmap` 沿第 0 维（时间轴）并行展开，对每个时间步 t 调用 `friction_model(phi, dq[t], ddv[t])`。

**等价于**：
```python
# 手动循环版本（慢）
pred = []
for t in range(T):
    pred.append(friction_model(phi, dq[t], ddv[t]))
pred = jnp.stack(pred)

# vmap 版本（快，且可被 JIT 一并编译）
pred = jax.vmap(lambda dq, ddv: friction_model(phi, dq, ddv))(dq, ddv)
```

**在此处的角色**：

- `friction_model` 设计为处理**单个时间步**（输入 dq 是 (12,)，输出也是 (12,)）
- `vmap` 把它沿时间轴 T 映射，一次调用输出 (T, 12)
- 和 `jax.grad` + `jax.jit` 组合：梯度计算也自动向量化，无需手动写批量梯度

---

### 4. `jnp.array(..., dtype=jnp.float32)` — 数据类型转换

```python
dq_jax = jnp.array(dq_joint, dtype=jnp.float32)
```

**原理**：将 NumPy 的默认 float64 转为 float32。

**为什么**：
- JAX 的 XLA 编译器对 float32 有更好的 kernel 优化
- GPU/TPU 上 float32 吞吐量是 float64 的 2~16 倍
- 摩擦辨识问题的精度需求不需要 float64 的 15 位有效数字

---

## Optax 库函数详解

### 5. `optax.adam` — Adam 优化器

```python
opt = optax.adam(learning_rate=lr)
```

**Adam 原理**：结合了 Momentum 和 RMSProp 的优点，对每个参数维护两个历史统计量：

| 变量 | 含义 | 作用 |
|------|------|------|
| 一阶矩 m_t | 梯度的指数移动平均（EMA） | 提供动量，加速收敛，平滑振荡 |
| 二阶矩 v_t | 梯度平方的 EMA | 自适应调整学习率——大梯度方向减小步长，小梯度方向增大 |

**更新公式**：

```
m_t = b1 * m_{t-1} + (1-b1) * g_t       // 梯度平滑
v_t = b2 * v_{t-1} + (1-b2) * g_t^2     // 平方梯度平滑

m_hat = m_t / (1 - b1^t)                // 偏差修正（初期步 m_0=0 会被低估）
v_hat = v_t / (1 - b2^t)

phi = phi - lr * m_hat / (sqrt(v_hat) + eps)
```

**默认超参**：b1=0.9, b2=0.999, eps=1e-8。大多数场景无需调整。

**为什么用 Adam 而非朴素 SGD**：

| 方法 | 问题 |
|------|------|
| SGD (phi -= lr * g) | 学习率对所有参数相同，不同参数的最优 lr 差几个数量级时会卡住 |
| SGD + Momentum | 缓解方向震荡，但学习率仍需手动调 |
| **Adam** | 每个参数独立自适应学习率，初值不敏感，收敛稳定 |

在本问题中，b（粘滞系数 ~0.1）和 fc（库仑摩擦 ~0.5）的量级不同，`tanh` 的非线性让不同参数的梯度规模也不同——Adam 的自适应特性恰好解决了这个问题。

---

### 6. `opt.init` — 初始化优化器状态

```python
opt_state = opt.init(phi)
```

创建 Adam 的内部状态，包括：
- 一阶矩 m：与 phi 同形状的零数组
- 二阶矩 v：与 phi 同形状的零数组
- 步数计数器 step：整数 0

优化器状态需要**在循环外初始化一次，然后在循环内持续更新**。

---

### 7. `opt.update` — 计算更新量

```python
updates, opt_state = opt.update(g, opt_state)
```

**作用**：输入当前梯度 g 和当前优化器状态，输出：
- `updates`：与 phi 同形状的参数变化量 Δphi（已包含学习率和自适应缩放）
- 新的 `opt_state`：更新后的 m、v 和步数计数器

注意 `opt.update` **不修改参数本身**，只输出应该加上的差值。这种设计将"计算怎么改"和"真的改"解耦，便于实现梯度裁剪、学习率调度等中间操作。

---

### 8. `optax.apply_updates` — 应用更新

```python
phi = optax.apply_updates(phi, updates)
```

等价于 `phi = phi + updates`（某些优化器用更复杂的合并逻辑，但 Adam 就是加法）。

---

## 训练循环详解

```python
for step in range(n_steps):
    g = grad_fn(phi)                          # ①
    loss_val = loss_jit(phi)                  # ②

    updates, opt_state = opt.update(g, opt_state)  # ③
    phi = optax.apply_updates(phi, updates)        # ④
    phi = jnp.maximum(phi, 0.0)                    # ⑤

    loss_np = float(loss_val)                 # ⑥
    grad_norm = float(jnp.linalg.norm(g))
    loss_history.append(loss_np)
    grad_norm_history.append(grad_norm)
```

| 步骤 | 操作 | 作用 |
|------|------|------|
| ① | `grad_fn(phi)` | JIT 编译的自动微分，算 d(loss)/d(phi)，O(p) 时间 |
| ② | `loss_jit(phi)` | JIT 编译的 loss 计算，仅用于打印和记录 |
| ③ | `opt.update` | Adam 内部：更新 EMA，输出缩放后的更新量 |
| ④ | `apply_updates` | phi = phi - lr * m_hat / sqrt(v_hat) |
| ⑤ | `jnp.maximum` | 非负投影，确保摩擦系数物理上 ≥ 0 |
| ⑥ | `float()` | 将 JAX 标量转 Python 标量，脱离计算图防止 OOM |

**注意**：`float(loss_val)` 和 `float(jnp.linalg.norm(g))` 很关键——JAX 的 JIT 会记住整条计算图用于下次编译，如果在 `loss_history` 里直接存 JAX 数组引用，计算图会随循环积累导致内存爆炸。`float()` 脱离图，只存一个数字。

---

## 和经典 LS 的对比总结

| 维度 | 经典 LS (least_squares_identify) | 可微方法 (differentiable_identify) |
|------|------|------|
| 建模 | 手工推导 W 矩阵（36 列） | 只写 forward 函数（10 行） |
| 求解 | 一步解正规方程 | 迭代 2000 步 Adam |
| 梯度 | 解析（隐含在 solve 中） | 自动微分 (`jax.grad`) |
| 换模型代价 | 重推 W | 只改 `friction_model` 函数体 |
| 扩展性 | 要求参数线性 | 非线性的 LuGre/Stribeck 也直接支持 |
| 速度 | T≈数百步时 ~ms | JIT 编译后单步 ~微秒，2000 步 ~1s |
| 精度 | 精确解（正则化后） | 近似解（收敛到局部最优） |
