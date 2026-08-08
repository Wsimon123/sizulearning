# least_squares_identify 函数说明

## 概述

经典最小二乘辨识方法，用于从关节数据中估计摩擦模型参数。核心思想是将 N 个时间步、12 个关节的摩擦力观测值排列为向量 Y，手工构建回归矩阵 W，通过求解正规方程得到参数向量 `phi_hat`。

## 数学原理

摩擦模型假设摩擦力是参数的线性函数：

```
tau_friction = W * phi
```

其中 W 由关节速度和加速度构造，phi 为待辨识参数。三种模型层次：

| level | 公式 | 参数维度 p |
|-------|------|-----------|
| basic | tau_i = b_i · dθ_i | 12 |
| standard | tau_i = b_i · dθ_i + fc_i · tanh(k · dθ_i) | 24 |
| advanced | tau_i = b_i · dθ_i + fc_i · tanh(k · dθ_i) + Ir_i · ddθ_i | 36 |

参数说明：
- **b_i**（粘滞摩擦系数）：与速度成正比，单位 Nm/(rad/s)
- **fc_i**（库仑摩擦系数）：符号函数用 tanh(k·dθ) 平滑近似，k 为固定斜率常量
- **Ir_i**（惯量比）：与加速度成正比，反映转子惯量对关节力矩的贡献

## 输入参数

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| dq_joint | ndarray | (T, 12) | 12 个关节的速度序列 |
| ddv_joint | ndarray | (T, 12) | 12 个关节的加速度序列 |
| friction_target | ndarray | (T, 12) | 摩擦力真值（带噪声），从仿真中 qfrc_passive 提取 |
| level | str | — | 模型层次："basic" / "standard" / "advanced" |
| lam | float | — | Tikhonov 正则化系数，默认 1e-4 |

## 核心步骤

### 1. 数据展平（reshape）

```python
Y = friction_target.reshape(-1)
```

将 (T, 12) 的摩擦力矩阵按 C 顺序（行优先）展平为长度 T×12 的一维向量。展平后第 0 个元素对应 t=0 的 joint 0，第 1 个元素对应 t=0 的 joint 1，依此类推。

### 2. 构造回归矩阵（build_regressor_matrix）

W 的形状为 (T×12, p)，每行对应一个时间步的一个关节：

```
W 的行排布规律：
  row 0    → t=0, joint 0   (速度填充列 0)
  row 1    → t=0, joint 1   (速度填充列 1)
  ...
  row 11   → t=0, joint 11  (速度填充列 11)
  row 12   → t=1, joint 0   (速度填充列 0)
  ...
```

以 standard 为例，第 i 个关节对应的 W 行：
- 第 i 列 = dq_joint 的第 i 个分量
- 第 (12 + i) 列 = tanh(k · dq_joint 的第 i 个分量)

这种交叉编排保证了 W 的行与 Y 的元素一一对应。

### 3. 正规方程求解

```
目标: min ||W * phi - Y||^2 + λ||phi||^2
解:   phi = (W^T W + λI)^(-1) * W^T Y
```

代码实现：
```python
WtW = W.T @ W          # (p, p) 矩阵
WtY = W.T @ Y          # (p,) 向量
phi_hat = np.linalg.solve(WtW + lam * eye_p, WtY)
```

使用 `np.linalg.solve` 而非显式求逆 `inv()`，因为 solve 通过 LU/Cholesky 分解直接求解，数值更稳定。

### 4. 物理约束投影

```python
phi_hat = np.maximum(phi_hat, 0.0)
```

摩擦系数物理意义决定了它们必须非负——粘滞阻尼不可能推动运动，库仑摩擦方向一定与速度反向。将负值置零是合理的后处理。

### 5. 质量评估

```python
cond = np.linalg.cond(WtW)                    # 条件数
residual = np.mean((W @ phi_hat - Y) ** 2)    # MSE
```

- **条件数**：衡量数值稳定性，λ_max / λ_min。若 > 10^6 ~ 10^8，说明 W 的列近似线性相关（如速度和加速度高度相关），解不可靠。
- **MSE 残差**：预测值与真值的均方误差。越小拟合越好，但过小可能过拟合。

## 输入输出形状追踪

整个函数中数据形状的流转如下（以 standard 为例，p=24）：

```
d──T──┐                ┌──T──┐
       │ 12个关节       │      │ 12个关节
     dq_joint         ddv_joint
    (T, 12)           (T, 12)
         │                │
         └──────┬─────────┘
                ↓
     build_regressor_matrix()
                │   交叉编排：同一时间步的12个关节横向展开
                │   row 0→joint0, row 1→joint1, ..., row 11→joint11
                │   row 12→joint0, ...
                ↓
         ┌──────┴──────┐
         │   W          │    Y = friction_target.reshape(-1)
         │ (T×12, p)    │    (T×12,)    ← 同样按 C 顺序展平
         └──────┬───────┘         │
                │                 │
       W.T @ W  │        W.T @ Y  │
         (p,p)  │         (p,)    │
                ↓          ↓      │
         WtW + λI         WtY ────┘
           (p,p)           (p,)
                │           │
                └─────┬─────┘
                      ↓
              np.linalg.solve()
                      │
                      ↓
               phi_hat (p,)
                 → 12个b + 12个fc

        评估:
          cond = cond(WtW)     → 标量
          residual = MSE       → 标量
```

关键约束：W 的**行数**（T×12）必须等于 Y 的**长度**，且两者按同一顺序编排——C 顺序展平与交叉编排恰好匹配，保证第 k 行 W 对应的关节和时间步就是 Y 的第 k 个元素。

## 输出

| 输出 | 类型 | 形状 | 说明 |
|------|------|------|------|
| phi_hat | ndarray | (p,) | 辨识出的摩擦参数向量 |
| cond | float | — | W^T W 的 2-范数条件数 |
| residual | float | — | 拟合均方误差 |

## Tikhonov 正则化的作用

当数据噪声较大或 W 的列存在多重共线性时，W^T W 接近奇异。加上 λI 等价于对参数向量的 L2 范数施加惩罚，使得优化目标变为：

```
min ||W phi - Y||^2 + λ||phi||^2
```

效应：
- **收缩参数**：防止 phi 的某些分量爆炸
- **改善条件数**：最小特征值从接近 0 提升到至少 λ
- **偏差-方差权衡**：λ 越大偏差越大但方差越小；λ=1e-4 作为默认值在大部分场景下足够

## 局限性

1. **必须手工推导 W**：每换一种摩擦模型（如 LuGre、Stribeck），W 的结构完全不同
2. **线性假设**：要求摩擦形式关于参数线性，无法直接拟合非线性依赖关系
3. **噪声敏感**：数据中的离群点会影响最小二乘估计（L2 范数对异常值敏感）
4. **非负投影粗糙**：`max(phi, 0)` 只是后处理，并未在优化中引入约束
