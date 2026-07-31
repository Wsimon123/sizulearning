# 广义质量矩阵 M 的完整推导与结构

## 1. 调用接口

代码位置：[freefall_verification.py:110-111](freefall_verification.py#L110-L111)

```python
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)
```

MuJoCo 内部的 `mj_fullM` 使用 **CRBA (Composite Rigid Body Algorithm)** 计算整个系统的广义质量矩阵。

### 物理定义

由系统动能：

$$
K = \frac{1}{2} \dot{\mathbf{q}}^\top \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}
$$

其中：

$$
\dot{\mathbf{q}} = \begin{bmatrix}
\mathbf{v}_b \\ \boldsymbol{\omega}_b \\ \dot{\mathbf{q}}_j
\end{bmatrix}_{18\times 1}
$$

- $\mathbf{v}_b = [v_x, v_y, v_z]^\top$ — 基座线速度 (3D)
- $\boldsymbol{\omega}_b = [\omega_x, \omega_y, \omega_z]^\top$ — 基座角速度 (3D)
- $\dot{\mathbf{q}}_j = [\dot{q}_1, \dot{q}_2, \dots, \dot{q}_{12}]^\top$ — 12 个关节角速度

**注**：位置空间 `nq = 19`（3 平移 + 4 四元数 + 12 关节角），但速度空间 `nv = 18`（3 线速度 + 3 角速度 + 12 关节速度），$\mathbf{M} \in \mathbb{R}^{18\times 18}$。

---

## 2. 分块结构

$$
\mathbf{M} = \begin{bmatrix}
\mathbf{M}_{bb} & \mathbf{M}_{b\theta} \\
\mathbf{M}_{\theta b} & \mathbf{M}_{\theta\theta}
\end{bmatrix}_{18\times 18}
$$

| 块 | 维度 | 含义 |
|----|------|------|
| $\mathbf{M}_{bb}$ | 6×6 | 基座惯性：总质量 + 空间转动惯量 |
| $\mathbf{M}_{b\theta}$ | 6×12 | 基座速度与关节加速度的惯性耦合 |
| $\mathbf{M}_{\theta b}$ | 12×6 | $\mathbf{M}_{b\theta}^\top$（对称性保证） |
| $\mathbf{M}_{\theta\theta}$ | 12×12 | 各关节的有效惯量及关节间耦合 |

---

## 3. 基座块 $\mathbf{M}_{bb}$ (6×6)

### 完整矩阵形式

$$
\mathbf{M}_{bb} = \begin{bmatrix}
m & 0 & 0 & 0 & m\,\bar{z}_c & -m\,\bar{y}_c \\
0 & m & 0 & -m\,\bar{z}_c & 0 & m\,\bar{x}_c \\
0 & 0 & m & m\,\bar{y}_c & -m\,\bar{x}_c & 0 \\
0 & -m\,\bar{z}_c & m\,\bar{y}_c & \bar{I}_{xx} & \bar{I}_{xy} & \bar{I}_{xz} \\
m\,\bar{z}_c & 0 & -m\,\bar{x}_c & \bar{I}_{yx} & \bar{I}_{yy} & \bar{I}_{yz} \\
-m\,\bar{y}_c & m\,\bar{x}_c & 0 & \bar{I}_{zx} & \bar{I}_{zy} & \bar{I}_{zz}
\end{bmatrix}
$$

### 元素物理含义

#### 左上 3×3：平移惯性

$$
\mathbf{M}_{bb}[0:3, 0:3] = m \; \mathbb{I}_3
$$

| 元素 | 符号 | 含义 |
|------|------|------|
| $M[0,0]$ | $m$ | 基座 x 方向平动惯量 = 系统总质量 |
| $M[1,1]$ | $m$ | 基座 y 方向平动惯量 = 系统总质量 |
| $M[2,2]$ | $m$ | 基座 z 方向平动惯量 = 系统总质量 |

> 代码中通过 `total_mass = sum(model.body_mass)` 计算总质量作为参考值。

#### 右上/左下 3×3：质心耦合项（反对称）

$$
\mathbf{M}_{bb}[0:3, 3:6] = m \, [\bar{\mathbf{c}}]_\times = \begin{bmatrix}
0 & m\bar{z}_c & -m\bar{y}_c \\
-m\bar{z}_c & 0 & m\bar{x}_c \\
m\bar{y}_c & -m\bar{x}_c & 0
\end{bmatrix}
$$

其中 $\bar{\mathbf{c}} = [\bar{x}_c, \bar{y}_c, \bar{z}_c]^\top$ 是**系统质心在基座坐标系下的位置**（即从基座原点指向系统质心的向量，对每个连杆质心按质量加权平均）。

$[\bar{\mathbf{c}}]_\times$ 是反对称叉乘矩阵：

$$
[\bar{\mathbf{c}}]_\times = \begin{bmatrix}
0 & -\bar{z}_c & \bar{y}_c \\
\bar{z}_c & 0 & -\bar{x}_c \\
-\bar{y}_c & \bar{x}_c & 0
\end{bmatrix}
$$

物理意义：基座旋转会带动质心线运动，这部分耦合反映了"质心不在基座原点"的事实。

$$
\mathbf{M}_{bb}[3:6, 0:3] = \mathbf{M}_{bb}[0:3, 3:6]^\top = -m\,[\bar{\mathbf{c}}]_\times
$$

#### 右下 3×3：空间转动惯量

$$
\mathbf{M}_{bb}[3:6, 3:6] = \bar{\mathbf{I}} = \begin{bmatrix}
\bar{I}_{xx} & \bar{I}_{xy} & \bar{I}_{xz} \\
\bar{I}_{yx} & \bar{I}_{yy} & \bar{I}_{yz} \\
\bar{I}_{zx} & \bar{I}_{zy} & \bar{I}_{zz}
\end{bmatrix}
$$

$\bar{\mathbf{I}}$ 是系统关于**基座原点**的总转动惯量，由每个连杆 $k$ 的转动惯量 $\mathbf{I}_k$ 加上 Steiner 平行轴定理贡献：

$$
\bar{\mathbf{I}} = \sum_{k} \left( \mathbf{I}_k + m_k \left( \|\mathbf{r}_k\|^2 \mathbb{I}_3 - \mathbf{r}_k \mathbf{r}_k^\top \right) \right)
$$

其中 $\mathbf{r}_k = \mathbf{c}_k - \mathbf{c}_b$ 是连杆质心到基座原点的矢量，$\mathbf{I}_k$ 是连杆关于自身质心的转动惯量。

---

## 4. 基座-关节耦合块 $\mathbf{M}_{b\theta}$ (6×12)

$$
\mathbf{M}_{b\theta} = \begin{bmatrix}
M_{v_x,q_1} & M_{v_x,q_2} & \cdots & M_{v_x,q_{12}} \\
M_{v_y,q_1} & M_{v_y,q_2} & \cdots & M_{v_y,q_{12}} \\
M_{v_z,q_1} & M_{v_z,q_2} & \cdots & M_{v_z,q_{12}} \\
M_{\omega_x,q_1} & M_{\omega_x,q_2} & \cdots & M_{\omega_x,q_{12}} \\
M_{\omega_y,q_1} & M_{\omega_y,q_2} & \cdots & M_{\omega_y,q_{12}} \\
M_{\omega_z,q_1} & M_{\omega_z,q_2} & \cdots & M_{\omega_z,q_{12}}
\end{bmatrix}
$$

### 物理含义

元素 $M_{\dot{q}_i, \dot{q}_j}$ 表示：**关节 $j$ 加速度为 1 时，在自由度 $i$ 方向产生的广义惯性力**。

通过雅可比矩阵的等价形式：

$$
\mathbf{M}_{b\theta} = \sum_k m_k \, \mathbf{J}_{v,k}^\top \, \mathbf{I}_k^{\text{spatial}} \, \mathbf{J}_{v,k}^{\theta}
$$

其中 $\mathbf{J}_{v,k}$ 是连杆 $k$ 的 6D 速度雅可比矩阵。

### 具体展开（按关节排列）

速度下标定义如下（对应 [freefall_verification.py](freefall_verification.py) 中 `q0[7:19]` 的顺序）：

| 索引 | 速度符号 | XML 关节 | 所属腿 |
|------|---------|---------|--------|
| 0 | $v_x$ | (基座平动) | — |
| 1 | $v_y$ | (基座平动) | — |
| 2 | $v_z$ | (基座平动) | — |
| 3 | $\omega_x$ | (基座转动) | — |
| 4 | $\omega_y$ | (基座转动) | — |
| 5 | $\omega_z$ | (基座转动) | — |
| 6 | $\dot{q}_1$ | FAR_ABAD_JOINT | FAR (前右) |
| 7 | $\dot{q}_2$ | FAR_HIP_JOINT | FAR (前右) |
| 8 | $\dot{q}_3$ | FAR_KNEE_JOINT | FAR (前右) |
| 9 | $\dot{q}_4$ | FBL_ABAD_JOINT | FBL (前左) |
| 10 | $\dot{q}_5$ | FBL_HIP_JOINT | FBL (前左) |
| 11 | $\dot{q}_6$ | FBL_KNEE_JOINT | FBL (前左) |
| 12 | $\dot{q}_7$ | RAR_ABAD_JOINT | RAR (后右) |
| 13 | $\dot{q}_8$ | RAR_HIP_JOINT | RAR (后右) |
| 14 | $\dot{q}_9$ | RAR_KNEE_JOINT | RAR (后右) |
| 15 | $\dot{q}_{10}$ | RBL_ABAD_JOINT | RBL (后左) |
| 16 | $\dot{q}_{11}$ | RBL_HIP_JOINT | RBL (后左) |
| 17 | $\dot{q}_{12}$ | RBL_KNEE_JOINT | RBL (后左) |

### 耦合元素的具体含义

以 FAR 腿为例：

| 元素 | 符号 | 物理含义 |
|------|------|----------|
| $M[v_x, q_1]$ | $M_{v_x, \dot{q}_{FA}}$ | FAR 侧摆关节加速 → 基座 x 方向反力 |
| $M[v_z, q_2]$ | $M_{v_z, \dot{q}_{FH}}$ | FAR 大腿加速 → 基座 z 方向反力 |
| $M[\omega_y, q_3]$ | $M_{\omega_y, \dot{q}_{FK}}$ | FAR 膝关节加速 → 基座绕 y 轴扭矩 |

---

## 5. 关节-关节块 $\mathbf{M}_{\theta\theta}$ (12×12)

### 完整展开

$$
\mathbf{M}_{\theta\theta} = \begin{bmatrix}
M_{1,1} & M_{1,2} & M_{1,3} & M_{1,4} & M_{1,5} & M_{1,6} & M_{1,7} & M_{1,8} & M_{1,9} & M_{1,10} & M_{1,11} & M_{1,12} \\
M_{2,1} & M_{2,2} & M_{2,3} & M_{2,4} & M_{2,5} & M_{2,6} & M_{2,7} & M_{2,8} & M_{2,9} & M_{2,10} & M_{2,11} & M_{2,12} \\
M_{3,1} & M_{3,2} & M_{3,3} & M_{3,4} & M_{3,5} & M_{3,6} & M_{3,7} & M_{3,8} & M_{3,9} & M_{3,10} & M_{3,11} & M_{3,12} \\
M_{4,1} & M_{4,2} & M_{4,3} & M_{4,4} & M_{4,5} & M_{4,6} & M_{4,7} & M_{4,8} & M_{4,9} & M_{4,10} & M_{4,11} & M_{4,12} \\
M_{5,1} & M_{5,2} & M_{5,3} & M_{5,4} & M_{5,5} & M_{5,6} & M_{5,7} & M_{5,8} & M_{5,9} & M_{5,10} & M_{5,11} & M_{5,12} \\
M_{6,1} & M_{6,2} & M_{6,3} & M_{6,4} & M_{6,5} & M_{6,6} & M_{6,7} & M_{6,8} & M_{6,9} & M_{6,10} & M_{6,11} & M_{6,12} \\
M_{7,1} & M_{7,2} & M_{7,3} & M_{7,4} & M_{7,5} & M_{7,6} & M_{7,7} & M_{7,8} & M_{7,9} & M_{7,10} & M_{7,11} & M_{7,12} \\
M_{8,1} & M_{8,2} & M_{8,3} & M_{8,4} & M_{8,5} & M_{8,6} & M_{8,7} & M_{8,8} & M_{8,9} & M_{8,10} & M_{8,11} & M_{8,12} \\
M_{9,1} & M_{9,2} & M_{9,3} & M_{9,4} & M_{9,5} & M_{9,6} & M_{9,7} & M_{9,8} & M_{9,9} & M_{9,10} & M_{9,11} & M_{9,12} \\
M_{10,1} & M_{10,2} & M_{10,3} & M_{10,4} & M_{10,5} & M_{10,6} & M_{10,7} & M_{10,8} & M_{10,9} & M_{10,10} & M_{10,11} & M_{10,12} \\
M_{11,1} & M_{11,2} & M_{11,3} & M_{11,4} & M_{11,5} & M_{11,6} & M_{11,7} & M_{11,8} & M_{11,9} & M_{11,10} & M_{11,11} & M_{11,12} \\
M_{12,1} & M_{12,2} & M_{12,3} & M_{12,4} & M_{12,5} & M_{12,6} & M_{12,7} & M_{12,8} & M_{12,9} & M_{12,10} & M_{12,11} & M_{12,12}
\end{bmatrix}
$$

其中 $M_{i,j}$ 是关节 $i$（速度索引 $i+5$，位置索引 $i+6$）和关节 $j$（速度索引 $j+5$，位置索引 $j+6$）之间的惯性耦合。

### 按腿分块结构

$$
\mathbf{M}_{\theta\theta} = \begin{bmatrix}
\mathbf{M}_{\text{FAR}} & \mathbf{M}_{\text{FAR-FBL}} & \mathbf{M}_{\text{FAR-RAR}} & \mathbf{M}_{\text{FAR-RBL}} \\
\mathbf{M}_{\text{FBL-FAR}} & \mathbf{M}_{\text{FBL}} & \mathbf{M}_{\text{FBL-RAR}} & \mathbf{M}_{\text{FBL-RBL}} \\
\mathbf{M}_{\text{RAR-FAR}} & \mathbf{M}_{\text{RAR-FBL}} & \mathbf{M}_{\text{RAR}} & \mathbf{M}_{\text{RAR-RBL}} \\
\mathbf{M}_{\text{RBL-FAR}} & \mathbf{M}_{\text{RBL-FBL}} & \mathbf{M}_{\text{RBL-RAR}} & \mathbf{M}_{\text{RBL}}
\end{bmatrix}
$$

### 单腿块 $\mathbf{M}_{\text{FAR}}$ (3×3) — 串联链

FAR 腿的运动学链是 `torso → FAR_ABAD → FAR_HIP → FAR_KNEE`，CRBA 的反向传播产生三对角结构：

$$
\mathbf{M}_{\text{FAR}} = \begin{bmatrix}
M_{\text{F}A,\text{F}A} & M_{\text{F}A,\text{F}H} & M_{\text{F}A,\text{F}K} \\
M_{\text{F}H,\text{F}A} & M_{\text{F}H,\text{F}H} & M_{\text{F}H,\text{F}K} \\
M_{\text{F}K,\text{F}A} & M_{\text{F}K,\text{F}H} & M_{\text{F}K,\text{F}K}
\end{bmatrix}
$$

| 元素 | 物理含义 |
|------|----------|
| $M_{\text{F}A,\text{F}A}$ | ABAD 关节自身的转动惯量（HIP 和 KNEE 被锁定） |
| $M_{\text{F}H,\text{F}H}$ | HIP 关节自身的转动惯量（ABAD 和 KNEE 被锁定） |
| $M_{\text{F}K,\text{F}K}$ | KNEE 关节自身的转动惯量（ABAD 和 HIP 被锁定） |
| $M_{\text{F}A,\text{F}H}$ | HIP 加速对 ABAD 产生的反扭矩 = ABAD 加速对 HIP 产生的反扭矩 |
| $M_{\text{F}A,\text{F}K}$ | KNEE 加速对 ABAD 产生的反扭矩（通过 HIP 传递，数值较小） |
| $M_{\text{F}H,\text{F}K}$ | KNEE 加速对 HIP 产生的反扭矩（直接串联耦合） |

其余三条腿的结构完全对称。

### 跨腿耦合块（3×3，非零但较小）

以 $\mathbf{M}_{\text{FAR-FBL}}$ 为例：

$$
\mathbf{M}_{\text{FAR-FBL}} = \begin{bmatrix}
M_{\text{F}A,\text{L}A} & M_{\text{F}A,\text{L}H} & M_{\text{F}A,\text{L}K} \\
M_{\text{F}H,\text{L}A} & M_{\text{F}H,\text{L}H} & M_{\text{F}H,\text{L}K} \\
M_{\text{F}K,\text{L}A} & M_{\text{F}K,\text{L}H} & M_{\text{F}K,\text{L}K}
\end{bmatrix}
$$

这些元素通常比同腿耦合小 1-2 个数量级，因为它们**仅通过躯干的转动来传递耦合**，而非直接机械连接。

---

## 6. CRBA 计算流程

MuJoCo 内部按以下步骤计算 $\mathbf{M}$：

### Step 1: 构建每个连杆的空间惯性

对每个刚体 $k$，构造 6×6 空间惯性矩阵：

$$
\mathbf{I}_k^{\text{spatial}} = \begin{bmatrix}
m_k \mathbb{I}_3 & -m_k [\mathbf{c}_k]_\times \\
m_k [\mathbf{c}_k]_\times & \mathbf{I}_k - m_k [\mathbf{c}_k]_\times^2
\end{bmatrix}
$$

其中 $\mathbf{c}_k$ 是连杆 $k$ 质心在其自身坐标系中的位置，$[\mathbf{c}_k]_\times$ 是叉乘反对称矩阵。

### Step 2: 反向累积 (Backward Pass)

从运动学树的叶节点向根节点传播，在每个关节 $i$ 将其子树（包括自己）的空间惯性"反射"到自身坐标系：

$$
\mathbf{I}_i^{\text{comp}} = \underbrace{\mathbf{I}_i^{\text{spatial}}}_{\text{自身体惯性}} + \sum_{j \in \text{child}(i)} \underbrace{\mathbf{X}_j^{i\,\top} \; \mathbf{I}_j^{\text{comp}} \; \mathbf{X}_j^i}_{\text{子树 j 的惯性反射到节点 i}}
$$

其中 $\mathbf{X}_j^i$ 是 6×6 的空间变换矩阵（从坐标系 $j$ 变换到坐标系 $i$），包含旋转和位置的贡献。

### Step 3: 填充矩阵 (Forward / CRBA Fill)

对每对关节 $(i, j)$，找到它们的最小公共祖先（LCA），利用 Step 2 的复合惯性计算耦合元素：

$$
M_{ij} = \mathbf{s}_i^\top \; \mathbf{I}_{\text{LCA}(i,j)}^{\text{comp}} \; \mathbf{s}_j = \sum_{k \in \nu(i) \cap \nu(j)} \mathbf{s}_i^\top \mathbf{X}_k^{i\,\top} \mathbf{I}_k^{\text{spatial}} \mathbf{X}_k^j \mathbf{s}_j
$$

其中：
- $\mathbf{s}_i$ 是关节 $i$ 的 6D 运动子空间（转动关节为 $[0,0,0, a_x,a_y,a_z]^\top$，平动关节为 $[a_x,a_y,a_z, 0,0,0]^\top$）
- $\mathbf{s}_i^\top \mathbf{I}^{\text{comp}} \mathbf{s}_j$ 将 6×6 的空间惯性投影到两个关节的运动方向上
- $\nu(i)$ 是关节 $i$ 支撑的连杆集合（即该关节在运动学树上影响的所有子树连杆）

**算法复杂度**：$O(n^3)$ 对于一般的运动学树（n 为关节数），因为需要对所有关节对 $(i,j)$ 计算。对于串联链退化为 $O(n^2)$，对于纯串联可以通过 $O(n)$ 的 ABA (Articulated Body Algorithm) 优化，但 CRBA 适用于含浮动基座的一般树结构。

---

## 7. 代码中的验证

[freefall_verification.py](freefall_verification.py) 对 M 做了三个物理正确性检查：

| 检查 | 代码 | 物理依据 |
|------|------|----------|
| 对称性 | `max|M - Mᵀ| < 1e-10` | 动能是标量，M 是 Hessian: $M_{ij} = \frac{\partial^2 K}{\partial \dot{q}_i \partial \dot{q}_j}$，必须对称 |
| 正定性 | `min(eigvals(M)) > 0` | 动能 $K \geq 0$，且仅当 $\dot{\mathbf{q}} = 0$ 时 $K = 0$，M 正定 |
| 平动惯量 | `M[0,0] ≈ total_mass` | 基座 x 方向的平动惯量应等于系统总质量（同理 M[1,1], M[2,2]） |

---

## 8. 引用关系

- [freefall_verification.py](freefall_verification.py): 主验证脚本，第 110-111 行调用 `mj_fullM`
- [xg_freefall.xml](resources/xg/xg_freefall.xml): MJCF 模型文件，定义了本体 + 4 条腿的质量分布


## 9. M[0,0] 等于总质量而不是基座质量，原因在于 广义质量矩阵的定义。

直觉理解
想象你在太空里，旁边漂浮着一把折叠椅：


你抓住椅子的一角，用力一推
  → 你推的只是"一个角"，但整把椅子都在动
  → 你感受到的惯性 = 整把椅子的质量，不是那个角的质量
基座（torso）就是那个"角"——通过关节，所有腿都挂在基座上。推基座 = 推整个机器人。

数学推导
广义质量矩阵从动能定义：


T = ½ Σ m_i · |v_i|² + ½ Σ ω_iᵀ · I_i · ω_i
  = ½ q̇ᵀ · M(q) · q̇
当基座沿 x 方向纯平移时（q̇ = [v_x, 0, 0, 0, 0, 0, 0, ...]）：


每条腿没有关节驱动力 → 腿跟随基座刚性平移
所有 17 个 body 都以完全相同的 v_x 运动

T = ½ · (m₁ + m₂ + ... + m₁₇) · v_x²
  = ½ · (总质量) · v_x²
  = ½ · M[0,0] · v_x²

→ M[0,0] = 总质量 = 10.16 kg
如果是固定基座呢？
固定基座机器人（如工业机械臂）：

基座不运动 → 没有 M[0,0] 这个概念
关节 1 的 M[1,1] = 关节 1 + 后续连杆的转动惯量（不是总质量）
代码中的数据验证

M[0,0] = 10.16 kg  ≈ 总质量 10.16 kg  ← 基座 x 平移时，整体平移
M[1,1] = 10.16 kg  ≈ 总质量           ← y 方向同理
M[2,2] = 10.16 kg  ≈ 总质量           ← z 方向同理
如果测的是基座转动分量（M[3,3]~M[5,5]），那就不是总质量了——那是转动惯量，取决于质量分布和姿态。

# 条件数
对，就是 cond(M)，针对广义质量矩阵 M(q) 算的条件数。


## 代码里就这两步：
mujoco.mj_fullM(model, data, M)          # 1. 算出 18×18 的 M 矩阵
cond = max(eigvals(M)) / min(eigvals(M)) # 2. 最大特征值 ÷ 最小特征值
条件数描述的是 M 这个矩阵本身的性质——它的"各向异性"有多强：平移整机（15 kg）vs 动一下膝盖（0.012），差了 1200 倍

怎么算？

cond(M) = 最大的特征值 ÷ 最小的特征值 = 14.98 ÷ 0.0124 ≈ 1200
对矩阵 M，找两个数：一个最大的"硬度"，一个最小的"硬度"，一除就行。

意义是什么？
条件数 = 解方程时误差被放大的最大倍数。


解方程: M · x = b

b 有误差 → x 的误差 ≤ cond(M) × (b 的误差)

所以 cond(M) = 1200 意味着：
  输入 0.1% 误差 → 输出最大可能偏 120%
一句话
条件数越大，矩阵越"病态"——小的输入扰动会在输出端产生大的摆动。就像一根很长的杠杆，短臂端轻轻一抖，长臂端就大幅度甩出去


你说得对，核心确实是"不同部件所需驱动力差 1200 倍"——这是 M 本身的性质。

至于之前提到的"误差放大"，那是这个物理事实带来的数值计算副作用：


求解加速度时:  q̈ = M⁻¹ · (τ - h)

如果一个小的计算误差 δ 出现在 τ 里:
  → 基座加速度偏差: δ / 15     ← 几乎没影响
  → 膝盖加速度偏差: δ / 0.012  ← 放大了 ~80 倍
两者关系是：正因为膝盖方向"轻"（λ_min 小），所以同样的力矩误差在膝盖上会产生更大的加速度偏差。这不是两个独立的概念，而是同一件事的两个角度：

角度	说辞
物理本质（你说的）	动膝盖 vs 动整机，所需力差 1200 倍
数值后果（报告里提的）	M 求逆时，误差被 λ_min 方向放大
本质是同一句：λ_min 太小 → 膝盖方向太"轻" → 力矩/计算的一点点误差就会让膝盖乱动。

你的理解是对的——这两个不是独立的两件事，我之前的表述把它们说得像两个独立结论了，容易让人混淆。