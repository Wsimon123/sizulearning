# 广义质量矩阵 M 的计算方法

## 概述

MuJoCo 通过 **CRBA（Composite Rigid Body Algorithm，复合刚体算法）** 计算广义质量矩阵 $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{18 \times 18}$。

**核心思想**：对每对关节 $(i, j)$，$M_{ij}$ 等于——**将关节 $j$ 以单位加速度驱动时，在关节 $i$ 处感受到的广义惯性力**。

---

## 1. 物理定义

从系统动能出发：

$$K = \frac{1}{2} \dot{\mathbf{q}}^\top \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}$$

对速度求二阶偏导：

$$M_{ij} = \frac{\partial^2 K}{\partial \dot{q}_i \partial \dot{q}_j}$$

$\mathbf{M}$ 的分块结构（18×18）：

$$\mathbf{M} = \begin{bmatrix}
\mathbf{M}_{bb} & \mathbf{M}_{b\theta} \\
\mathbf{M}_{\theta b} & \mathbf{M}_{\theta\theta}
\end{bmatrix}$$

| 块 | 维度 | 含义 |
|----|------|------|
| $\mathbf{M}_{bb}$ | 6×6 | 基座惯性（平动 + 转动） |
| $\mathbf{M}_{b\theta}$ | 6×12 | 基座速度与关节加速度的惯性耦合 |
| $\mathbf{M}_{\theta b}$ | 12×6 | $\mathbf{M}_{b\theta}^\top$（对称性保证） |
| $\mathbf{M}_{\theta\theta}$ | 12×12 | 各关节的有效惯量及关节间耦合 |

代码调用：

```python
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)  # 内部执行 CRBA 算法
```

---

## 2. CRBA 算法三步走

### Step 1：构建每个连杆的空间惯性矩阵（6×6）

将每个刚体的质量、质心位置、转动惯量打包成一个统一的 6×6 矩阵：

$$\boxed{I_k^{\text{spatial}} = \begin{bmatrix} m_k \cdot \mathbb{I}_3 & -m_k[\mathbf{c}_k]_\times \\[4pt] m_k[\mathbf{c}_k]_\times & \mathbf{I}_k - m_k[\mathbf{c}_k]_\times^2 \end{bmatrix}}$$

其中 $[\mathbf{c}_k]_\times$ 是质心位置的反对称叉乘矩阵：

$$[\mathbf{c}]_\times = \begin{bmatrix} 0 & -c_z & c_y \\ c_z & 0 & -c_x \\ -c_y & c_x & 0 \end{bmatrix}$$

**物理解释**：上半部分（3 行）描述"力 → 线加速度"和"角加速度 → 线加速度"的关系；下半部分（3 行）描述"力 → 角加速度"和"扭矩 → 角加速度"的关系。

**举例**：对于 torso（$m=6.69$ kg），$I_{\text{torso}}^{\text{spatial}}$ 是一个 6×6 矩阵，概括了躯干的全部惯性特性——推它、转它需要多大的力/力矩，以及平动和转动之间如何交叉耦合。

---

### Step 2：反向累积（Backward Pass）— 构建复合刚体惯性

从运动学树的**叶节点向根节点**逐级汇总：

$$\boxed{I_i^{\text{comp}} = I_i^{\text{spatial}} + \sum_{j \in \text{child}(i)} (\mathbf{X}_j^i)^\top \cdot I_j^{\text{comp}} \cdot \mathbf{X}_j^i}$$

其中 $\mathbf{X}_j^i$ 是 6×6 **空间变换矩阵**，将关节 $j$ 坐标系下的力/速度变换到关节 $i$ 坐标系：

$$\mathbf{X}_j^i = \begin{bmatrix} \mathbf{R}_j^i & \mathbf{0} \\[4pt] -[\mathbf{p}_{j \to i}]_\times \mathbf{R}_j^i & \mathbf{R}_j^i \end{bmatrix}$$

- $\mathbf{R}_j^i$：3×3 旋转矩阵（坐标系 $j \to i$）
- $\mathbf{p}_{j \to i}$：从 $j$ 原点指向 $i$ 原点的平移向量
- 下半部分 $[\mathbf{p}]_\times$ 项反映了**力在不同作用点之间的等效变换**——在关节 $j$ 处施加的力，等效到关节 $i$ 处会额外产生一个力矩

**直觉**：$I_i^{\text{comp}}$ 的含义是——**假设关节 $i$ 以下的所有关节全部锁死，把整个子树当作一个刚体**，它在关节 $i$ 坐标系下的空间惯性。

对本机器人（18 DOF，深度 4）：

```
叶节点（4个膝关节）
    ↓ 累积到
HIP 关节（4个大腿关节）
    ↓ 累积到
ABAD 关节（4个髋侧摆关节）
    ↓ 累积到
基座 torso（根节点）
```

每条腿的质量分布（ABAD 0.42 kg → HIP 1.30 kg → KNEE 0.13 kg）通过反向累积逐级汇总到基座。最终 $I_{\text{torso}}^{\text{comp}}$ 包含了**整个机器人**关于基座原点的空间惯性——这正是 $\mathbf{M}_{bb}$ 块的来源。

---

### Step 3：填充矩阵元素（CRBA Fill）

对每对关节 $(i, j)$，找到它们的**最小公共祖先（LCA）**，用 LCA 处的复合惯量计算耦合：

$$\boxed{M_{ij} = \mathbf{s}_i^\top \cdot I_{\text{LCA}(i,j)}^{\text{comp}} \cdot \mathbf{s}_j}$$

其中 $\mathbf{s}_i$ 是关节 $i$ 的 **6D 运动子空间向量**：

| 关节类型 | $\mathbf{s}$ | 示例 |
|----------|-------------|------|
| 平动关节 | $[a_x, a_y, a_z, \; 0, 0, 0]^\top$ | 基座 x/y/z 平动 |
| 转动关节 | $[0, 0, 0, \; a_x, a_y, a_z]^\top$ | ABAD/HIP/KNEE（转轴方向） |

---

## 3. 完整公式

将三步合并，得到 CRBA 的完整表达式：

$$\boxed{M_{ij} = \sum_{k \in \nu(i) \cap \nu(j)} (\mathbf{X}_k^i \cdot \mathbf{s}_i)^\top \cdot I_k^{\text{spatial}} \cdot (\mathbf{X}_k^j \cdot \mathbf{s}_j)}$$

其中：
- $I_k^{\text{spatial}}$：Step 1 中连杆 $k$ 的空间惯性（6×6）
- $\nu(i)$：关节 $i$ 支撑的连杆集合（该关节在运动学树上的子树）
- $\nu(i) \cap \nu(j)$：两个关节**共同影响**的连杆集合
- $\mathbf{X}_k^i$：Step 2 中的空间变换矩阵
- $\mathbf{s}_i$：Step 3 中的运动子空间向量

---

## 4. 三种典型情况的物理解释

| 情况 | LCA | 公式简化 | 物理含义 | 数值举例 |
|------|-----|----------|----------|----------|
| **同一关节** $(i=j)$ | 关节 $i$ 本身 | $M_{ii} = \mathbf{s}_i^\top I_i^{\text{comp}} \mathbf{s}_i$ | 锁死其他所有关节，关节 $i$ 的有效惯量 | $M_{\text{KNEE,KNEE}} \approx 0.012$ kg·m² |
| **同链关节** $(i,j$ 在同一条腿上$)$ | 靠近基座的关节 | LCA 处的惯量包含两关节间所有连杆 | 串联耦合：远端关节加速通过中间连杆传递到近端 | $M_{\text{HIP,KNEE}}$ 在同一腿上显著 |
| **跨链关节** $(i,j$ 在不同腿上$)$ | 基座（torso） | LCA = torso，耦合仅通过躯干 | 跨腿耦合，仅靠躯干传递，比同腿耦合小 1-2 个数量级 | $M_{\text{FAR\_KNEE, FBL\_HIP}} \approx 0.0001$ |

---

## 5. 基座块 $\mathbf{M}_{bb}$ 的解析形式

基座块（前 6×6）有解析闭式解，可作为 CRBA 结果的验证：

$$\mathbf{M}_{bb} = \begin{bmatrix}
m & 0 & 0 & 0 & m\bar{z}_c & -m\bar{y}_c \\
0 & m & 0 & -m\bar{z}_c & 0 & m\bar{x}_c \\
0 & 0 & m & m\bar{y}_c & -m\bar{x}_c & 0 \\
0 & -m\bar{z}_c & m\bar{y}_c & \bar{I}_{xx} & \bar{I}_{xy} & \bar{I}_{xz} \\
m\bar{z}_c & 0 & -m\bar{x}_c & \bar{I}_{yx} & \bar{I}_{yy} & \bar{I}_{yz} \\
-m\bar{y}_c & m\bar{x}_c & 0 & \bar{I}_{zx} & \bar{I}_{zy} & \bar{I}_{zz}
\end{bmatrix}$$

其中：
- $m = \sum_k m_k$：系统总质量（$= 14.97$ kg）
- $\bar{\mathbf{c}} = [\bar{x}_c, \bar{y}_c, \bar{z}_c]^\top$：系统质心在基座坐标系下的位置
- $\bar{\mathbf{I}}$：系统关于基座原点的总转动惯量（含 Steiner 平行轴定理贡献）

**关键验证**：$M[0,0] = M[1,1] = M[2,2] = m = 14.97$ kg，等于总质量而非仅基座质量。

---

## 6. 验证方法

代码中对 $\mathbf{M}$ 做了三个物理正确性检查：

| 检查 | 方法 | 物理依据 | 结果 |
|------|------|----------|------|
| 对称性 | $\max\vert\mathbf{M} - \mathbf{M}^\top\vert < 10^{-10}$ | 动能是标量，$M_{ij} = \frac{\partial^2 K}{\partial \dot{q}_i \partial \dot{q}_j}$ | ✅ |
| 正定性 | $\min(\lambda_i) > 0$ | $K \geq 0$，且仅当 $\dot{\mathbf{q}} = 0$ 时 $K = 0$ | ✅ |
| 平动惯量 | $M[0,0] \approx \sum m_k$ | 基座纯平移时所有 body 同速，惯性 = 总质量 | ✅ |

---

## 7. 算法复杂度

CRBA 的复杂度为 $O(n \cdot d^2)$：
- $n = 18$（速度自由度）
- $d = 4$（运动学树深度：torso → ABAD → HIP → KNEE）

对于含浮动基座的一般树结构，CRBA 比 $O(n^3)$ 的暴力求逆高效得多。MuJoCo 内部对算法做了进一步优化（利用稀疏性和缓存空间变换矩阵），实际计算开销极小。

---

## 参考资料

- Featherstone, R. (2008). *Rigid Body Dynamics Algorithms*. Springer.
- MuJoCo 文档：[Computation of the inertia matrix](https://mujoco.readthedocs.io/)
- 本项目的质量矩阵分块详解：[mass_matrix_M_explanation.md](mass_matrix_M_explanation.md)
