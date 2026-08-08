"""
Project 2: XG 摩擦辨识 -- 可微方法 + 经典最小二乘 baseline
==============================================================

两种方法辨识同一个摩擦模型:
    tau_fric_i = b_i * dtheta_i + fc_i * tanh(100 * dtheta_i) [+ Ir_i * ddtheta_i]

区别:
  - 经典最小二乘：手推回归矩阵 W，闭式解 (W'W + λI)^{-1} W'Y
  - 可微方法：定义摩擦函数 → jax.grad 自动求梯度 → Adam 迭代优化
    换摩擦模型时，经典方法须重推 W；可微方法只改一处函数定义。

使用方法:
    python generate_synthetic_data.py            # 先生成数据
    python friction_identification.py --level basic      # 基础: b, 12维
    python friction_identification.py --level standard   # 标准: b+f_c, 24维
    python friction_identification.py --level advanced   # 进阶: b+f_c+I_r, 36维
"""

import os
import sys
import argparse
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_ACTUATOR = 12
K_TANH = 100.0


# JAX（底层计算 + 自动微分）
#  ├── MuJoCo MJX（物理仿真，并行环境）
#  └── Optax（优化器，更新策略网络参数）
# 一个典型的训练流程：用 MJX 在 GPU 上并行跑 N 个仿真环境收集数据 → 用 JAX 计算策略梯度 → 用 Optax 的 Adam 优化器更新网络权重。三者都是 JAX 生态的核心组件。

# =====================================================================
# Part A: 经典最小二乘 baseline
# =====================================================================
def build_regressor_matrix(dq_joint, ddv_joint, level="standard"):
    """
    构建回归矩阵 W，使得 tau_friction = W @ phi。

    经典方法的核心：必须针对摩擦模型的具体形式手工推导 W。
    换模型（如 LuGre / Stribeck）→ W 必须重新推导。
    """
    T = dq_joint.shape[0]

    if level == "basic":
        # tau_fric_i = b_i * dtheta_i
        W = np.zeros((T * N_ACTUATOR, N_ACTUATOR))
        for i in range(N_ACTUATOR):
            rows = np.arange(T) * N_ACTUATOR + i
            W[rows, i] = dq_joint[:, i]

    elif level == "standard":
        # tau_fric_i = b_i * dtheta_i + fc_i * tanh(k * dtheta_i)
        W = np.zeros((T * N_ACTUATOR, 2 * N_ACTUATOR))
        for i in range(N_ACTUATOR):
            rows = np.arange(T) * N_ACTUATOR + i
            W[rows, i] = dq_joint[:, i]
            W[rows, N_ACTUATOR + i] = np.tanh(K_TANH * dq_joint[:, i])

    else:  # advanced
        # tau_fric_i = b_i * dtheta_i + fc_i * tanh(k * dtheta_i) + Ir_i * ddtheta_i
        W = np.zeros((T * N_ACTUATOR, 3 * N_ACTUATOR))
        for i in range(N_ACTUATOR):
            rows = np.arange(T) * N_ACTUATOR + i
            W[rows, i] = dq_joint[:, i]
            W[rows, N_ACTUATOR + i] = np.tanh(K_TANH * dq_joint[:, i])
            W[rows, 2 * N_ACTUATOR + i] = ddv_joint[:, i]

    return W


def least_squares_identify(dq_joint, ddv_joint, friction_target, level="standard", lam=1e-4):
    """
    经典最小二乘辨识。

    Y = friction_target (从 qfrc_passive 中提取的摩擦力)
    W = 回归矩阵
    phi_hat = (W'W + λI)^{-1} W'Y    (Tikhonov 正则化)
    """
    Y = friction_target.reshape(-1)
    #所以 friction_target.reshape(-1) 用 C 顺序（默认行优先）把 (T, 12) 展平成 一维向量 (T×12,)，展平顺序恰好和 W 的行一一对应——同一时间步的 12 个关节值被依次拉平。Y 就是长度 T×12 的 float64 一维数组，和 W 的行数完全匹配。
    
    W = build_regressor_matrix(dq_joint, ddv_joint, level)#构建W矩阵

    # 参数向量维度：basic→12, standard→24, advanced→36
    phi_dim = W.shape[1]

    # 构建正规方程 A x = b，其中 A = W^T W + λI（Tikhonov正则化）
    WtW = W.T @ W                # (p, p) 对称正定矩阵（加上 λI 后）
    WtY = W.T @ Y                # (p,) 向量

    # solve 直接求解线性方程组，比 np.linalg.inv 更稳定
    phi_hat = np.linalg.solve(WtW + lam * np.eye(phi_dim), WtY)

    # 投影到非负象限：摩擦系数（粘滞、库仑、惯量比）物理上 ≥ 0
    phi_hat = np.maximum(phi_hat, 0.0)

    # 条件数：衡量解的数值稳定性，>1e8 说明数据病态
    cond = np.linalg.cond(WtW)

# 条件数告诉你这个解有多可信。

# 数学含义
# 对线性系统 A x = b，解 x 对 b 中微小误差的放大倍数由条件数决定：


# ||Δx|| / ||x||  ≤  κ(A) × ||Δb|| / ||b||
#   ↑解的相对误差        ↑条件数       ↑数据中的相对误差
# 1% 的数据噪声 × 条件数 10^8 = 解可能放大到 10^6%——完全不可靠。

# 具体到这行代码
# cond = np.linalg.cond(WtW) 算的是 W^T W 的条件数，等于最大特征值除以最小特征值：κ = λ_max / λ_min。

# 它主要检测一件事：W 的列之间是否存在近似线性相关。

# 对于摩擦辨识来说，最容易出问题的是 advanced 模型——速度和加速度高度相关（正弦轨迹下 ddθ ≈ -ω^2 θ，和 dθ ≈ ω cos(ωt) 仅差相位和因子），导致 W 中这两列近似线性依赖，某个特征值压到接近 0，条件数直接爆掉。

# 工程上怎么读
# κ 量级	判断
# < 10^3	干净、可靠
# 10^3 ~ 10^6	还行，个别参数不确定
# 10^6 ~ 10^8	告警，部分参数不可信
# > 10^8	本质奇异，解基本瞎了
# 一句话
# 条件数是把一个量纲统一的无量纲指标——你不需要知道数据的物理单位或数值大小，看一眼 κ 就知道解能不能用。high κ → 数据没提供足够独立的信息来分开估计每个参数

#     # MSE 残差：拟合值与真值的均方误差，越小越好
    residual = np.mean((W @ phi_hat - Y) ** 2)

    return phi_hat, cond, residual


# =====================================================================
# Part B: 可微方法 (JAX + Adam)
# =====================================================================
def differentiable_identify(dq_joint, ddv_joint, friction_target, level="standard",
                            lr=1e-2, n_steps=2000, seed=0):
    """
    可微辨识：定义摩擦模型为可微函数 → jax.grad 自动求梯度 → Adam 优化。

    核心优势（对比经典 LS）：
      - 不需要手推 W 矩阵
      - 换摩擦模型只改 friction_model 函数体
      - 扩展到非线性模型（LuGre 等）无额外代价

    在完整工程中，friction_model 可替换为 mjx.rne() 或 mjx.step()，
    实现"整个仿真器作为可微函数"的完整链路。

    Parameters
    ----------
    dq_joint : (T,12) 关节速度
    ddv_joint : (T,12) 关节加速度
    friction_target : (T,12) 摩擦力目标值（带噪声）
    level : 模型层次
    lr : Adam 学习率，默认 1e-2
    n_steps : 迭代步数，默认 2000
    seed : 随机种子（当前未使用，预留）

    Returns
    -------
    phi_final : (p,) 最终辨识参数
    loss_history : list[float] 每步 MSE 损失
    grad_norm_history : list[float] 每步梯度 L2 范数
    phi_history : list[ndarray] 每 100 步保存的参数快照
    """
    import jax
    import jax.numpy as jnp
    import optax

    # 数据转 JAX 数组，float32 加速 GPU/TPU 计算
    dq_jax = jnp.array(dq_joint, dtype=jnp.float32)
    ddv_jax = jnp.array(ddv_joint, dtype=jnp.float32)
    target_jax = jnp.array(friction_target, dtype=jnp.float32)

    # 参数维度: basic=12, standard=24, advanced=36
    if level == "basic":
        phi_dim = N_ACTUATOR
    elif level == "standard":
        phi_dim = 2 * N_ACTUATOR
    else:
        phi_dim = 3 * N_ACTUATOR

    def friction_model(phi, dq, ddv):
        """
        可微摩擦模型 — 前向计算：输入参数 phi，输出 12 个关节的摩擦力。

        输入 dq/ddv 都是 (12,) 向量，返回值也是 (12,) 向量。
        换模型时只改这个函数，梯度由 jax.grad 自动计算。
        经典方法则需要重推整个 W 矩阵。
        """
        n_a = N_ACTUATOR
        if level == "basic":
            # tau_i = b_i * dq_i
            b = phi[:n_a]
            return b * dq
        elif level == "standard":
            # tau_i = b_i * dq_i + fc_i * tanh(k * dq_i)
            b, fc = phi[:n_a], phi[n_a:2*n_a]
            return b * dq + fc * jnp.tanh(K_TANH * dq)
        else:
            # tau_i = b_i * dq_i + fc_i * tanh(k * dq_i) + Ir_i * ddq_i
            b, fc, ir = phi[:n_a], phi[n_a:2*n_a], phi[2*n_a:3*n_a]
            return b * dq + fc * jnp.tanh(K_TANH * dq) + ir * ddv

    def loss_fn(phi):
        """
        损失函数: MSE(模型预测值, 目标值)

        jax.vmap 将 friction_model 沿时间轴并行映射:
          (phi, dq[0], ddv[0]) → pred[0]   (12,)
          (phi, dq[1], ddv[1]) → pred[1]   (12,)
          ...
          得到 pred 形状 (T, 12)，与 target_jax (T, 12) 算 MSE
        """
        pred = jax.vmap(lambda dq, ddv: friction_model(phi, dq, ddv))(dq_jax, ddv_jax)
        #lambda dq, ddv: friction_model(phi, dq, ddv) — 定义一个匿名函数，它接收一个时间步的速度 dq 和加速度 ddv，然后用当前的参数 phi 去计算该时间步的摩擦力预测值。

#jax.vmap(...)(dq_jax, ddv_jax) — vmap 是 JAX 的向量化映射，相当于把上面那个函数沿着时间轴（第 0 维）广播。假设 dq_jax 的形状是 (T, 12)（T 个时间步，12 个关节），vmap 会自动拆成 T 次调用，每次取一个 (12,) 的向量传入，然后把 T 个结果拼回来，得到 pred 形状为 (T, 12)。

#一句话：用当前参数 phi，对全部 T 个时间步逐一计算摩擦力预测值，得到一个 (T, 12) 的预测矩阵。
        
        return jnp.mean((pred - target_jax) ** 2)
    
# 这是标准的均方误差（MSE）：

# pred - target_jax — 预测值减去真实值（真实摩擦力数据），形状 (T, 12)，得到每个时间步、每个关节的误差。
# (...) ** 2 — 每个误差取平方。
# jnp.mean(...) — 对所有元素求平均，得到一个标量。
# 一句话：计算预测值与真实值之间的均方误差，作为优化的损失函数。

# 整体逻辑：梯度下降优化器会不断调整 phi（摩擦参数），让这个 MSE 越来越小，从而找到最符合真实数据的摩擦模型参数。

    # jit 编译: 将 loss_fn 和 grad(loss_fn) 编译为 XLA 计算图，后续调用近乎零开销
    grad_fn = jax.jit(jax.grad(loss_fn))   # 梯度函数: phi → d(loss)/d(phi)
    #所以 grad_fn(phi) 直接返回长度为 24（或 36）的精确梯度向量。
    
    loss_jit = jax.jit(loss_fn)            # 编译后的 loss，用于监控不用于求导

    # 初始化 phi: 用物理合理的量级作为起点，避免从零开始（梯度为零）
    if level == "basic":
        phi = jnp.ones(phi_dim, dtype=jnp.float32) * 0.1
    elif level == "standard":
        phi = jnp.concatenate([
            jnp.ones(N_ACTUATOR) * 0.1,    # b: 粘滞系数 ~0.1 Nm/(rad/s)
            jnp.ones(N_ACTUATOR) * 0.5,    # fc: 库仑摩擦 ~0.5 Nm
        ])
    else:
        phi = jnp.concatenate([
            jnp.ones(N_ACTUATOR) * 0.1,
            jnp.ones(N_ACTUATOR) * 0.5,
            jnp.ones(N_ACTUATOR) * 0.005,  # Ir: 惯量比 ~0.005
        ])

    # Adam 优化器: 自适应学习率 + 动量，对非凸问题鲁棒
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(phi)              # 初始化 Adam 的一阶/二阶矩

    loss_history = []       # 每一步的 MSE，用于画收敛曲线
    grad_norm_history = []  # 每一步的梯度范数，用于判断优化平稳性
    phi_history = []        # 每 100 步的参数快照，用于调试

    print(f"\n  开始可微辨识 (level={level}, dim={phi_dim}, steps={n_steps})...")
    print(f"  {'Step':>6}  {'Loss':>12}  {'|grad|':>12}  {'Time':>8}")
    print(f"  {'-'*44}")

    t_start = time.time()
    for step in range(n_steps):
        g = grad_fn(phi)      # 自动微分: 计算 d(loss)/d(phi) 对各参数的梯度
        loss_val = loss_jit(phi)  # 当前损失值

        updates, opt_state = opt.update(g, opt_state)  # Adam 计算参数更新量
        phi = optax.apply_updates(phi, updates)        # 应用更新
        phi = jnp.maximum(phi, 0.0)  # 非负投影：摩擦系数物理上 ≥ 0
        
        # Adam 只需要知道"往哪个方向走、坡度多陡"（梯度），不需要知道"现在海拔多少米"（loss 值）。

        # loss_val 唯一的用途就是打印出来和画收敛曲线，让你判断"还在下降吗？是不是卡住了？"。把它删了，优化照样跑，只是你变成瞎子而已。
        
        # 第278行 — 根据梯度算出往哪调、调多大
        # 第279行 — 真正执行调整
        # 第280行 — 硬约束：摩擦系数不能为负


        # 转回 Python 标量，脱离 JAX 计算图，避免内存累积
        loss_np = float(loss_val)
        grad_norm = float(jnp.linalg.norm(g))
        loss_history.append(loss_np)
        grad_norm_history.append(grad_norm)

        if step % 100 == 0 or step == n_steps - 1:
            elapsed = time.time() - t_start
            print(f"  {step:6d}  {loss_np:12.6f}  {grad_norm:12.6f}  {elapsed:7.1f}s")
            phi_history.append(np.array(phi))

    phi_final = np.array(phi)
    return phi_final, loss_history, grad_norm_history, phi_history


# =====================================================================
# Part C: 结果可视化
# =====================================================================
def plot_results(phi_hat_diff, phi_hat_ls, phi_star, level,
                 loss_history, grad_norm_history, save_dir):
    """
    生成三张子图并排的结果图，保存为 PNG。

    子图内容:
      (a) Loss 曲线 — 半对数坐标，看收敛速度和最终残差
      (b) 梯度范数曲线 — 半对数坐标，看优化是否平稳，有无震荡
      (c) 参数柱状图 — 真值 phi_star vs 可微方法 phi_diff vs LS方法 phi_LS
          三组柱并排，直观对比每个参数的辨识精度
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Project 2: XG Friction Identification [{level}]",
                 fontsize=14, fontweight="bold")

    # (a) Loss 曲线 — 半对数 y 轴，便于分辨从大到小的下降过程
    ax = axes[0]
    ax.semilogy(loss_history, 'b-', lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("(a) Loss Curve")
    ax.grid(True, alpha=0.3)

    # (b) 梯度范数 — 半对数 y 轴，下降到接近 0 说明收敛
    ax = axes[1]
    ax.semilogy(grad_norm_history, 'r-', lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||")
    ax.set_title("(b) Gradient Norm")
    ax.grid(True, alpha=0.3)

    # (c) 参数对比柱状图 — x 轴是参数索引 (0~35)，三组柱挨着排列
    ax = axes[2]
    dim = len(phi_star)
    x = np.arange(dim)
    width = 0.25
    ax.bar(x - width, phi_star, width, label="phi* (true)", color='#1C4E8F', alpha=0.8)
    ax.bar(x, phi_hat_diff, width, label="phi_diff", color='#F59E0B', alpha=0.8)
    if phi_hat_ls is not None:
        ax.bar(x + width, phi_hat_ls, width, label="phi_LS", color='#059669', alpha=0.8)
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Value")
    ax.set_title("(c) phi_hat vs phi*")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # x 轴标签: 参数 ≤ 36 时用具体名称 (b0, fc0, Ir0...) 替代纯数字索引
    if level == "basic":
        labels = [f"b{i}" for i in range(12)]
    elif level == "standard":
        labels = [f"b{i}" for i in range(12)] + [f"fc{i}" for i in range(12)]
    else:
        labels = ([f"b{i}" for i in range(12)] +
                  [f"fc{i}" for i in range(12)] +
                  [f"Ir{i}" for i in range(12)])
    if dim <= 36:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=5)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"results_{level}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  图表已保存至: {out_path}")
    plt.close()


def print_comparison(phi_hat, phi_star, method_name):
    """
    逐参数打印辨识值 vs 真值的对比表。

    包含:
      - 绝对误差 |err| = |phi_hat - phi_star|
      - 相对误差 rel%  = |err| / (|phi_star| + eps) * 100%
      - 总相对误差 = ||phi_hat - phi_star|| / ||phi_star|| * 100%
                    (L2 范数之比，衡量整体辨识质量)
    """
    err = np.abs(phi_hat - phi_star)
    # +1e-10 防除零: 真值恰好为 0 时相对无意义
    rel_err = err / (np.abs(phi_star) + 1e-10)
    total_rel = np.linalg.norm(phi_hat - phi_star) / (np.linalg.norm(phi_star) + 1e-10)

    print(f"\n  --- {method_name} ---")
    print(f"  {'Idx':>4}  {'phi*':>10}  {'phi_hat':>10}  {'|err|':>10}  {'rel%':>8}")
    for i in range(len(phi_star)):
        print(f"  {i:4d}  {phi_star[i]:10.5f}  {phi_hat[i]:10.5f}  "
              f"{err[i]:10.5f}  {rel_err[i]*100:7.2f}%")
    print(f"  total relative error: {total_rel*100:.2f}%")
    return total_rel


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Project 2: XG friction identification")
    parser.add_argument("--level", type=str, default="standard",
                        choices=["basic", "standard", "advanced"])
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--steps", type=int, default=2000, help="optimization steps")
    parser.add_argument("--skip_ls", action="store_true", help="skip least squares baseline")
    parser.add_argument("--skip_diff", action="store_true", help="skip differentiable method")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Project 2: XG Friction Identification [{args.level}]")
    print("=" * 60)

    data_path = os.path.join(SCRIPT_DIR, f"data_{args.level}.npz")
    if not os.path.exists(data_path):
        print(f"[ERROR] data file not found: {data_path}")
        print("Run first: python generate_synthetic_data.py")
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)#加载数据
    dq_train = data["dq_joint_train"]#关节位置(测量得到：带噪声)
    ddv_train = data["ddv_joint_train"]#关节加速度(测量得到：带噪声)
    fric_train = data["friction_target_train"]#关节摩擦力(参数计算得到，可以理解为带噪声的真值)
    phi_star = data["phi_star"]
    # 真实摩擦参数，数据生成时用的"标准答案"（basic 模式下是 12 维的 B*，advanced 模式下是 36 维的 [B*, FC*, IR*] 拼接），用于评估辨识精度
    
    
    #ti 和 vi 分别是训练集索引和验证集索引。

    dq_val = data["dq_joint_val"]#验证集位置
    ddv_val = data["ddv_joint_val"]#验证集加速度
    fric_val = data["friction_target_val"]#验证集摩擦力

    print(f"  train samples: {dq_train.shape[0]}")
    print(f"  val   samples: {dq_val.shape[0]}")
    print(f"  phi* dim: {len(phi_star)}")
    print(f"  phi* (first 8): {phi_star[:8]}")

    phi_hat_ls = None
    #最小二乘法辨识结果，估计出的摩擦参数。形状与 phi_star 相同（basic 12 维，advanced 36 维）。如果 --skip_ls 则保持 None
    
    phi_hat_diff = None
    # JAX 梯度下降法辨识结果，同样是摩擦参数估计值。如果 --skip_diff 则保持 None
    
    loss_history = []
    #梯度下降的损失曲线，每一步存一个 MSE 值。用于画收敛图
    
    grad_norm_history = []
    #梯度范数曲线，每一步存一个梯度 L2 范数。用于画梯度变化图，观察优化是否平稳

    # ----- Least Squares -----
    if not args.skip_ls:
        print(f"\n{'='*60}")
        print("Method 1: Least Squares")
        print(f"{'='*60}")

        t0 = time.time()
        phi_hat_ls, cond, residual = least_squares_identify(#训练步
            dq_train, ddv_train, fric_train, level=args.level
        )
        t_ls = time.time() - t0

        # 1. 训练（拟合）

        # phi_hat_ls, cond, residual = least_squares_identify(
        #     dq_train, ddv_train, fric_train, level=args.level
        # )
        # 调用 least_squares_identify 函数，用训练数据解：


        # W_train * phi = fric_train  →  phi_hat = (W^T W)^(-1) W^T * fric_train
        # 其中 W_train 是回归矩阵（由 dq 和 ddv 按照摩擦模型构造），fric_train 是目标摩擦力。返回三个值：


        print(f"  cond(W'W): {cond:.2e}")
        print(f"  train MSE: {residual:.6f}")
        print(f"  time: {t_ls:.2f}s")
        print_comparison(phi_hat_ls, phi_star, "Least Squares")

        # validation
        W_val = build_regressor_matrix(dq_val, ddv_val, args.level)
        val_mse = np.mean((W_val @ phi_hat_ls - fric_val.reshape(-1)) ** 2)
        print(f"  val   MSE: {val_mse:.6f}")

    # ----- Differentiable -----
    if not args.skip_diff:
        print(f"\n{'='*60}")
        print("Method 2: Differentiable (JAX + Adam)")
        print(f"{'='*60}")

        # 可微辨识：用 JAX 自动微分 + Adam 优化器拟合摩擦参数
        # 与 LS 方法对比的核心区别：
        #   - 不需要手推 W 矩阵，只定义 forward 模型，梯度由 jax.grad 自动计算
        #   - 通过迭代优化 min MSE(friction_model(phi, dq, ddv) - fric_train)
        # 返回值：
        #   phi_hat_diff  : 辨识出的摩擦参数 (p,)
        #   loss_history   : 每一步的 loss 值列表，用于画收敛曲线
        #   grad_norm_history: 每一步的梯度范数列表，用于观察优化平稳性
        #   _              : phi_history，未使用
        phi_hat_diff, loss_history, grad_norm_history, _ = differentiable_identify(
            dq_train, ddv_train, fric_train,
            level=args.level, lr=args.lr, n_steps=args.steps
        )
        # 逐参数打印辨识值 vs 真值及相对误差
        print_comparison(phi_hat_diff, phi_star, "Differentiable")

    # ----- Plots -----
    if loss_history:
        if phi_hat_diff is None:
            phi_hat_diff = np.zeros_like(phi_star)
        plot_results(phi_hat_diff, phi_hat_ls, phi_star, args.level,
                     loss_history, grad_norm_history, SCRIPT_DIR)

    # ----- Summary -----
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    thresholds = {"basic": 10.0, "standard": 5.0, "advanced": 10.0}
    threshold = thresholds[args.level]

    if phi_hat_ls is not None:
        rel = np.linalg.norm(phi_hat_ls - phi_star) / np.linalg.norm(phi_star) * 100
        print(f"  LS   relative error: {rel:.2f}%  {'PASS' if rel < threshold else 'FAIL'} (threshold {threshold}%)")
    if phi_hat_diff is not None:
        rel = np.linalg.norm(phi_hat_diff - phi_star) / np.linalg.norm(phi_star) * 100
        print(f"  Diff relative error: {rel:.2f}%  {'PASS' if rel < threshold else 'FAIL'} (threshold {threshold}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
