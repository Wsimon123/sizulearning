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
    W = build_regressor_matrix(dq_joint, ddv_joint, level)

    phi_dim = W.shape[1]
    WtW = W.T @ W
    WtY = W.T @ Y
    phi_hat = np.linalg.solve(WtW + lam * np.eye(phi_dim), WtY)
    phi_hat = np.maximum(phi_hat, 0.0)

    cond = np.linalg.cond(WtW)
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
    """
    import jax
    import jax.numpy as jnp
    import optax

    dq_jax = jnp.array(dq_joint, dtype=jnp.float32)
    ddv_jax = jnp.array(ddv_joint, dtype=jnp.float32)
    target_jax = jnp.array(friction_target, dtype=jnp.float32)

    if level == "basic":
        phi_dim = N_ACTUATOR
    elif level == "standard":
        phi_dim = 2 * N_ACTUATOR
    else:
        phi_dim = 3 * N_ACTUATOR

    def friction_model(phi, dq, ddv):
        """
        可微摩擦模型。

        换模型时只改这个函数，梯度由 jax.grad 自动计算。
        经典方法则需要重推整个 W 矩阵。
        """
        n_a = N_ACTUATOR
        if level == "basic":
            b = phi[:n_a]
            return b * dq
        elif level == "standard":
            b, fc = phi[:n_a], phi[n_a:2*n_a]
            return b * dq + fc * jnp.tanh(K_TANH * dq)
        else:
            b, fc, ir = phi[:n_a], phi[n_a:2*n_a], phi[2*n_a:3*n_a]
            return b * dq + fc * jnp.tanh(K_TANH * dq) + ir * ddv

    def loss_fn(phi):
        pred = jax.vmap(lambda dq, ddv: friction_model(phi, dq, ddv))(dq_jax, ddv_jax)
        return jnp.mean((pred - target_jax) ** 2)

    # jax.grad: 一行代码，自动微分穿透 friction_model 内部所有运算
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    # 初始化 φ
    if level == "basic":
        phi = jnp.ones(phi_dim, dtype=jnp.float32) * 0.1
    elif level == "standard":
        phi = jnp.concatenate([
            jnp.ones(N_ACTUATOR) * 0.1,   # b 初始猜测
            jnp.ones(N_ACTUATOR) * 0.5,    # f_c 初始猜测
        ])
    else:
        phi = jnp.concatenate([
            jnp.ones(N_ACTUATOR) * 0.1,
            jnp.ones(N_ACTUATOR) * 0.5,
            jnp.ones(N_ACTUATOR) * 0.005,
        ])

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(phi)

    loss_history = []
    grad_norm_history = []
    phi_history = []

    print(f"\n  开始可微辨识 (level={level}, dim={phi_dim}, steps={n_steps})...")
    print(f"  {'Step':>6}  {'Loss':>12}  {'|grad|':>12}  {'Time':>8}")
    print(f"  {'-'*44}")

    t_start = time.time()
    for step in range(n_steps):
        g = grad_fn(phi)
        loss_val = loss_jit(phi)

        updates, opt_state = opt.update(g, opt_state)
        phi = optax.apply_updates(phi, updates)
        phi = jnp.maximum(phi, 0.0)  # 非负投影

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
    """生成三张核心图表。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Project 2: XG Friction Identification [{level}]",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.semilogy(loss_history, 'b-', lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("(a) Loss Curve")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(grad_norm_history, 'r-', lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||")
    ax.set_title("(b) Gradient Norm")
    ax.grid(True, alpha=0.3)

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
    """打印辨识结果对比。"""
    err = np.abs(phi_hat - phi_star)
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

    data = np.load(data_path, allow_pickle=True)
    dq_train = data["dq_joint_train"]
    ddv_train = data["ddv_joint_train"]
    fric_train = data["friction_target_train"]
    phi_star = data["phi_star"]

    dq_val = data["dq_joint_val"]
    ddv_val = data["ddv_joint_val"]
    fric_val = data["friction_target_val"]

    print(f"  train samples: {dq_train.shape[0]}")
    print(f"  val   samples: {dq_val.shape[0]}")
    print(f"  phi* dim: {len(phi_star)}")
    print(f"  phi* (first 8): {phi_star[:8]}")

    phi_hat_ls = None
    phi_hat_diff = None
    loss_history = []
    grad_norm_history = []

    # ----- Least Squares -----
    if not args.skip_ls:
        print(f"\n{'='*60}")
        print("Method 1: Least Squares")
        print(f"{'='*60}")

        t0 = time.time()
        phi_hat_ls, cond, residual = least_squares_identify(
            dq_train, ddv_train, fric_train, level=args.level
        )
        t_ls = time.time() - t0

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

        phi_hat_diff, loss_history, grad_norm_history, _ = differentiable_identify(
            dq_train, ddv_train, fric_train,
            level=args.level, lr=args.lr, n_steps=args.steps
        )
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
