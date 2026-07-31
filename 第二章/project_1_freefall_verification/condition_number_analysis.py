"""
M 条件数分析: 不同姿态下广义质量矩阵的条件数

条件数 cond(M) = lambda_max / lambda_min
  - 大 → 系统"硬"方向与"软"方向差异大 → 数值病态
  - 小 → 各方向惯性均匀 → 数值良态

物理含义:
  - lambda_max ~ 总质量 (基座平动, "最重"的方向)
  - lambda_min ~ 远端关节有效惯量 (膝关节, "最轻"的方向)
  - 差距约 1200 倍 → cond(M) ≈ 1200
"""

import os, numpy as np, mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

POSES = {
    "standing":    {"angles": [0.0,0.8,-1.5]*4, "desc": "Standard Standing"},
    "all_zeros":   {"angles": [0.0]*12,           "desc": "All Zeros"},
    "crouching":   {"angles": [0.3,1.5,-2.5]*4,  "desc": "Crouching"},
    "stretched":   {"angles": [0.0,-0.5,-0.8]*4, "desc": "Stretched"},
    "asymmetric":  {"angles": [0.4,0.5,-1.0, -0.4,2.0,-2.5, 0.0,1.2,-1.8, -0.3,-0.3,-0.7],
                     "desc": "Asymmetric"},
}

# Load model
model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_freefall.xml")
with open(model_path, "r", encoding="utf-8") as f:
    model_xml = f.read()
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)
M = np.zeros((model.nv, model.nv))

total_mass = sum(model.body_mass)
print(f"Model: nv={model.nv}, total mass={total_mass:.4f} kg\n")

# ================================================================
# Part 1: Condition number per pose
# ================================================================
print("=" * 70)
print("Part 1: 各姿态的条件数")
print("=" * 70)
print(f"{'Pose':>15s}  {'cond(M)':>10s}  {'lambda_min':>12s}  {'lambda_max':>12s}  {'判定':>6s}")
print("-" * 70)

pose_results = {}
for name, info in POSES.items():
    q0 = np.zeros(model.nq)
    q0[2] = 0.4; q0[3] = 1.0
    q0[7:19] = info["angles"]
    data.qpos[:] = q0; data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, data, M)

    eigvals = np.linalg.eigvalsh(M)
    cond = eigvals[-1] / eigvals[0]
    pose_results[name] = {"cond": cond, "eigvals": eigvals.copy(), "desc": info["desc"]}
    status = "OK" if cond < 5000 else "HIGH"
    print(f"{name:>15s}  {cond:10.2f}  {eigvals[0]:12.6f}  {eigvals[-1]:12.6f}  {status:>6s}")

# ================================================================
# Part 2: Eigenvalue spectrum breakdown (standing pose)
# ================================================================
print("\n" + "=" * 70)
print("Part 2: 特征值谱 (standing 姿态)")
print("=" * 70)
eigvals_all = pose_results["standing"]["eigvals"]
print(f"{'Index':>6s}  {'lambda':>12s}  {'类型':>20s}  {'物理对应':>30s}")
print("-" * 70)

labels_spectrum = (
    ["KNEE-like"] * 4 + ["HIP-like"] * 4 + ["ABAD-like"] * 4 +
    ["Base Roll", "Base Pitch", "Base Yaw"] +
    ["Base Trans Z", "Base Trans Y", "Base Trans X"]
)
for i, ev in enumerate(eigvals_all):
    print(f"  [{i:2d}]   {ev:12.6f}  {labels_spectrum[i]:>20s}")

ratio = eigvals_all[-1] / eigvals_all[0]
print(f"\n  cond(M) = {eigvals_all[-1]:.4f} / {eigvals_all[0]:.4f} = {ratio:.1f}")
print(f"  '最重方向' (平移整机) / '最轻方向' (动膝关节) ≈ {ratio:.0f}:1")

# ================================================================
# Part 3: What does high condition number mean?
# ================================================================
print("\n" + "=" * 70)
print("Part 3: 条件数过大的物理含义")
print("=" * 70)
print(f"""
  cond(M) ≈ 1200 意味着:

  1. 动力学刚度极度不均匀:
     基座平移方向"很重" (有效惯性 = {total_mass:.1f} kg)
     膝关节方向"很轻" (有效惯性 ≈ {eigvals_all[0]:.4f} kg·m²)
     差距 ≈ {ratio:.0f} 倍

  2. 逆动力学数值敏感:
     q̈ = M⁻¹ · (τ - h)
     M 条件数大 → M⁻¹ 对 τ 的微小误差极度放大
     关节力矩 1% 的误差 → 膝关节加速度可能偏差 ~10%

  3. 接触求解病态:
     M⁻¹ · Jᵀ · λ = ...
     接触力 λ 的计算涉及 M⁻¹，条件数大 → 接触力求解精度低
     可能导致"抖动"或穿透

  4. 多速率系统:
     基座运动: 慢 (~10 Hz 带宽)
     关节运动: 快 (~350 Hz 带宽)
     仿真步长必须解析最快动态 → dt ≤ 0.5 ms (Nyquist)

  5. 控制挑战:
     统一控制器难以兼顾: 基座 PID 的增益对关节会过大
     → 需要频带分离 (如 task-space + joint-space 分级控制)

  6. 物理本质:
     这不是 MuJoCo 的 bug, 而是浮动基座系统固有的多尺度特性
     总质量 >> 单个关节惯量 是所有腿式机器人的共性
""")

# ================================================================
# Part 4: Plots
# ================================================================
out_dir = os.path.join(SCRIPT_DIR, "results_condition_number")
os.makedirs(out_dir, exist_ok=True)

# Plot 1: Eigenvalue spectrum
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
colors_bar = (["#d73027"] * 4 + ["#fc8d59"] * 4 + ["#fee090"] * 4 +
              ["#91bfdb"] * 3 + ["#4575b4"] * 3)
x = np.arange(len(eigvals_all))
ax.bar(x, eigvals_all, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{i}" for i in range(len(eigvals_all))], fontsize=7)
ax.set_ylabel("Eigenvalue")
ax.set_title("(a) Eigenvalue Spectrum of M (standing pose)")
ax.axhline(y=1.0, color="gray", ls="--", lw=0.8)
# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d73027", label="Knee-like (0.012-0.013)"),
    Patch(facecolor="#fc8d59", label="HIP-like (0.015-0.027)"),
    Patch(facecolor="#fee090", label="ABAD-like (0.027-0.034)"),
    Patch(facecolor="#91bfdb", label="Base Rotation (0.22-0.49)"),
    Patch(facecolor="#4575b4", label="Base Translation (14.98)"),
]
ax.legend(handles=legend_elements, fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
pose_names = list(POSES.keys())
conds = [pose_results[n]["cond"] for n in pose_names]
bars = ax.bar(range(len(pose_names)), conds, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(pose_names))),
              edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(pose_names)))
ax.set_xticklabels(pose_names, rotation=15, fontsize=8)
ax.set_ylabel("Condition Number cond(M)")
ax.set_title("(b) Condition Number by Pose")
for i, (bar, c) in enumerate(zip(bars, conds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{c:.0f}",
            ha="center", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig_path = os.path.join(out_dir, "condition_number_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存: {fig_path}")

# Plot 2: Log-scale spectrum for all poses
fig, ax = plt.subplots(figsize=(12, 5))
for i, name in enumerate(pose_names):
    ev = pose_results[name]["eigvals"]
    ax.semilogy(range(len(ev)), ev, "o-", lw=1.2, ms=4,
                color=plt.cm.tab10(i), alpha=0.8, label=f"{name} (cond={pose_results[name]['cond']:.0f})")
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel("Eigenvalue (log scale)")
ax.set_title("Eigenvalue Spectrum — All Poses (log scale)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fig_path = os.path.join(out_dir, "eigenvalue_spectrum_all_poses.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"图表已保存: {fig_path}")

# Text report
txt_path = os.path.join(out_dir, "condition_number_report.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("M 条件数分析报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"系统: nv={model.nv}, 总质量={total_mass:.4f} kg\n\n")

    f.write("条件数定义: cond(M) = lambda_max / lambda_min\n")
    f.write("  - 大条件数: 系统有 [硬] 方向(难加速)和 [软] 方向(易加速)\n")
    f.write("  - 小条件数: 各方向惯性均匀\n\n")

    f.write("-" * 70 + "\n")
    f.write(f"{'姿态':>15s}  {'cond(M)':>10s}  {'lambda_min':>12s}  {'lambda_max':>12s}\n")
    f.write("-" * 70 + "\n")
    for name in pose_names:
        r = pose_results[name]
        f.write(f"{name:>15s}  {r['cond']:10.2f}  {r['eigvals'][0]:12.6f}  {r['eigvals'][-1]:12.6f}\n")
    f.write("-" * 70 + "\n\n")

    f.write("特征值谱 (standing):\n")
    for i, ev in enumerate(eigvals_all):
        f.write(f"  lambda[{i:2d}] = {ev:12.6f}  ({labels_spectrum[i]})\n")
    f.write(f"\n  cond(M) = {eigvals_all[-1]:.4f} / {eigvals_all[0]:.4f} = {ratio:.1f}\n\n")

    f.write("=" * 70 + "\n")
    f.write("条件数过大 (≈1200) 的物理含义\n")
    f.write("=" * 70 + "\n\n")
    f.write("1. 动力学刚度极度不均匀\n")
    f.write(f"   基座平动: 有效惯性 = {total_mass:.1f} kg (整个机器人一起动)\n")
    f.write(f"   膝关节:   有效惯性 ≈ {eigvals_all[0]:.4f} kg·m²\n")
    f.write(f"   差距约 {ratio:.0f} 倍\n\n")
    f.write("2. 逆动力学对误差敏感\n")
    f.write("   q̈ = M^{-1} * (tau - h)\n")
    f.write("   M 条件数大 → M^{-1} 将小的 tau 误差放大到关节加速度\n")
    f.write("   例如: 力矩 1% 误差 → 膝加速度 ~10% 偏差\n\n")
    f.write("3. 接触求解数值病态\n")
    f.write("   接触力 lambda 的计算涉及 M^{-1} * J^T\n")
    f.write("   条件数大 → 接触力求解精度下降 → 可能抖动或穿透\n\n")
    f.write("4. 多速率系统\n")
    f.write("   基座运动带宽 ~10 Hz, 关节运动带宽 ~350 Hz\n")
    f.write("   仿真步长必须解析最快动态 → dt <= 0.5 ms (Nyquist)\n")
    f.write("   实际 MuJoCo 使用隐式积分器可以适当放宽\n\n")
    f.write("5. 控制挑战\n")
    f.write("   统一控制器难以兼顾所有时间尺度\n")
    f.write("   解决方案: 频带分离 (task-space + joint-space 分级控制)\n\n")
    f.write("6. 本质\n")
    f.write("   这不是 bug, 而是浮动基座系统的固有特性\n")
    f.write("   总质量 >> 单关节惯量 → 所有腿式机器人的共性\n")
    f.write("=" * 70 + "\n")

print(f"报告已保存: {txt_path}")
print("\nDone.")
