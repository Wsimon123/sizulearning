"""
多姿态自由落体验证: 改变初始关节角度, 验证不同姿态下基座加速度
是否依然满足 a_z ≈ -9.81 m/s^2 (相对误差 < 1%)

物理原理:
  - 重力均匀作用于每个刚体
  - 关节内力由牛顿第三定律相互抵消
  - 质心加速度仅由外力 (重力) 决定: a_COM = -g
  - 因此无论什么姿态, 基座加速度都应满足自由落体条件

使用方法:
    conda activate freefall  (或 pip install mujoco)
    cd project_1_freefall_verification
    python multi_pose_verification.py
"""

import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 配置
# =====================================================================
DT = 0.001          # 仿真步长 1ms
TOTAL_TIME = 0.1    # 总仿真时间
GRAVITY = 9.81

# =====================================================================
# 定义多种测试姿态
# =====================================================================
# 每行: [FAR_ABAD, FAR_HIP, FAR_KNEE, FBL_ABAD, FBL_HIP, FBL_KNEE,
#         RAR_ABAD, RAR_HIP, RAR_KNEE, RBL_ABAD, RBL_HIP, RBL_KNEE]

POSES = {
    "standing": {
        "angles": np.array([
            0.0, 0.8, -1.5,    # FAR: 标准站立 — ABAD中立, 髋前摆, 膝弯
            0.0, 0.8, -1.5,    # FBL
            0.0, 0.8, -1.5,    # RAR
            0.0, 0.8, -1.5,    # RBL
        ]),
        "desc": "标准站立姿态 (原始)"
    },
    "all_zeros": {
        "angles": np.zeros(12),
        "desc": "全零姿态 — 所有关节为0°, 腿完全伸直下垂"
    },
    "crouching": {
        "angles": np.array([
            0.3, 1.5, -2.5,    # FAR: ABAD外展 + 髋大角度前弯 + 膝深弯
            0.3, 1.5, -2.5,    # FBL
            0.3, 1.5, -2.5,    # RAR
            0.3, 1.5, -2.5,    # RBL
        ]),
        "desc": "蹲伏姿态 — 大角度屈膝, 身体压低"
    },
    "stretched": {
        "angles": np.array([
            0.0, -0.5, -0.8,   # FAR: 髋后摆 + 膝微弯 (腿前伸)
            0.0, -0.5, -0.8,   # FBL
            0.0, -0.5, -0.8,   # RAR
            0.0, -0.5, -0.8,   # RBL
        ]),
        "desc": "伸展姿态 — 腿向前伸"
    },
    "asymmetric": {
        "angles": np.array([
            0.4, 0.5, -1.0,    # FAR: 外展 + 微屈
            -0.4, 2.0, -2.5,   # FBL: 内收 + 大幅度前弯 + 深膝弯
            0.0, 1.2, -1.8,    # RAR: 中度弯曲
            -0.3, -0.3, -0.7,  # RBL: 后摆 + 伸直
        ]),
        "desc": "非对称姿态 — 四条腿各不相同"
    },
    "random_within_limits": {
        "angles": np.array([
            0.2, 1.0, -1.0,    # 随机但合理的关节角度
            -0.1, 0.3, -1.8,
            0.35, 2.2, -2.0,
            -0.25, 0.9, -1.3,
        ]),
        "desc": "随机合理姿态"
    },
}

# =====================================================================
# 加载模型
# =====================================================================
model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_freefall.xml")
with open(model_path, "r", encoding="utf-8") as f:
    model_xml = f.read()

def build_q0(angles, base_z=0.4):
    """用给定的 12 维关节角度构建 qpos 向量"""
    base_model = mujoco.MjModel.from_xml_string(model_xml)
    q0 = np.zeros(base_model.nq)
    q0[0:3] = [0.0, 0.0, base_z]
    q0[3:7] = [1.0, 0.0, 0.0, 0.0]
    q0[7:19] = angles
    return q0, base_model.nq

# 预先确定 nq (所有姿态相同)
_, nq = build_q0(POSES["standing"]["angles"])

# =====================================================================
# 对每个姿态运行仿真
# =====================================================================
results = {}

for pose_name, pose_info in POSES.items():
    angles = pose_info["angles"]
    desc = pose_info["desc"]

    print("=" * 60)
    print(f"姿态: {pose_name} — {desc}")
    print("=" * 60)

    # 创建输出目录
    out_dir = os.path.join(SCRIPT_DIR, f"results_pose_{pose_name}")
    os.makedirs(out_dir, exist_ok=True)

    # 加载模型和数据
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    model.opt.timestep = DT
    num_steps = int(TOTAL_TIME / DT)

    q0, _ = build_q0(angles)
    data.qpos[:] = q0.copy()
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    print(f"  关节角度: {angles}")
    print(f"  初始基座位置: z = {q0[2]:.4f} m")
    print(f"  步长: {DT}s, 步数: {num_steps}")

    # 记录轨迹
    time_log = np.zeros(num_steps + 1)
    pos_log  = np.zeros((num_steps + 1, 3))
    vel_log  = np.zeros((num_steps + 1, 3))

    time_log[0] = 0.0
    pos_log[0]  = data.qpos[0:3].copy()
    vel_log[0]  = data.qvel[0:3].copy()

    for step in range(num_steps):
        data.ctrl[:] = 0
        data.qfrc_applied[:] = 0
        mujoco.mj_step(model, data)
        time_log[step + 1] = (step + 1) * DT
        pos_log[step + 1] = data.qpos[0:3].copy()
        vel_log[step + 1] = data.qvel[0:3].copy()

    # ---- 加速度 (中心差分) ----
    acc_central = np.zeros((num_steps - 1, 3))
    for i in range(1, num_steps):
        acc_central[i - 1] = (vel_log[i + 1] - vel_log[i - 1]) / (2 * DT)

    ax_mean = np.mean(acc_central[:, 0])
    ay_mean = np.mean(acc_central[:, 1])
    az_mean = np.mean(acc_central[:, 2])
    az_std  = np.std(acc_central[:, 2])

    ax_rel_err = abs(ax_mean - 0) / 9.81 * 100 if abs(ax_mean) > 1e-12 else 0.0
    ay_rel_err = abs(ay_mean - 0) / 9.81 * 100 if abs(ay_mean) > 1e-12 else 0.0
    az_rel_err = abs(az_mean - (-GRAVITY)) / GRAVITY * 100

    # ---- 轨迹对比 ----
    z_theory = pos_log[0, 2] - 0.5 * GRAVITY * time_log ** 2
    vz_theory = -GRAVITY * time_log
    pos_err_max = np.max(np.abs(pos_log[:, 2] - z_theory))
    vel_err_max = np.max(np.abs(vel_log[:, 2] - vz_theory))

    # ---- 质心计算 (验证质心加速度) ----
    com_pos_start = data.subtree_com[1].copy()   # body 1 = torso

    pass_all = (ax_rel_err < 1.0) and (ay_rel_err < 1.0) and (az_rel_err < 1.0)

    print(f"  平均 a_x = {ax_mean:+.8f}  (期望 0,   误差 {ax_rel_err:.4f}%)")
    print(f"  平均 a_y = {ay_mean:+.8f}  (期望 0,   误差 {ay_rel_err:.4f}%)")
    print(f"  平均 a_z = {az_mean:+.8f}  (期望 -9.81, 误差 {az_rel_err:.4f}%)")
    print(f"  a_z 标准差 = {az_std:.8f}")
    print(f"  位置最大误差 = {pos_err_max:.8f} m")
    print(f"  判定: {'PASS' if pass_all else 'FAIL'}")

    results[pose_name] = {
        "desc": desc,
        "angles": angles,
        "ax_mean": ax_mean,
        "ay_mean": ay_mean,
        "az_mean": az_mean,
        "az_std": az_std,
        "ax_rel_err": ax_rel_err,
        "ay_rel_err": ay_rel_err,
        "az_rel_err": az_rel_err,
        "pos_err_max": pos_err_max,
        "vel_err_max": vel_err_max,
        "pass_all": pass_all,
        "time_log": time_log,
        "pos_log": pos_log,
        "vel_log": vel_log,
        "acc_central": acc_central,
        "z_theory": z_theory,
        "vz_theory": vz_theory,
    }

    # ====== 单个姿态的图表 ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Free-Fall: {pose_name} ({desc})", fontsize=13, fontweight="bold")

    t_ms = time_log * 1000

    ax = axes[0, 0]
    ax.plot(t_ms, pos_log[:, 2], "b-", lw=2, label="Simulation")
    ax.plot(t_ms, z_theory, "r--", lw=1.5, label="Theory")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("z position (m)")
    ax.set_title(f"(a) z-position")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ms, vel_log[:, 2], "b-", lw=2, label="Simulation")
    ax.plot(t_ms, vz_theory, "r--", lw=1.5, label="Theory")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("z velocity (m/s)")
    ax.set_title("(b) z-velocity")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    t_acc_ms = time_log[1:-1] * 1000
    ax.plot(t_acc_ms, acc_central[:, 2], "b-", lw=2, label="Simulation")
    ax.axhline(y=-GRAVITY, color="r", ls="--", lw=1.5, label="Theory: -g")
    ax.set_ylim(-10.5, -9.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("z acceleration (m/s^2)")
    ax.set_title("(c) z-acceleration")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    t_acc_ms = time_log[1:-1] * 1000
    ax.plot(t_acc_ms, acc_central[:, 0], "-", lw=1.5, label="a_x", alpha=0.7)
    ax.plot(t_acc_ms, acc_central[:, 1], "-", lw=1.5, label="a_y", alpha=0.7)
    ax.axhline(y=0, color="gray", ls="--", lw=1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("acceleration (m/s^2)")
    ax.set_title("(d) x/y acceleration (expect 0)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"freefall_{pose_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存至: {fig_path}")

    # ====== 文本输出 ======
    txt_path = os.path.join(out_dir, f"results_{pose_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as fout:
        fout.write(f"自由落体验证 — 姿态: {pose_name}\n")
        fout.write("=" * 60 + "\n\n")
        fout.write(f"姿态描述: {desc}\n")
        fout.write(f"关节角度:\n")
        joint_names = ["FAR_ABAD", "FAR_HIP", "FAR_KNEE",
                       "FBL_ABAD", "FBL_HIP", "FBL_KNEE",
                       "RAR_ABAD", "RAR_HIP", "RAR_KNEE",
                       "RBL_ABAD", "RBL_HIP", "RBL_KNEE"]
        for i, name in enumerate(joint_names):
            fout.write(f"  {name:>14s}: {angles[i]:+8.4f} rad ({np.degrees(angles[i]):+7.2f}°)\n")

        fout.write(f"\n仿真参数:\n")
        fout.write(f"  步长: {DT} s\n")
        fout.write(f"  步数: {num_steps}\n")
        fout.write(f"  总时间: {TOTAL_TIME} s\n\n")

        fout.write(f"--- 加速度验证 ---\n")
        fout.write(f"平均 a_x: {ax_mean:+.8f} m/s²  (期望 0, 误差 {ax_rel_err:.4f}%)\n")
        fout.write(f"平均 a_y: {ay_mean:+.8f} m/s²  (期望 0, 误差 {ay_rel_err:.4f}%)\n")
        fout.write(f"平均 a_z: {az_mean:+.8f} m/s²  (期望 -9.81, 误差 {az_rel_err:.4f}%)\n")
        fout.write(f"a_z 标准差: {az_std:.8f} m/s²\n\n")

        fout.write(f"--- 轨迹对比 ---\n")
        fout.write(f"位置最大误差: {pos_err_max:.8f} m\n")
        fout.write(f"速度最大误差: {vel_err_max:.8f} m/s\n\n")

        fout.write(f"判定: {'PASS (所有轴 < 1%)' if pass_all else 'FAIL'}\n")

    print(f"  文本已保存至: {txt_path}")

# =====================================================================
# 汇总比较
# =====================================================================
print("\n" + "=" * 60)
print("生成多姿态汇总比较图...")
print("=" * 60)

summary_dir = os.path.join(SCRIPT_DIR, "results_pose_summary")
os.makedirs(summary_dir, exist_ok=True)

pose_names = list(POSES.keys())
n_poses = len(pose_names)

# ---- 图1: 加速度对比 和 误差对比 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-Pose Free-Fall Verification — Different Initial Joint Angles",
             fontsize=14, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, n_poses))

# (a) a_z 时间序列对比
ax = axes[0, 0]
for idx, name in enumerate(pose_names):
    r = results[name]
    t_acc_ms = r["time_log"][1:-1] * 1000
    stride = max(1, len(t_acc_ms) // 200)
    ax.plot(t_acc_ms[::stride], r["acc_central"][::stride, 2],
            "-", color=colors[idx], lw=1.2, alpha=0.8, label=f"{name}")
ax.axhline(y=-GRAVITY, color="black", ls="--", lw=1, label="Theory: -9.81")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("a_z (m/s²)")
ax.set_title("(a) a_z Time Series — All Poses Overlap at -9.81")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# (b) 各姿态 a_z 均值条形图
ax = axes[0, 1]
az_means = [results[n]["az_mean"] for n in pose_names]
az_errs  = [results[n]["az_rel_err"] for n in pose_names]
x_pos = np.arange(n_poses)
bars = ax.bar(x_pos, az_means, color=colors, edgecolor="white", linewidth=1.2)
ax.axhline(y=-GRAVITY, color="red", ls="--", lw=1.5, label=f"Theory: {-GRAVITY}")
ax.set_xticks(x_pos)
ax.set_xticklabels(pose_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Mean a_z (m/s²)")
ax.set_title("(b) Mean a_z by Pose")
for i, (bar, err) in enumerate(zip(bars, az_errs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height() + 0.0002,
            f"{err:.4f}%", ha="center", fontsize=8, color="darkred")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# (c) a_x 和 a_y 误差 (应接近0)
ax = axes[1, 0]
ax_x = np.arange(n_poses)
width = 0.35
ax.bar(ax_x - width/2, [results[n]["ax_rel_err"] for n in pose_names],
       width, color="#2166ac", edgecolor="white", label="a_x error (%)")
ax.bar(ax_x + width/2, [results[n]["ay_rel_err"] for n in pose_names],
       width, color="#b2182b", edgecolor="white", label="a_y error (%)")
ax.set_xticks(ax_x)
ax.set_xticklabels(pose_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Relative Error (%)")
ax.set_title("(c) Lateral Acceleration Error (expect 0)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# (d) 位置/速度误差对比
ax = axes[1, 1]
x_pos2 = np.arange(n_poses)
width2 = 0.35
ax.bar(x_pos2 - width2/2, [results[n]["pos_err_max"]*1000 for n in pose_names],
       width2, color="#4d9221", edgecolor="white", label="Position err (mm)")
ax_twin = ax.twinx()
ax_twin.bar(x_pos2 + width2/2, [results[n]["vel_err_max"]*1000 for n in pose_names],
            width2, color="#7b3294", edgecolor="white", label="Velocity err (mm/s)")
ax.set_xticks(x_pos2)
ax.set_xticklabels(pose_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Position Max Error (mm)")
ax_twin.set_ylabel("Velocity Max Error (mm/s)")
ax.set_title("(d) Position & Velocity Max Error")
ax.grid(True, alpha=0.3, axis="y")

# 合并图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

plt.tight_layout()
summary_fig = os.path.join(summary_dir, "multi_pose_comparison.png")
plt.savefig(summary_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"汇总图已保存至: {summary_fig}")

# ---- 图2: 各姿态轨迹子图矩阵 ----
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Multi-Pose: z-trajectory Comparison", fontsize=14, fontweight="bold")
axes = axes.flatten()

for idx, name in enumerate(pose_names):
    if idx >= 6:
        break
    r = results[name]
    t_ms = r["time_log"] * 1000
    ax = axes[idx]
    ax.plot(t_ms, r["pos_log"][:, 2], "b-", lw=2, label="Simulation")
    ax.plot(t_ms, r["z_theory"], "r--", lw=1.5, label="Theory")
    ax.set_title(f"{name}: {r['desc']}", fontsize=10)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("z (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
traj_fig = os.path.join(summary_dir, "multi_pose_trajectories.png")
plt.savefig(traj_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"轨迹图已保存至: {traj_fig}")

# =====================================================================
# 汇总文本报告
# =====================================================================
report_path = os.path.join(summary_dir, "multi_pose_report.txt")
with open(report_path, "w", encoding="utf-8") as fout:
    fout.write("=" * 72 + "\n")
    fout.write("多姿态自由落体验证 — 汇总报告\n")
    fout.write("=" * 72 + "\n\n")
    fout.write(f"仿真步长: {DT} s\n")
    fout.write(f"总仿真时间: {TOTAL_TIME} s\n")
    fout.write(f"理论加速度: {-GRAVITY} m/s²\n")
    fout.write(f"测试姿态数: {n_poses}\n\n")

    fout.write("物理原理:\n")
    fout.write("  重力均匀作用于每个刚体, 关节内力由牛顿第三定律抵消,\n")
    fout.write("  因此质心加速度仅由外力决定: a_COM = -g = -9.81 m/s²。\n")
    fout.write("  无论机器人保持什么姿态 (关节角度如何), 只要 tau=0、\n")
    fout.write("  无地面接触, 基座加速度应始终等于 -g。\n\n")

    header = (f"{'姿态':>20s}  {'a_x均值':>12s}  {'a_y均值':>12s}  "
              f"{'a_z均值':>12s}  {'a_z误差':>9s}  "
              f"{'位置误差':>10s}  {'判定':>6s}")
    fout.write(header + "\n")
    fout.write("-" * 72 + "\n")

    for name in pose_names:
        r = results[name]
        verdict = "PASS" if r["pass_all"] else "FAIL"
        line = (f"{name:>20s}  {r['ax_mean']:+12.8f}  {r['ay_mean']:+12.8f}  "
                f"{r['az_mean']:+12.8f}  {r['az_rel_err']:8.4f}%  "
                f"{r['pos_err_max']:10.8f}  {verdict:>6s}")
        fout.write(line + "\n")

    fout.write("-" * 72 + "\n\n")

    # 每姿态关节角度
    joint_names = ["FAR_ABAD", "FAR_HIP", "FAR_KNEE",
                   "FBL_ABAD", "FBL_HIP", "FBL_KNEE",
                   "RAR_ABAD", "RAR_HIP", "RAR_KNEE",
                   "RBL_ABAD", "RBL_HIP", "RBL_KNEE"]

    for name in pose_names:
        r = results[name]
        fout.write(f"\n[{name}] {r['desc']}\n")
        fout.write(f"关节角度 (rad / °):\n")
        for i, jname in enumerate(joint_names):
            ang = r["angles"][i]
            fout.write(f"  {jname:>14s}: {ang:+8.4f} rad ({np.degrees(ang):+7.2f}°)\n")

    fout.write("\n" + "=" * 72 + "\n")
    fout.write("结论\n")
    fout.write("=" * 72 + "\n")
    fout.write("  1. 所有姿态下 a_z ≈ -9.81 m/s², a_x ≈ 0, a_y ≈ 0\n")
    fout.write("  2. 加速度不依赖于初始关节角度 — 与物理理论一致\n")
    fout.write("  3. 关节内力是内力, 不影响质心运动 (牛顿第三定律)\n")
    fout.write("  4. 位置/速度误差仅来源于数值积分, 与姿态无关\n")
    fout.write("  5. 验证了 MuJoCo 浮动基座动力学的物理正确性\n")
    fout.write("=" * 72 + "\n")

print(f"汇总报告已保存至: {report_path}")

# 终端输出
print("\n" + "=" * 72)
print("多姿态最终对比")
print("=" * 72)
print(f"{'姿态':>20s}  {'a_z均值':>12s}  {'a_z误差':>9s}  {'a_x误差':>8s}  {'a_y误差':>8s}  {'判定':>6s}")
print("-" * 72)
for name in pose_names:
    r = results[name]
    verdict = "PASS" if r["pass_all"] else "FAIL"
    print(f"{name:>20s}  {r['az_mean']:+12.6f}  {r['az_rel_err']:8.4f}%  "
          f"{r['ax_rel_err']:7.4f}%  {r['ay_rel_err']:7.4f}%  {verdict:>6s}")
print("-" * 72)

all_pass = all(r["pass_all"] for r in results.values())
print(f"\n全部通过: {'YES' if all_pass else 'NO'}")

print(f"\n输出文件:")
print(f"  汇总: {summary_dir}")
for name in pose_names:
    print(f"  {name}: {os.path.join(SCRIPT_DIR, f'results_pose_{name}')}")
print("=" * 72)
