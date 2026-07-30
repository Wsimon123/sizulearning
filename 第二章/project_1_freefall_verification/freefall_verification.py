"""
Project 1: 自由落体验证 -- 浮动基座动力学的物理正确性验证

验证在 tau=0、无地面约束条件下，四足机器人基座加速度是否满足:
    a_z 约等于 -9.81 m/s^2  (相对误差 < 1%)

使用方法:
    conda activate freefall
    cd project_1_freefall_verification
    python freefall_verification.py
"""

import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# Step 1: 加载模型
# =====================================================================
print("=" * 60)
print("Step 1: 加载 MJCF 模型")
print("=" * 60)

model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_freefall.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"模型文件: {model_path}")
print(f"广义坐标维度 (nq): {model.nq}")
print(f"自由度 (nv):        {model.nv}")
print(f"刚体数 (nbody):     {model.nbody}")
print(f"关节数 (njnt):      {model.njnt}")
print(f"执行器数 (nu):      {model.nu}")
print(f"仿真步长:           {model.opt.timestep} s")

total_mass = sum(model.body_mass)
print(f"系统总质量:         {total_mass:.4f} kg")

# =====================================================================
# Step 2: 计算静态量 -- M(q0) 和 g(q0)
# =====================================================================
print("\n" + "=" * 60)
print("Step 2: 计算静态量")
print("=" * 60)

# 2.1 设定标准站立姿态 q0
q0 = np.zeros(model.nq)
q0[0:3] = [0.0, 0.0, 0.4]
q0[3:7] = [1.0, 0.0, 0.0, 0.0]
joint_angles = np.array([
    0.0, 0.8, -1.5,    # FAR (Front Right): ABAD, HIP, KNEE
    0.0, 0.8, -1.5,    # FBL (Front Left)
    0.0, 0.8, -1.5,    # RAR (Rear Right)
    0.0, 0.8, -1.5,    # RBL (Rear Left)
])
q0[7:19] = joint_angles

data.qpos[:] = q0
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# 2.2 计算广义质量矩阵 M(q0)
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)

print(f"\n--- 质量矩阵 M(q0) ---")
print(f"形状: {M.shape}")
print(f"条件数: {np.linalg.cond(M):.2f}")

# 2.3 验证对称性
symmetry_error = np.max(np.abs(M - M.T))
is_symmetric = symmetry_error < 1e-10
print(f"\n[对称性] max|M - M^T| = {symmetry_error:.2e}  {'PASS' if is_symmetric else 'FAIL'}")

# 2.4 验证正定性
eigenvalues = np.linalg.eigvalsh(M)
is_pd = np.all(eigenvalues > 0)
print(f"[正定性] lambda_min = {eigenvalues[0]:.6f}, lambda_max = {eigenvalues[-1]:.6f}  {'PASS' if is_pd else 'FAIL'}")

# 2.5 物理解读
print(f"\n[物理解读]")
print(f"  M[0,0] (x平移惯量) = {M[0,0]:.4f} kg  (应约等于总质量 {total_mass:.4f} kg)")
print(f"  M[1,1] (y平移惯量) = {M[1,1]:.4f} kg")
print(f"  M[2,2] (z平移惯量) = {M[2,2]:.4f} kg")

# 2.6 计算重力向量 g(q0)
data.qvel[:] = 0
mujoco.mj_forward(model, data)
g_vec = data.qfrc_bias.copy()

print(f"\n--- 重力向量 g(q0) ---")
print(f"  基座线性分量: [{g_vec[0]:.6f}, {g_vec[1]:.6f}, {g_vec[2]:.6f}]")
print(f"  g[2] (z方向)  = {g_vec[2]:.4f} N")
print(f"  理论值 +m*g   = {total_mass * 9.81:.4f} N")
grav_err = abs(g_vec[2] - total_mass * 9.81) / (total_mass * 9.81) * 100
print(f"  相对误差      = {grav_err:.4f}%")

# =====================================================================
# Step 3: 自由落体仿真
# =====================================================================
print("\n" + "=" * 60)
print("Step 3: 自由落体仿真")
print("=" * 60)

dt = 0.001
num_steps = 100
model.opt.timestep = dt

data.qpos[:] = q0.copy()
data.qvel[:] = 0
mujoco.mj_forward(model, data)

time_log = np.zeros(num_steps + 1)
pos_log = np.zeros((num_steps + 1, 3))
vel_log = np.zeros((num_steps + 1, 3))

time_log[0] = 0.0
pos_log[0] = data.qpos[0:3].copy()
vel_log[0] = data.qvel[0:3].copy()

print(f"步长: {dt} s,  步数: {num_steps},  总时长: {dt * num_steps} s")
print(f"初始位置: [{pos_log[0, 0]:.4f}, {pos_log[0, 1]:.4f}, {pos_log[0, 2]:.4f}]")

for step in range(num_steps):
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_step(model, data)
    time_log[step + 1] = (step + 1) * dt
    pos_log[step + 1] = data.qpos[0:3].copy()
    vel_log[step + 1] = data.qvel[0:3].copy()

print(f"最终位置: [{pos_log[-1, 0]:.6f}, {pos_log[-1, 1]:.6f}, {pos_log[-1, 2]:.6f}]")
delta_z = pos_log[0, 2] - pos_log[-1, 2]
theory_dz = 0.5 * 9.81 * (dt * num_steps) ** 2
print(f"z方向下落: {delta_z:.6f} m  (理论值: {theory_dz:.6f} m)")

# =====================================================================
# Step 4: 物理特征验证
# =====================================================================
print("\n" + "=" * 60)
print("Step 4: 物理特征验证")
print("=" * 60)

# 4.1 加速度计算 (中心差分)
acc_central = np.zeros((num_steps - 1, 3))
for i in range(1, num_steps):
    acc_central[i - 1] = (vel_log[i + 1] - vel_log[i - 1]) / (2 * dt)

print("\n--- 加速度验证 (中心差分) ---")
ax_mean = np.mean(acc_central[:, 0])
ay_mean = np.mean(acc_central[:, 1])
az_mean = np.mean(acc_central[:, 2])
print(f"  平均 a_x = {ax_mean:+.6f} m/s^2 (期望: 0)")
print(f"  平均 a_y = {ay_mean:+.6f} m/s^2 (期望: 0)")
print(f"  平均 a_z = {az_mean:+.6f} m/s^2 (期望: -9.81)")

rel_err_z = abs(az_mean - (-9.81)) / 9.81 * 100
print(f"\n  z方向相对误差: {rel_err_z:.4f}%")
print(f"  是否满足 < 1%: {'PASS' if rel_err_z < 1 else 'FAIL'}")

# 4.2 轨迹对比
z_theory = pos_log[0, 2] - 0.5 * 9.81 * time_log ** 2
vz_theory = -9.81 * time_log

pos_err_max = np.max(np.abs(pos_log[:, 2] - z_theory))
vel_err_max = np.max(np.abs(vel_log[:, 2] - vz_theory))
print(f"\n--- 轨迹对比 ---")
print(f"  位置最大误差: {pos_err_max:.8f} m")
print(f"  速度最大误差: {vel_err_max:.8f} m/s")

# =====================================================================
# 绘图
# =====================================================================
print("\n" + "=" * 60)
print("生成结果图表...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Project 1: Free-Fall Verification (XG Quadruped)", fontsize=14, fontweight="bold")

t_ms = time_log * 1000

ax = axes[0, 0]
ax.plot(t_ms, pos_log[:, 2], "b-", lw=2, label="Simulation")
ax.plot(t_ms, z_theory, "r--", lw=1.5, label="Theory: z0 - 0.5*g*t^2")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("z position (m)")
ax.set_title("(a) Base z-position")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(t_ms, vel_log[:, 2], "b-", lw=2, label="Simulation")
ax.plot(t_ms, vz_theory, "r--", lw=1.5, label="Theory: -g*t")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("z velocity (m/s)")
ax.set_title("(b) Base z-velocity")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
t_acc_ms = time_log[1:-1] * 1000
ax.plot(t_acc_ms, acc_central[:, 2], "b-", lw=2, label="Simulation")
ax.axhline(y=-9.81, color="r", ls="--", lw=1.5, label="Theory: -g")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("z acceleration (m/s^2)")
ax.set_title("(c) Base z-acceleration")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(t_ms, pos_log[:, 0], "-", lw=2, label="x (expect: 0)")
ax.plot(t_ms, pos_log[:, 1], "-", lw=2, label="y (expect: 0)")
ax.plot(t_ms, pos_log[:, 2] - pos_log[0, 2], "-", lw=2, label="delta z")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Displacement (m)")
ax.set_title("(d) All-axis displacement")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, "freefall_results.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"图表已保存至: {output_path}")
plt.close()

# =====================================================================
# Step 5: 生成自由落体动画 (GIF)
# =====================================================================
print("\n" + "=" * 60)
print("Step 5: 生成自由落体动画...")
print("=" * 60)

try:
    from PIL import Image, ImageDraw, ImageFont

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    vis_model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_converted.xml")
    vis_model = mujoco.MjModel.from_xml_path(vis_model_path)
    vis_model.vis.global_.offwidth = 800
    vis_model.vis.global_.offheight = 600

    for i in range(vis_model.ngeom):
        vis_model.geom_contype[i] = 0
        vis_model.geom_conaffinity[i] = 0

    vis_model.opt.timestep = 0.001
    vis_data = mujoco.MjData(vis_model)
    vis_data.qpos[:] = q0.copy()
    vis_data.qvel[:] = 0
    mujoco.mj_forward(vis_model, vis_data)

    width, height = 800, 600
    renderer = mujoco.Renderer(vis_model, height, width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        vis_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    cam.distance = 1.8
    cam.azimuth = 145
    cam.elevation = -15

    anim_duration = 1.5
    fps = 30
    steps_per_frame = int(1.0 / (vis_model.opt.timestep * fps))
    total_anim_steps = int(anim_duration / vis_model.opt.timestep)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    frames = []
    z_start = vis_data.qpos[2]
    print(f"  动画参数: 时长={anim_duration}s, 帧率={fps}fps, 总帧数≈{int(anim_duration * fps)}")

    for step in range(total_anim_steps + 1):
        if step % steps_per_frame == 0:
            renderer.update_scene(vis_data, cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)

            t_now = step * vis_model.opt.timestep
            z_now = vis_data.qpos[2]
            vz_now = vis_data.qvel[2]
            drop = z_start - z_now

            info_lines = [
                f"t = {t_now:.3f} s",
                f"z = {z_now:.3f} m",
                f"vz = {vz_now:.3f} m/s",
                f"drop = {drop:.4f} m",
            ]
            y_pos = 15
            for line in info_lines:
                draw.rectangle([12, y_pos - 2, 280, y_pos + 22], fill=(0, 0, 0, 160))
                draw.text((15, y_pos), line, fill=(0, 255, 100), font=font)
                y_pos += 26

            frames.append(img)

        if step < total_anim_steps:
            vis_data.ctrl[:] = 0
            vis_data.qfrc_applied[:] = 0
            mujoco.mj_step(vis_model, vis_data)

    gif_path = os.path.join(SCRIPT_DIR, "freefall_animation.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    renderer.close()

    z_end = vis_data.qpos[2]
    print(f"  下落范围: z = {z_start:.2f} → {z_end:.2f} m (下落 {z_start - z_end:.2f} m)")
    print(f"  动画已保存至: {gif_path}")

except ImportError as e:
    print(f"  [跳过] 缺少依赖: {e}")
    print("  运行 pip install Pillow 后重试")
except Exception as e:
    print(f"  [跳过] 动画生成失败: {e}")
    print("  提示: export MUJOCO_GL=egl 或 export MUJOCO_GL=osmesa")

# =====================================================================
# 总结
# =====================================================================
print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)
print(f"  质量矩阵对称性:       {'PASS' if is_symmetric else 'FAIL'}")
print(f"  质量矩阵正定性:       {'PASS' if is_pd else 'FAIL'}")
print(f"  重力向量 g[2] 误差:   {grav_err:.4f}%")
print(f"  基座加速度 a_z 误差:  {rel_err_z:.4f}%")
print(f"  自由落体验证:         {'PASS' if rel_err_z < 1 else 'FAIL'}")
print("=" * 60)
