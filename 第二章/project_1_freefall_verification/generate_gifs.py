"""
生成自由落体动画 GIF — 多姿态 & 多步长

为每种姿态和每种步长生成自由落体动画, 保存在各自的结果文件夹中。

使用方法:
    conda activate freefall  (或 pip install mujoco pillow)
    cd project_1_freefall_verification
    python generate_gifs.py
"""

import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 配置
# =====================================================================
ANIM_DURATION = 1.0   # 动画时长 (s)
FPS = 30
WIDTH, HEIGHT = 800, 600

# ---- 多姿态配置 ----
POSES = {
    "standing": {
        "angles": np.array([
            0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
            0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
        ]),
        "desc": "Standard Standing",
    },
    "all_zeros": {
        "angles": np.zeros(12),
        "desc": "All Zeros (KNEE out of limit!)",
    },
    "crouching": {
        "angles": np.array([
            0.3, 1.5, -2.5, 0.3, 1.5, -2.5,
            0.3, 1.5, -2.5, 0.3, 1.5, -2.5,
        ]),
        "desc": "Crouching",
    },
    "stretched": {
        "angles": np.array([
            0.0, -0.5, -0.8, 0.0, -0.5, -0.8,
            0.0, -0.5, -0.8, 0.0, -0.5, -0.8,
        ]),
        "desc": "Stretched Forward",
    },
    "asymmetric": {
        "angles": np.array([
            0.4, 0.5, -1.0, -0.4, 2.0, -2.5,
            0.0, 1.2, -1.8, -0.3, -0.3, -0.7,
        ]),
        "desc": "Asymmetric",
    },
    "random_within_limits": {
        "angles": np.array([
            0.2, 1.0, -1.0, -0.1, 0.3, -1.8,
            0.35, 2.2, -2.0, -0.25, 0.9, -1.3,
        ]),
        "desc": "Random (within limits)",
    },
}

# ---- 多步长配置 ----
TIMESTEPS = {
    "0.1ms": 0.0001,
    "1ms":   0.001,
    "5ms":   0.005,
    "10ms":  0.01,
}
TIMESTEP_ANGLES = np.array([0.0, 0.8, -1.5] * 4)  # standing pose for timestep comparison


# =====================================================================
# 工具函数
# =====================================================================

def load_model():
    model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_freefall.xml")
    with open(model_path, "r", encoding="utf-8") as f:
        return f.read()

def setup_vis_model(model_xml, dt):
    """创建可视化模型并配置相机"""
    model = mujoco.MjModel.from_xml_string(model_xml)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    # 确保无接触
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    return model, data

def setup_camera(model):
    """配置跟踪相机"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 1.8
    cam.azimuth = 145
    cam.elevation = -15
    return cam

def build_q0(angles, base_z=0.4, nq=19):
    """构建 qpos"""
    q0 = np.zeros(nq)
    q0[0:3] = [0.0, 0.0, base_z]
    q0[3:7] = [1.0, 0.0, 0.0, 0.0]
    q0[7:19] = angles
    return q0

def get_font():
    """获取可用字体"""
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, 18)
            except Exception:
                continue
    return ImageFont.load_default()

def render_info_text(draw, font, t, z, vz, drop, extra_lines=None):
    """在帧上绘制信息"""
    info = [
        f"t = {t:.3f} s",
        f"z = {z:.4f} m",
        f"vz = {vz:.4f} m/s",
        f"drop = {drop:.4f} m",
    ]
    if extra_lines:
        info.extend(extra_lines)
    y_pos = 15
    for line in info:
        draw.rectangle([12, y_pos - 2, 320, y_pos + 22], fill=(0, 0, 0, 160))
        draw.text((15, y_pos), line, fill=(0, 255, 100), font=font)
        y_pos += 26

def generate_gif(model_xml, angles, dt, output_path, desc="", extra_info=None):
    """生成一个自由落体 GIF 动画"""
    model, data = setup_vis_model(model_xml, dt)
    cam = setup_camera(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    font = get_font()

    q0 = build_q0(angles, nq=model.nq)
    data.qpos[:] = q0.copy()
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    total_steps = int(ANIM_DURATION / dt)
    steps_per_frame = max(1, int(1.0 / (dt * FPS)))
    z_start = data.qpos[2]

    frames = []
    for step in range(total_steps + 1):
        if step % steps_per_frame == 0:
            renderer.update_scene(data, cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)

            t_now = step * dt
            z_now = data.qpos[2]
            vz_now = data.qvel[2]
            drop = z_start - z_now

            extra = [desc]
            if extra_info:
                extra.extend(extra_info)
            render_info_text(draw, font, t_now, z_now, vz_now, drop, extra)

            frames.append(img)

        if step < total_steps:
            data.ctrl[:] = 0
            data.qfrc_applied[:] = 0
            mujoco.mj_step(model, data)

    # 保存 GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
    )
    renderer.close()
    return z_start, data.qpos[2]


# =====================================================================
# 主流程
# =====================================================================

model_xml = load_model()

# ====== 第一部分: 多姿态 GIF ======
print("=" * 60)
print("Part 1: 多姿态自由落体 GIF")
print("=" * 60)

for pose_name, pose_info in POSES.items():
    angles = pose_info["angles"]
    desc = pose_info["desc"]

    out_dir = os.path.join(SCRIPT_DIR, f"results_pose_{pose_name}")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"freefall_{pose_name}.gif")

    print(f"\n生成动画: {pose_name} ({desc})")
    print(f"  关节角度: {angles}")

    # 检查膝关节是否超出限位
    knee_angles = angles[[2, 5, 8, 11]]  # FAR_KNEE, FBL_KNEE, RAR_KNEE, RBL_KNEE
    knee_ok = np.all((knee_angles >= -2.723) & (knee_angles <= -0.602))
    extra_info = None
    if not knee_ok:
        extra_info = ["WARNING: KNEE out of range!", "Range: [-2.723, -0.602] rad"]

    try:
        z_start, z_end = generate_gif(
            model_xml, angles, dt=0.001, output_path=gif_path,
            desc=desc, extra_info=extra_info
        )
        drop = z_start - z_end
        print(f"  下落: z = {z_start:.2f} → {z_end:.2f} m (Δz = {drop:.4f} m)")
        print(f"  已保存: {gif_path}")
    except Exception as e:
        print(f"  [跳过] 生成失败: {e}")

# ====== 第二部分: 多步长 GIF ======
print("\n" + "=" * 60)
print("Part 2: 多步长自由落体 GIF")
print("=" * 60)

angles = TIMESTEP_ANGLES  # 统一使用 standing 姿态

for label, dt in TIMESTEPS.items():
    out_dir = os.path.join(SCRIPT_DIR, f"results_{label}")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"freefall_dt_{label}.gif")

    total_steps = int(ANIM_DURATION / dt)
    print(f"\n生成动画: dt = {label} ({dt:.5f} s), 总步数 = {total_steps}")

    desc = f"dt = {label} ({dt:.5f} s)"
    extra_info = [f"Steps: {total_steps}", f"Total: {ANIM_DURATION}s"]

    try:
        z_start, z_end = generate_gif(
            model_xml, angles, dt=dt, output_path=gif_path,
            desc=desc, extra_info=extra_info
        )
        theory_drop = 0.5 * 9.81 * ANIM_DURATION ** 2
        actual_drop = z_start - z_end
        print(f"  下落: z = {z_start:.2f} → {z_end:.2f} m")
        print(f"  Δz = {actual_drop:.4f} m (理论: {theory_drop:.4f} m)")
        print(f"  已保存: {gif_path}")
    except Exception as e:
        print(f"  [跳过] 生成失败: {e}")

print("\n" + "=" * 60)
print("所有 GIF 生成完成!")
print("=" * 60)
