"""
可视化仿真运行画面 —— 离屏渲染输出 MP4 视频。

用法:
    python visualize_simulation.py                 # 默认: standard + trot
    python visualize_simulation.py chirp           # chirp 扫频激励
    python visualize_simulation.py trot            # trot 步态
    python visualize_simulation.py standard chirp  # 指定级别 + 激励

输出:
    simulation_trot_standard.mp4  (或其他组合的文件名)
"""

import os
import sys
import numpy as np
import mujoco
import imageio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_ACTUATOR = 12
DOF_OFFSET = 6

# ---------- 真值参数 ----------
B_STAR = np.array([0.35, 0.50, 0.40, 0.38, 0.52, 0.42,
                   0.36, 0.48, 0.39, 0.37, 0.51, 0.41])
FC_STAR = np.array([0.20, 0.30, 0.25, 0.22, 0.28, 0.24,
                    0.21, 0.31, 0.26, 0.23, 0.29, 0.25])
IR_STAR = np.array([0.010, 0.015, 0.012, 0.011, 0.014, 0.013,
                    0.010, 0.016, 0.012, 0.011, 0.015, 0.013])

DEFAULT_ANGLES = np.array([0.0, 0.8, -1.5] * 4)


def load_model(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        return mujoco.MjModel.from_xml_string(f.read())


def inject_friction(model, level):
    for i in range(N_ACTUATOR):
        dof_id = DOF_OFFSET + i
        model.dof_damping[dof_id] = B_STAR[i]
        model.dof_frictionloss[dof_id] = FC_STAR[i]
        if level == "advanced":
            model.dof_armature[dof_id] = IR_STAR[i]


def init_standing(model, data):
    data.qpos[:] = [0, 0, 0.4, 1, 0, 0, 0] + DEFAULT_ANGLES.tolist()
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def trot_ctrl_fn(t):
    sin_val = np.sin(2 * np.pi * 2.0 * t)
    cos_val = np.cos(2 * np.pi * 2.0 * t)
    target = DEFAULT_ANGLES.copy()
    target[1] += 0.3 * sin_val
    target[2] += 0.5 * sin_val
    target[4] += 0.3 * cos_val
    target[5] += 0.5 * cos_val
    target[7] += 0.3 * cos_val
    target[8] += 0.5 * cos_val
    target[10] += 0.3 * sin_val
    target[11] += 0.5 * sin_val
    return target


def chirp_ctrl_fn(t, amps, phases, f_start, f_end, duration):
    freq = f_start + (f_end - f_start) * t / duration
    ctrl = np.zeros(N_ACTUATOR)
    for j in range(N_ACTUATOR):
        ctrl[j] = amps[j] * np.sin(2 * np.pi * freq * t + phases[j])
    return ctrl


def main():
    level = "standard"
    mode = "trot"

    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in ("basic", "standard", "advanced"):
            level = arg_lower
        elif arg_lower in ("trot", "chirp"):
            mode = arg_lower

    # 加载模型
    model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_friction_id.xml")
    model = load_model(model_path)
    inject_friction(model, level)
    data = mujoco.MjData(model)
    init_standing(model, data)

    dt = model.opt.timestep
    duration = 5.0
    video_fps = 30                        # 输出视频帧率
    render_every = max(1, int(1.0 / (dt * video_fps)))  # 每 N 个物理步渲染一帧
    n_frames = int(duration * video_fps)

    # chirp 参数
    rng = np.random.RandomState(42)
    chirp_amps = rng.uniform(2.0, 5.0, size=N_ACTUATOR)
    chirp_phases = rng.uniform(0, 2 * np.pi, size=N_ACTUATOR)

    # 离屏渲染器
    renderer = mujoco.Renderer(model, 480, 640)

    out_name = os.path.join(SCRIPT_DIR, f"simulation_{mode}_{level}.mp4")
    print(f"级别: {level}  激励: {mode}")
    print(f"输出: {out_name}")
    print(f"帧率: {video_fps} fps  时长: {duration}s  总帧数: {n_frames}")
    print("渲染中...")

    writer = imageio.get_writer(out_name, fps=video_fps, codec="libx264")

    for frame_idx in range(n_frames):
        # 跑 render_every 个物理步
        for _ in range(render_every):
            t = data.time

            if mode == "trot":
                target = trot_ctrl_fn(t)
                dof_pos = data.qpos[7:19]
                dof_vel = data.qvel[DOF_OFFSET:DOF_OFFSET + N_ACTUATOR]
                data.ctrl[:] = 20.0 * (target - dof_pos) - 0.5 * dof_vel
            else:
                data.ctrl[:] = chirp_ctrl_fn(t, chirp_amps, chirp_phases,
                                             0.5, 8.0, duration)

            mujoco.mj_step(model, data)

        # 渲染当前帧
        renderer.update_scene(data)
        pixels = renderer.render()
        writer.append_data(pixels)

        if (frame_idx + 1) % 30 == 0:
            print(f"  {frame_idx + 1}/{n_frames} 帧")

    writer.close()
    renderer.close()
    print(f"完成！视频已保存至: {out_name}")


if __name__ == "__main__":
    main()
