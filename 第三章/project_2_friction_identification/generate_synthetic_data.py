"""
使用方法:
    conda activate freefall
    cd project_2_friction_identification
    python generate_synthetic_data.py
"""

import os
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_ACTUATOR = 12
DOF_OFFSET = 6

# =====================================================================
# 真值参数 φ*
# =====================================================================
# 关节顺序: FAR_ABAD, FAR_HIP, FAR_KNEE,
#           FBL_ABAD, FBL_HIP, FBL_KNEE,
#           RAR_ABAD, RAR_HIP, RAR_KNEE,
#           RBL_ABAD, RBL_HIP, RBL_KNEE

B_STAR = np.array([
    0.35, 0.50, 0.40,   # FAR
    0.38, 0.52, 0.42,   # FBL
    0.36, 0.48, 0.39,   # RAR
    0.37, 0.51, 0.41,   # RBL
])

FC_STAR = np.array([
    0.20, 0.30, 0.25,   # FAR
    0.22, 0.28, 0.24,   # FBL
    0.21, 0.31, 0.26,   # RAR
    0.23, 0.29, 0.25,   # RBL
])

IR_STAR = np.array([
    0.010, 0.015, 0.012,  # FAR
    0.011, 0.014, 0.013,  # FBL
    0.010, 0.016, 0.012,  # RAR
    0.011, 0.015, 0.013,  # RBL
])


def inject_true_params(model, level="standard"):
    """将真值参数注入模型。"""
    for i in range(N_ACTUATOR):
        dof_id = DOF_OFFSET + i
        model.dof_damping[dof_id] = B_STAR[i]
        model.dof_frictionloss[dof_id] = FC_STAR[i]
        if level == "advanced":
            model.dof_armature[dof_id] = IR_STAR[i]


def run_trajectory(model, data, ctrl_fn, duration_s=5.0):
    """
    运行仿真并记录完整数据。

    返回:
        q_log:   (T, nq)  广义坐标
        dq_log:  (T, nv)  广义速度
    """
    dt = model.opt.timestep
    n_steps = int(duration_s / dt)

    q_log = np.zeros((n_steps, model.nq))
    dq_log = np.zeros((n_steps, model.nv))

    for step in range(n_steps):
        t = step * dt
        data.ctrl[:] = ctrl_fn(t, data)
        mujoco.mj_step(model, data)
        q_log[step] = data.qpos.copy()
        dq_log[step] = data.qvel.copy()

    return q_log, dq_log


def make_chirp_ctrl(amplitudes, phases, f_start, f_end, duration_s):
    """生成 chirp 激励控制函数。"""
    def ctrl_fn(t, data):
        freq = f_start + (f_end - f_start) * t / duration_s
        ctrl = np.zeros(N_ACTUATOR)
        for j in range(N_ACTUATOR):
            ctrl[j] = amplitudes[j] * np.sin(2 * np.pi * freq * t + phases[j])
        return ctrl
    return ctrl_fn


def make_trot_ctrl(default_angles, kp=20.0, kd=0.5, trot_freq=2.0,
                   amp_hip=0.3, amp_knee=0.5):
    """生成 trot 步态 PD 控制函数。"""
    def ctrl_fn(t, data):
        sin_val = np.sin(2 * np.pi * trot_freq * t)
        cos_val = np.cos(2 * np.pi * trot_freq * t)
        target = default_angles.copy()
        target[1] += amp_hip * sin_val     # FAR HIP
        target[2] += amp_knee * sin_val    # FAR KNEE
        target[4] += amp_hip * cos_val     # FBL HIP
        target[5] += amp_knee * cos_val    # FBL KNEE
        target[7] += amp_hip * cos_val     # RAR HIP
        target[8] += amp_knee * cos_val    # RAR KNEE
        target[10] += amp_hip * sin_val    # RBL HIP
        target[11] += amp_knee * sin_val   # RBL KNEE

        dof_pos = data.qpos[7:19]
        dof_vel = data.qvel[DOF_OFFSET:DOF_OFFSET + N_ACTUATOR]
        return kp * (target - dof_pos) - kd * dof_vel
    return ctrl_fn


def compute_ddv(dq_log, dt):
    """中心差分计算广义加速度。"""
    ddv = np.zeros_like(dq_log)
    ddv[1:-1] = (dq_log[2:] - dq_log[:-2]) / (2 * dt)
    ddv[0] = (dq_log[1] - dq_log[0]) / dt
    ddv[-1] = (dq_log[-1] - dq_log[-2]) / dt
    return ddv


def add_noise(arr, noise_ratio=0.01, seed=0):
    """添加高斯噪声（相对于数据标准差）。"""
    rng = np.random.RandomState(seed)
    std = np.std(arr, axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    return arr + rng.randn(*arr.shape) * std * noise_ratio


def init_standing(model, data):
    """初始化为站立姿态。"""
    q0 = np.zeros(model.nq)
    q0[0:3] = [0.0, 0.0, 0.4]
    q0[3:7] = [1.0, 0.0, 0.0, 0.0]
    q0[7:19] = [0.0, 0.8, -1.5] * 4
    data.qpos[:] = q0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def main():
    print("=" * 60)
    print("Project 2: 生成合成\"实机\"数据")
    print("=" * 60)

    model_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_friction_id.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    print(f"模型文件: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"步长: {dt} s")

    default_angles = np.array([0.0, 0.8, -1.5] * 4)

    rng = np.random.RandomState(42)
    chirp_amps = rng.uniform(2.0, 5.0, size=N_ACTUATOR)
    chirp_phases = rng.uniform(0, 2 * np.pi, size=N_ACTUATOR)

    for level in ["basic", "standard", "advanced"]:
        print(f"\n{'='*60}")
        print(f"生成 [{level}] 级别数据...")
        print(f"{'='*60}")

        model_lvl = mujoco.MjModel.from_xml_path(model_path)
        inject_true_params(model_lvl, level=level if level == "advanced" else "standard")

        # chirp 激励
        print("  → chirp 激励 (5s)...")
        data_c = mujoco.MjData(model_lvl)
        init_standing(model_lvl, data_c)
        chirp_fn = make_chirp_ctrl(chirp_amps, chirp_phases, 0.5, 8.0, 5.0)
        q_c, dq_c = run_trajectory(model_lvl, data_c, chirp_fn, 5.0)

        # trot 激励
        print("  → trot 步态 (5s)...")
        data_t = mujoco.MjData(model_lvl)
        init_standing(model_lvl, data_t)
        trot_fn = make_trot_ctrl(default_angles)
        q_t, dq_t = run_trajectory(model_lvl, data_t, trot_fn, 5.0)

        # 合并
        dq_all = np.concatenate([dq_c, dq_t])
        ddv_all = compute_ddv(dq_all, dt)

        dq_joint = dq_all[:, DOF_OFFSET:DOF_OFFSET + N_ACTUATOR]
        ddv_joint = ddv_all[:, DOF_OFFSET:DOF_OFFSET + N_ACTUATOR]

        # 从已知真值直接计算摩擦力目标
        if level == "basic":
            friction_target = B_STAR[np.newaxis, :] * dq_joint
        elif level == "standard":
            friction_target = B_STAR[np.newaxis, :] * dq_joint + \
                              FC_STAR[np.newaxis, :] * np.sign(dq_joint)
        else:  # advanced
            friction_target = B_STAR[np.newaxis, :] * dq_joint + \
                              FC_STAR[np.newaxis, :] * np.sign(dq_joint) + \
                              IR_STAR[np.newaxis, :] * ddv_joint

        # 加 1% 噪声
        dq_joint_noisy = add_noise(dq_joint, 0.01, seed=1)
        ddv_joint_noisy = add_noise(ddv_joint, 0.01, seed=2)
        friction_target_noisy = add_noise(friction_target, 0.01, seed=3)

        # 80/20 split
        n_total = dq_joint_noisy.shape[0]
        n_train = int(n_total * 0.8)
        idx = np.random.RandomState(99).permutation(n_total)
        ti, vi = idx[:n_train], idx[n_train:]

        out = {
            "dq_joint_train": dq_joint_noisy[ti],
            "ddv_joint_train": ddv_joint_noisy[ti],
            "friction_target_train": friction_target_noisy[ti],
            "dq_joint_val": dq_joint_noisy[vi],
            "ddv_joint_val": ddv_joint_noisy[vi],
            "friction_target_val": friction_target_noisy[vi],
            "dt": dt,
        }

        if level == "basic":
            out["phi_star"] = B_STAR
            out["phi_dim"] = 12
        elif level == "standard":
            out["phi_star"] = np.concatenate([B_STAR, FC_STAR])
            out["phi_dim"] = 24
        else:
            out["phi_star"] = np.concatenate([B_STAR, FC_STAR, IR_STAR])
            out["phi_dim"] = 36

        out_path = os.path.join(SCRIPT_DIR, f"data_{level}.npz")
        np.savez(out_path, **out)

        # 打印数据统计
        print(f"  |dq_joint| range: [{np.min(np.abs(dq_joint)):.4f}, {np.max(np.abs(dq_joint)):.4f}]")
        print(f"  |friction_target| range: [{np.min(np.abs(friction_target)):.6f}, {np.max(np.abs(friction_target)):.4f}]")
        print(f"  训练样本: {n_train}, 验证样本: {n_total - n_train}")
        print(f"  φ* 维度: {out['phi_dim']}")
        print(f"  保存至: {out_path}")

    print(f"\n{'='*60}")
    print("数据生成完成！")
    print(f"{'='*60}")
    print("  data_basic.npz    → 基础档 (b, 12维)")
    print("  data_standard.npz → 标准档 (b + f_c, 24维)")
    print("  data_advanced.npz → 进阶档 (b + f_c + I_r, 36维)")


if __name__ == "__main__":
    main()
