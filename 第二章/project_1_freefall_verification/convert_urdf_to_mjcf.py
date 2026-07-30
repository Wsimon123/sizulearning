"""
辅助脚本: 将 URDF 转换为 MJCF 格式

MuJoCo 加载 URDF 时要求 mesh 文件与 URDF 在同一目录下,
本脚本通过临时设置工作目录来正确加载 URDF 并导出 MJCF。

使用方法:
    conda activate freefall
    cd project_1_freefall_verification
    python convert_urdf_to_mjcf.py
"""

import os
import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("URDF → MJCF 转换演示")
print("=" * 60)

urdf_dir = os.path.join(SCRIPT_DIR, "resources", "xg", "urdf")
urdf_file = os.path.join(urdf_dir, "xg.urdf")

print(f"URDF 文件: {urdf_file}")
print(f"URDF 目录: {urdf_dir}")

original_cwd = os.getcwd()
os.chdir(urdf_dir)

try:
    model = mujoco.MjModel.from_xml_path("xg.urdf")
    print(f"\nURDF 加载成功!")
    print(f"  自由度 (nv):      {model.nv}")
    print(f"  广义坐标维度 (nq): {model.nq}")
    print(f"  刚体数 (nbody):    {model.nbody}")
    print(f"  关节数 (njnt):     {model.njnt}")
    print(f"  系统总质量:        {sum(model.body_mass):.4f} kg")
    print()
    print("  [注意] URDF 默认是固定基座 (nv=12), 没有浮动基座的 6 个自由度。")
    print("  转换为 MJCF 后需要手动添加 <freejoint/> 才能模拟浮动基座。")
    print("  本项目提供的 xg_freefall.xml 和 xg_converted.xml 已经添加了 freejoint (nv=18)。")

    output_path = os.path.join(SCRIPT_DIR, "resources", "xg", "xg_from_urdf.xml")
    mujoco.mj_saveLastXML(output_path, model)
    print(f"\nMJCF 已导出至: {output_path}")

finally:
    os.chdir(original_cwd)

print("\n" + "=" * 60)
print("转换完成!")
print("=" * 60)
print("\n提示: 自动转换的 MJCF 可能需要手动优化, 例如:")
print("  1. 将 mesh 碰撞体替换为简单几何体 (提高仿真速度)")
print("  2. 调整接触参数 (摩擦、刚度等)")
print("  3. 添加执行器 (actuator) 定义")
print("  4. 添加传感器 (sensor) 定义")
print("\n本项目已提供优化后的 MJCF:")
print("  - resources/xg/xg_freefall.xml   (自由落体验证专用)")
print("  - resources/xg/xg_converted.xml  (完整版, 含地面和 mesh 渲染)")
