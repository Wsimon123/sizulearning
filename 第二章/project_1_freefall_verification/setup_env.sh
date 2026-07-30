#!/bin/bash
# Project 1: 自由落体验证 - 环境配置脚本
# 使用方法: bash setup_env.sh

set -e

ENV_NAME="freefall"

echo "====================================="
echo " Project 1: 自由落体验证 - 环境配置"
echo "====================================="

# 检查 conda 是否已安装
if ! command -v conda &> /dev/null; then
    echo "[错误] 未检测到 conda，请先安装 Miniconda 或 Anaconda。"
    echo "安装命令:"
    echo "  mkdir -p ~/miniconda3"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh"
    echo "  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3"
    echo "  rm ~/miniconda3/miniconda.sh"
    echo "  ~/miniconda3/bin/conda init --all"
    echo "  source ~/.bashrc"
    exit 1
fi

echo "[1/3] 创建 conda 环境: ${ENV_NAME} (Python 3.10) ..."
conda create -n ${ENV_NAME} python=3.10 -y

echo "[2/3] 激活环境并安装依赖 ..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
pip install -r requirements.txt

echo "[3/3] 验证安装 ..."
python -c "import mujoco; print(f'mujoco {mujoco.__version__} 安装成功')"
python -c "import numpy; print(f'numpy {numpy.__version__} 安装成功')"
python -c "import matplotlib; print(f'matplotlib {matplotlib.__version__} 安装成功')"
python -c "from PIL import Image; print('Pillow 安装成功')"

echo ""
echo "====================================="
echo " 环境配置完成!"
echo "====================================="
echo ""
echo "使用方法:"
echo "  conda activate ${ENV_NAME}"
echo "  cd project_1_freefall_verification"
echo "  python freefall_verification.py"
echo ""
echo "运行后将自动生成:"
echo "  - freefall_results.png   (数据分析图表)"
echo "  - freefall_animation.gif (自由落体 3D 动画)"
echo ""
