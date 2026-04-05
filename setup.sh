#!/usr/bin/env bash
# ============================================================
#  AMR Robot Brain – Jetson JetPack 6.2 / Ubuntu 22.04
#  CUDA 12.6 / Python 3.10
# ============================================================
set -euo pipefail

echo "=== [1/5] System dependencies ==="
sudo apt-get update -y
sudo apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libopencv-dev python3-opencv \
    librealsense2-dev librealsense2-utils \
    
echo "=== [2/5] PyTorch for Jetson (JetPack 6 / CUDA 12.6) ==="
# Official NVIDIA Jetson PyTorch wheel
TORCH_WHL="https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/torch-2.3.0a0+40ec155e58.nv24.05.14593579-cp310-cp310-linux_aarch64.whl"
pip3 install --no-cache-dir "${TORCH_WHL}"
# torchvision matching torch 2.3
pip3 install --no-cache-dir \
    "torchvision==0.18.0" \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch

echo "=== [3/5] Python requirements ==="
pip3 install --no-cache-dir -r requirements.txt

echo "=== [4/5] Verify CUDA availability ==="
python3 - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
EOF

echo "=== [5/5] Done — run with: python3 main.py ==="
