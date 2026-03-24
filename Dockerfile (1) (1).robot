# ================================================================
#  AMR Robot — Dockerfile (jetson-containers edition)
#  NIA Semi-Humanoid Platform
#  L4T 36.4.7 / JetPack 6.2.1 / CUDA 12.8 / Ubuntu 24.04
#
#  AUTOTAG WARNING — 36.4.7:
#  autotag prints "[Warn] Unknown L4T_VERSION: 36.4.7" but still
#  resolves the correct r36.4 image. Safe to ignore.
#
#  HOW TO BUILD:
#    Option A — autotag development mode:
#      jetson-containers run \
#        -v $(pwd):/workspace \
#        $(autotag ultralytics)
#
#    Option B — production build:
#      docker build -f Dockerfile.robot -t amr_robot .
#      docker compose up
#
#  BASE IMAGE gives you (pre-built ARM64 Jetson):
#    PyTorch 2.x + CUDA 12.8 + TensorRT + cuDNN
#    OpenCV CUDA build + Ultralytics YOLOv8
#    numpy, cmake, onnx, torchvision
# ================================================================

FROM dustynv/ultralytics:r36.4.0

# ── System packages ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 libxext6 libxrender-dev \
    curl wget \
    && echo 'deb https://librealsense.intel.com/Debian/apt-repo jammy main' \
       | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update && apt-get install -y \
       librealsense2-dev \
       librealsense2-utils \
    && apt-get install -y \
       ros-humble-rclpy \
       ros-humble-std-msgs \
       ros-humble-geometry-msgs \
       ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── RealSense Python bindings ──────────────────────────────────
RUN pip install --no-cache-dir pyrealsense2

# ── pip packages ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── OpenAI CLIP — must install from GitHub NOT pip ────────────
# "pip install clip" installs a completely different package
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# ── FAISS ARM64 ────────────────────────────────────────────────
RUN pip install --no-cache-dir faiss-cpu || \
    echo "faiss-cpu failed — run: jetson-containers run $(autotag faiss_lite)"

# ── Pre-download models — baked in so robot works offline ──────
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-seg.pt')"
RUN python3 -c "import clip; clip.load('ViT-B/32')" || \
    echo "CLIP download failed — will download on first run"
RUN python3 -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy project source ────────────────────────────────────────
COPY Perception/   ./Perception/
COPY world_model/  ./world_model/
COPY planner/      ./planner/
COPY main.py       ./main.py

# ── Environment defaults (overridden by docker-compose / .env) ─
ENV PYTHONPATH=/workspace
ENV REDIS_URL=redis://redis:6379
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV FAISS_INDEX_PATH=/workspace/data/object_index.faiss

# NEW — local LLM settings (Gemma 3 1B on Jetson)
# LLM_MODE=auto: tries local first, falls back to cloud
ENV LLM_MODE=auto
ENV LOCAL_LLM_URL=http://local_llm:9000/v1
ENV LOCAL_LLM_MODEL=google/gemma-3-1b-it

# Robot settings
ENV SHOW_DISPLAY=0
ENV EXECUTOR_MOCK=0
ENV ROS_DOMAIN_ID=42
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python3", "main.py"]
