# NIA Robot — AI Environment Setup
## Jetson Orin Nano 8GB | L4T 36.4.7 | JetPack 6.2.1

---

## Hardware
| Component | Spec |
|---|---|
| Device | Jetson Orin Nano 8GB |
| L4T Version | 36.4.7 |
| JetPack | 6.2.1 |
| CUDA | 12.8 |
| cuDNN | 9.3 |
| TensorRT | 10.x |
| OS | Ubuntu 22.04 (host) |
| Camera | Intel RealSense D435 |
| Arm Controller | ESP32 via USB UART 115200 baud |
| Motors | NEMA Stepper motors |

---

## Container Stack
| Container | Image | Purpose |
|---|---|---|
| robot_brain | dustynv/ultralytics:r36.4.0 | AI brain — perception + world model + planner |
| redis | redis/redis-stack:7.4.0-v1 | Object/spatial/experience memory |
| neo4j | neo4j:5 | Scene graph — spatial relations |
| local_llm | dustynv/local_llm:r36.4.0 | Gemma 3 1B — offline LLM inference |

---

## AI Models
| Model | Version | Purpose | Speed |
|---|---|---|---|
| YOLOv8n | TensorRT FP16 | Object detection | ~8ms/frame |
| YOLOv8n-seg | TensorRT FP16 | Instance segmentation | ~12ms/frame |
| CLIP ViT-B/32 | OpenAI | Object identity verification | ~15ms |
| Gemma 3 1B | W4A16 quantized | Local LLM planning | ~100 tok/s |
| sentence-transformers | all-MiniLM-L6-v2 | Experience memory retrieval | ~5ms |
| DeepSORT | - | Multi-object tracking | ~3ms |

---

## Python Packages (pip install -r requirements.txt)
```
ultralytics>=8.3.0
deep-sort-realtime>=1.3.2
Pillow>=10.0.0
ftfy>=6.1.0
regex>=2023.0.0
neo4j>=5.0.0
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
openai>=1.40.0
sentence-transformers>=3.0.0
pydantic>=2.0.0
redis>=5.0.0
scipy>=1.11.0
typing-extensions>=4.9.0
pyserial>=3.5
```

---

## Packages from Base Image (DO NOT pip install)
| Package | Version | Source |
|---|---|---|
| PyTorch | 2.x | dustynv/ultralytics base image |
| torchvision | 0.x | dustynv/ultralytics base image |
| OpenCV CUDA | 4.x | dustynv/ultralytics base image |
| TensorRT | 10.x | JetPack pre-installed |
| FAISS | latest | jetson-containers pip server |

---

## Separately Installed
| Package | Install Command |
|---|---|
| OpenAI CLIP | pip install git+https://github.com/openai/CLIP.git |
| pyrealsense2 | apt install librealsense2-dev + pip install pyrealsense2 |
| ROS2 Humble | apt install ros-humble-rclpy ros-humble-std-msgs |

---

## Environment Variables (.env)
```
OPENAI_API_KEY=sk-...           # Cloud LLM fallback
NEO4J_PASSWORD=your_password
LLM_MODE=auto                   # auto = local first, cloud fallback
LOCAL_LLM_URL=http://local_llm:9000/v1
LOCAL_LLM_MODEL=google/gemma-3-1b-it
SHOW_DISPLAY=0
EXECUTOR_MOCK=0
ROS_DOMAIN_ID=42
REDIS_URL=redis://redis:6379
NEO4J_URI=bolt://neo4j:7687
```

---

## ROS2 Topics (Brain publishes → Hardware receives)
| Topic | Type | Action |
|---|---|---|
| /arm/cmd | std_msgs/String | ARM_EXTEND, ARM_RETRACT, ARM_HOME |
| /gripper/cmd | std_msgs/String | GRIP_CLOSE, GRIP_OPEN, GRIP_ALIGN |
| /cmd_vel | geometry_msgs/Twist | Wheel motor speed |
| /head/cmd | std_msgs/String | HEAD_LEFT, HEAD_RIGHT, HEAD_CENTER |
| /audio/speak | std_msgs/String | Text to speech |

---

## Startup Order
```bash
# 1. Install jetson-containers
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# 2. Start all containers
docker compose up

# 3. Run robot brain
python3 main.py
```

---

## AI Code Modules (26 files)
```
Perception/          — camera, YOLO, tracking, depth, CLIP, segmentation
world_model/         — Redis memory, FAISS vectors, Neo4j scene graph
planner/             — LangGraph, LLM client, executor, experience memory
main.py              — startup, wires all 6 integration handshakes
ros2_bridge.py       — ROS2 subscriber bridge (Sanjay integration)
executor_interface.py — UART/ROS2 publisher (Prithvi ESP32 integration)
```
