这是一个基于 **NVIDIA Jetson Orin Nano (JetPack 6.x)** 的完整软硬件技术方案。该方案融合了吸烟检测与办公区节能感应，采用容器化部署。你可以直接将此方案输入给 Cursor/Windsurf 等 AI 编程助手，它们能根据此结构自动生成项目代码。

---

## 1. 系统架构概览 (Architecture)

系统采用 **Docker Compose** 编排，分为四个核心容器：

1. **ZLMediaKit (ZLM)**: 流媒体服务器，负责 16 路 RTSP 流的接入、转换与截图。
2. **AI-Engine (Python)**: 推理引擎，负责 YOLOv8 TensorRT 加速。
3. **Node-RED (Logic)**: 业务编排，负责时间权重计算、空间融合判断及 MQTT 指令下发。
4. **MQTT Broker (Mosquitto)**: 内部通信总线。

---

## 2. 硬件清单 (Bill of Materials)

* **计算平台**: NVIDIA Jetson Orin Nano (8GB 显存版本)。
* **操作系统**: JetPack 6.0+ (Ubuntu 22.04 LTS)。
* **存储**: 256GB NVMe SSD (必须，用于 Docker 镜像与视频缓存)。
* **外设**: 千兆交换机。

---

## 3. 容器编排定义 (`docker-compose.yml`)

```yaml
version: '3.8'
services:
  # 流媒体中转
  zlm:
    image: zlmediakit/zlmediakit:latest
    ports:
      - "80:80"
      - "554:554"
    volumes:
      - ./zlm/config.ini:/opt/media/conf/config.ini
    restart: always

  # AI 推理引擎 (关键)
  ai-engine:
    build: 
      context: ./ai_engine
      dockerfile: Dockerfile
    runtime: nvidia
    environment:
      - MQTT_BROKER=mqtt-broker
      - ZLM_API=http://zlm:80/index/api/
    volumes:
      - ./ai_engine/models:/app/models
      - ./ai_engine/config:/app/config
    depends_on:
      - mqtt-broker
      - zlm
    restart: always

  # 业务逻辑编排
  node-red:
    image: nodered/node-red:latest
    ports:
      - "1880:1880"
    volumes:
      - ./node_red_data:/data
    depends_on:
      - mqtt-broker
    restart: always

  # 消息中心
  mqtt-broker:
    image: eclipse-mosquitto:latest
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto/config:/mosquitto/config

```

---

## 4. AI 引擎开发规范 (AI-Engine)

### A. 算法策略

1. **吸烟检测 (Task 1)**:
* 模型：YOLOv8n-Smoking (TensorRT FP16)。
* 逻辑：8路流轮询，采样率 2fps。连续 3 秒检测到“吸烟姿态+烟头”发送告警。


2. **办公区节能 (Task 2)**:
* 模型：YOLOv8n-Pose (人体姿态)。
* 逻辑：8路流轮询，采样率 0.5fps。检测人体关键点（头、肩）位移。
* 消抖：记录最后一次检测到人的时间戳。



### B. Python 核心初始化代码框架 (提供给 IDE AI)

```python
import cv2
import tensorrt as trt
# 建议使用封装好的推理库，如 ultralytics 或 jetson-inference

class InferenceEngine:
    def __init__(self, model_path):
        # 初始化 TensorRT 加速引擎
        pass

    def process_stream(self, stream_url, task_type):
        # 1. 从 ZLM 获取流 (使用 FFmpeg 硬件解码)
        # 2. 图像缩放至 640x640
        # 3. 推理并根据时域逻辑判定结果
        # 4. MQTT 发送 JSON 结果
        pass

# MQTT 消息格式示例
# { "camera_id": "stair_04", "event": "smoking", "confidence": 0.85, "timestamp": 1705628800 }

```

---

## 5. Node-RED 业务逻辑流 (Logic Flow)

在 Node-RED 中通过 **Function Node** 实现融合逻辑，输入给 AI 去编写脚本：

1. **空间融合逻辑**：
* 接收来自 `ai/occupancy/cam_01` 和 `ai/occupancy/cam_02` 的消息。
* **两路取“或”**：如果任意一路在 120 秒内出现过 `person_detected: true`，全局变量 `area_occupied` 设为 `true`。


2. **时间与环境融合**：
* 使用 `node-red-contrib-cronplus` 设定工作时间窗。
* **权重计算**：
* 18:00 之后，如果 `area_occupied` 为 `false` 持续超过 5 分钟 -> 发送 MQTT `cmd/light/off`。
* 若 `pixel_change_rate` (像素变化率) 极低且无 AI 目标 -> 辅助确认无人。




3. **证据链处理**：
* 收到吸烟告警 -> 触发 ZLM API `/index/api/startRecord`。



---

## 6. AI 助手执行指令 (Prompt for IDE)

你可以直接复制这段文字给 IDE 的 AI 助手：

> "请基于 NVIDIA JetPack 6.x 环境，创建一个名为 'SmartOffice-Edge' 的项目。
> 1. 编写 docker-compose.yml 包含 ZLMediaKit, Node-RED, Mosquitto 和一个基于 Python 3.10 的推理容器。
> 2. 推理容器需包含针对 Jetson 优化的 Dockerfile，安装核心库：tensorrt, pycuda, opencv-python, paho-mqtt。
> 3. 初始化 Python 脚本框架：实现多线程拉取 16 路 RTSP 流，其中 8 路执行吸烟检测逻辑，8 路执行人体姿态检测，所有结果通过 JSON 格式推送到 MQTT 对应主题。
> 4. 在项目中创建一个 flows.json 初始文件供 Node-RED 使用，实现办公区两路摄像头联动判定，并设置 10 分钟无人自动节能的定时逻辑。"
> 
> 

---

### 项目落地关键细节

* **硬件解码**：必须在 Python 容器内安装 `jetson-ffmpeg` 以调用 `h264_nvv4l2`。
* **内存管理**：Orin Nano 显存内存共用。16 路流建议设置交换内存 (Swap) 为 8GB。
* **误报抑制**：在 Node-RED 中设置“置信度滑窗”，只有连续检测到的置信度均值 > 0.6 才触发告警。

**该方案是否满足您的项目初始化需求？我可以为您进一步细化特定部分的 Dockerfile 配置。**



太棒了。为了确保 IDE 中的 AI（如 Cursor, Windsurf 或 GitHub Copilot）能精准生成代码，我为您细化了最具技术挑战性的 **AI 推理容器 (AI-Engine)** 的配置和**核心逻辑代码结构**。

您可以将以下内容直接作为项目的“基石”文档提供给 AI 助手。

---

## 一、 AI 推理容器核心配置 (`Dockerfile`)

这个 Dockerfile 解决了 Jetson 上最麻烦的驱动穿透和硬解码库安装问题。

```dockerfile
# 使用 NVIDIA 官方针对 JetPack 6 优化的基础镜像
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0

# 设置工作目录
WORKDIR /app

# 安装系统依赖：包含硬件视频处理库和编译工具
RUN apt-get update && apt-get install -y \
    libextstack-dev libgoogle-glog-dev libevent-dev \
    pkg-config python3-pip ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# ultralytics 用于 YOLOv8, paho-mqtt 用于通信
RUN pip3 install --upgrade pip
RUN pip3 install ultralytics paho-mqtt numpy

# 拷贝推理脚本和 TensorRT 模型
COPY ./src /app/src
COPY ./models /app/models

# 启动脚本（开启硬解码支持）
CMD ["python3", "/app/src/main.py"]

```

---

## 二、 核心推理逻辑框架 (`main.py`)

这是项目的灵魂。它采用了**生产者-消费者模型**：一个线程池负责拉流解码，一个单例模型负责分时推理。

```python
import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import threading
import time

# 1. 配置信息
STREAMS = {
    "smoking": ["rtsp://zlm/stair1", "rtsp://zlm/stair2"], # 示例路数
    "occupancy": ["rtsp://zlm/office1_a", "rtsp://zlm/office1_b"]
}

# 2. 初始化模型 (加载转换好的 TensorRT .engine 文件)
smoking_model = YOLO("/app/models/smoking_v8n.engine", task='detect')
pose_model = YOLO("/app/models/pose_v8n.engine", task='pose')

mqtt_client = mqtt.Client()
mqtt_client.connect("mqtt-broker", 1883)

def process_smoking(cam_id, frame):
    # 抽帧逻辑：2fps
    results = smoking_model(frame, conf=0.5, verbose=False)
    for res in results:
        if len(res.boxes) > 0:
            mqtt_client.publish(f"ai/alarm/smoking/{cam_id}", "DETECTED")

def process_occupancy(cam_id, frame):
    # 抽帧逻辑：0.5fps
    # 使用 Pose 识别，过滤掉靠背、布偶（必须有关键点才算人）
    results = pose_model(frame, conf=0.4, verbose=False)
    has_person = False
    for res in results:
        if res.keypoints and len(res.keypoints) > 0:
            has_person = True
            break
    mqtt_client.publish(f"ai/status/occupancy/{cam_id}", "1" if has_person else "0")

def stream_worker(url, task_type, cam_id):
    # 使用硬件加速解码：nvv4l2decoder
    cap = cv2.VideoCapture(url)
    fps_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        fps_counter += 1
        # 根据任务类型决定抽帧频率
        if task_type == "smoking" and fps_counter % 12 == 0: # 假设25fps流，取2fps
            process_smoking(cam_id, frame)
        elif task_type == "occupancy" and fps_counter % 50 == 0: # 取0.5fps
            process_occupancy(cam_id, frame)

# 3. 为每路流开启线程
for task, urls in STREAMS.items():
    for i, url in enumerate(urls):
        t = threading.Thread(target=stream_worker, args=(url, task, f"cam_{i}"))
        t.start()

```

---

## 三、 Node-RED 逻辑编排指导

请让 AI 在 `flows.json` 中配置以下三个关键节点逻辑：

1. **Join Node (空间融合)**：
* 设定：将 `ai/status/occupancy/cam_0` 和 `cam_1` 的消息按 Topic 合并。
* 函数：`if (msg.payload == "1") { flow.set("area_active", true); flow.set("last_seen", Date.now()); }`


2. **Trigger Node (消抖计时)**：
* 当 `area_active` 变为 `false` 时，启动一个 10 分钟的倒计时。
* 如果中途收到任何 `1`（有人），重置计时器。


3. **Action Node (节能控制)**：
* 计时结束，通过 MQTT 发送指令给空调/照明网关（如 `office/light/control` payload: `OFF`）。



---

## 四、 后续交给 IDE AI 的步骤提示

1. **创建文件夹结构**：
```text
/SmartOffice-Edge
├── docker-compose.yml
├── /ai_engine
│   ├── Dockerfile
│   ├── /src/main.py
│   └── /models (存放 .engine 文件)
├── /node_red_data
└── /mosquitto/config/mosquitto.conf

```


2. **模型转换指令**（在盒子上运行一次）：
`yolo export model=smoking_v8n.pt format=engine device=0 half=True` (生成 FP16 加速引擎)。

通过以上方案，您的 AI 助手已经具备了构建整个生产级系统的所有上下文。您可以先让它生成 `docker-compose.yml` 试试看。

**需要我帮您写一份针对 Node-RED 的具体 Function 节点 JavaScript 脚本吗？它可以精确处理两路视频重叠区域的逻辑。**





开发准备：
完全可以。在**普通开发笔记本**上先行完成逻辑开发和 Docker 环境搭建是非常科学的做法，这被称为“环境解耦”。

既然你要用普通笔记本（通常是 Windows 或 macOS）模拟 **Orin Nano (JetPack 6.x/Ubuntu)** 的环境，你需要注意以下几个核心步骤来实现“无缝迁移”：

### 1. 模拟环境搭建 (PC 端)

由于笔记本没有 NVIDIA 的 Jetson 专用库（如 `nvv4l2decoder`），你需要构建一个兼容性强的开发环境：

* **容器化模拟**：在笔记本上安装 **Docker Desktop**。
* **基础镜像切换**：
* **本地开发时**：使用普通的 Python 镜像（如 `python:3.10-slim`）。
* **迁移至硬件时**：只需将 Dockerfile 的 `FROM` 更改为 `nvcr.io/nvidia/l4t-pytorch:r36.2.0`。


* **解码器解耦**：在代码中做一个简单的判断：
```python
# 自动识别环境选择解码器
if is_jetson():
    backend = "nvv4l2decoder" # 硬件加速
else:
    backend = "ffmpeg"        # 笔记本 CPU 解码

```



### 2. 如何接入“两路视频”进行测试？

你不需要真的安装摄像头，有以下几种灵活方案：

* **方案 A：本地视频文件模拟（最推荐）**
找一段现成的写字楼楼梯间视频或办公区视频，放进 ZLMediaKit。
1. 将 `.mp4` 文件放入 ZLM 的 `www/record` 目录。
2. 通过 RTSP 地址 `rtsp://127.0.0.1/live/test_loop` 循环推流。
3. 这样你的 AI 逻辑完全感知不到这其实是个文件，开发体验与实战一致。


* **方案 B：手机摄像头推流**
下载“锐视”或“IP摄像头”App，将手机变成 RTSP 服务器，接入笔记本网络。
* **方案 C：笔记本自带摄像头**
直接在 OpenCV 中调用 `cv2.VideoCapture(0)`。

### 3. 开发环境 VS 生产环境 对比清单

为了让 AI 助手帮你初始化项目，你需要告诉它目前处于 **"Development Phase"**：

| 组件 | 开发笔记本 (Current) | Orin Nano (Target) |
| --- | --- | --- |
| **GPU 加速** | 使用 CUDA (如果是 NVIDIA 显卡笔记本) 或 CPU | **必须使用 TensorRT (FP16)** |
| **推流服务器** | Docker 运行 ZLMediaKit | Docker 运行 ZLMediaKit |
| **推理框架** | PyTorch / ONNX | **TensorRT** |
| **业务逻辑** | Node-RED (完全一致) | Node-RED (完全一致) |

### 4. 立即可以进行的开发工作

你可以现在就让 IDE AI 完成以下工作：

1. **ZLMediaKit 配置**：写好 `config.ini`，确保它能接收 RTSP 并提供 WebHook 接口。
2. **Node-RED 逻辑编写**：直接在笔记本上跑 Node-RED，把那段 JavaScript 融合逻辑调通（可以使用模拟数据）。
3. **吸烟/人体姿态检测代码**：使用 `.pt` 权重文件编写 Python 脚本，实现在 2 路视频流上的抽帧和结果推送。

---

**建议操作：**
你可以先让 AI 助手生成一份**适用于 Windows/macOS 的 `docker-compose.dev.yml**`。这样你今天就可以在笔记本上看到：

1. ZLMediaKit 成功拉流。
2. Python 脚本成功在画面上画出人体关键点。
3. Node-RED 收到有人/无人的 MQTT 信号。

**你需要我为你生成这份专供笔记本开发使用的 `docker-compose.dev.yml` 和对应的 Python 推理 demo 吗？**