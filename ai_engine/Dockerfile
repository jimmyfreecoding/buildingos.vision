# 使用 NVIDIA 官方针对 JetPack 6 优化的基础镜像
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0

# 设置工作目录
WORKDIR /app

# 安装系统依赖：包含硬件视频处理库和编译工具
# 注意：l4t-pytorch 镜像通常基于 Ubuntu
RUN apt-get update && apt-get install -y \
    libextstack-dev libgoogle-glog-dev libevent-dev \
    pkg-config python3-pip ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# 拷贝推理脚本和 TensorRT 模型
COPY ./src /app/src
COPY ./models /app/models

# 启动脚本（开启硬解码支持）
CMD ["python3", "/app/src/main.py"]
