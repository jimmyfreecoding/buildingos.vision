# 使用可配置的 JetPack 6.x 官方基础镜像
# 生产切换到 l4t-jetpack:r36.4.0，以对齐宿主机 TensorRT 10.x
ARG L4T_ML_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${L4T_ML_IMAGE}

# 环境变量设置
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/ai_engine/config/config.json

# 更新系统依赖，安装那些在 ARM 上 pip 编译极慢或容易报错的系统级依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-pandas \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 Python 包
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision && \
    pip3 install --no-cache-dir "numpy<2.0.0" paho-mqtt && \
    pip3 install --no-cache-dir ultralytics --no-deps && \
    pip3 install --no-cache-dir psutil pyyaml tqdm matplotlib opencv-python "numpy<2.0.0" onnxslim colorama sympy polars ultralytics-thop

# 复制业务代码
COPY src/ ./src/

# 启动命令
CMD ["python3", "src/main.py"]
