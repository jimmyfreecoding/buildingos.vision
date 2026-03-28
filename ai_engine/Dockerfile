# 使用适用于 JetPack 6.0 (Orin Nano) 的官方机器学习基础镜像
# NVIDIA 在 JetPack 6.0 中调整了策略，推荐使用 l4t-ml 镜像作为底座
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# 环境变量设置
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 更新系统依赖，安装那些在 ARM 上 pip 编译极慢或容易报错的系统级依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    python3-pandas \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 Python 包
COPY requirements.txt .

# 关键：由于 l4t-ml 镜像已经包含了完美适配 Jetson GPU 的 torch 和 torchvision，
# 如果直接 pip install ultralytics，它可能会尝试去下载 x86/标准版的 torch 覆盖掉它。
# 所以我们只更新 pip，并安装除了 torch 以外的依赖，或者让 ultralytics 忽略 torch 依赖
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "numpy<2.0.0" paho-mqtt && \
    pip3 install --no-cache-dir ultralytics --no-deps && \
    pip3 install --no-cache-dir psutil pyyaml tqdm matplotlib opencv-python "numpy<2.0.0" onnxslim colorama sympy polars ultralytics-thop

# 复制业务代码
COPY src/ ./src/

# 启动命令
CMD ["python3", "src/main.py"]
