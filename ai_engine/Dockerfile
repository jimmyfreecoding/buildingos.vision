# 使用适用于 JetPack 6.0 (Orin Nano) 的官方 PyTorch 镜像
# 注意：NVIDIA 官方镜像 tag 必须带上 pth 版本后缀，否则会报 not found
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3

# 环境变量设置
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 更新系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 Python 包
# (镜像自带了 PyTorch, torchvision, torchaudio，这里安装其他依赖)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制业务代码
COPY src/ ./src/

# 启动命令
CMD ["python3", "src/main.py"]
