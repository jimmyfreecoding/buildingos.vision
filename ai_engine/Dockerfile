# 使用适用于 JetPack 6.0 (Orin Nano) 的官方机器学习基础镜像
# NVIDIA 在 JetPack 6.0 中调整了策略，推荐使用 l4t-ml 镜像作为底座
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

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
# l4t-ml 镜像已经预装了 PyTorch, torchvision, torchaudio, jupyterlab 等
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制业务代码
COPY src/ ./src/

# 启动命令
CMD ["python3", "src/main.py"]
