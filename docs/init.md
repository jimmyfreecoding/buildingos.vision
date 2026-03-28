# NVIDIA Jetson Orin Nano 初始化与部署环境准备指南

本文档详细记录了从拿到一台全新的 NVIDIA Jetson Orin Nano Developer Kit 开始，到将其打造成能够支撑 BuildingOS Vision AI 引擎的生产级边缘计算节点的完整流程。

本指南特别针对 **Windows 用户** 设计，无需复杂的 Linux 交叉编译环境即可完成系统刷机与 SSD 迁移。

---

## 1. 硬件准备

*   **核心板卡**: NVIDIA Jetson Orin Nano Developer Kit (推荐 8GB 版本)
*   **高速存储**: NVMe M.2 SSD (推荐 256GB 以上，PCIe Gen3x4 或更高，用于承载系统、Docker 镜像及视频录像)
*   **引导介质**: microSD 卡 (建议 64GB 以上，UHS-I 速度标准)
*   **操作电脑**: Windows 10/11 电脑 (用于烧录镜像)
*   **其他外设**: USB-C 电源适配器、显示器 (DisplayPort)、键盘、鼠标、网线。

---

## 2. 系统刷机 (MicroSD 引导法)

由于直接向 NVMe SSD 烧录系统对于 Windows 用户较为复杂，我们采用官方推荐的“先 SD 卡引导，后系统克隆”策略。

### 2.1 制作启动 SD 卡
1.  **下载镜像**: 访问 [NVIDIA Jetson 下载中心](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#prepare)，下载适用于 Orin Nano 的 **JetPack 6.x SD Card Image** (`.zip` 格式)。
2.  **安装烧录工具**: 下载并安装 [Balena Etcher](https://etcher.balena.io/)。
3.  **烧录**: 
    *   将 microSD 卡通过读卡器插入 Windows 电脑。
    *   打开 Balena Etcher，选择下载好的镜像文件，选择目标 microSD 卡。
    *   点击 **Flash!** 开始烧录。
    *   *注意：烧录完成后，Windows 可能会弹出多次“需要格式化磁盘”的警告，**请一律点击“取消”**。*

### 2.2 硬件组装与初次开机
1.  **断电操作**: 确保 Jetson 开发板未连接电源。
2.  **安装存储**: 
    *   将 NVMe SSD 插入开发板底部的 M.2 Key M 插槽，并用螺丝固定。
    *   将烧录好的 microSD 卡插入开发板卡槽。
3.  **开机**: 连接显示器、键鼠、网线，最后插入 USB-C 电源。
4.  **初始化设置**: 系统启动后，跟随 Ubuntu 的 `oem-config` 向导完成初始化（接受协议、连接网络、设置时区、**创建用户名和密码**）。

---

## 3. 将系统迁移至 NVMe SSD (性能关键)

为了让 Docker 容器和视频处理达到最佳 IO 性能，必须将操作系统从较慢的 SD 卡迁移到 NVMe SSD。

### 3.1 格式化 SSD
1.  进入 Ubuntu 桌面后，点击左上角搜索并打开 **"Disks" (磁盘)** 工具。
2.  在左侧列表中选择您的 NVMe SSD（通常标识为 `/dev/nvme0n1`）。
3.  点击右上角菜单（三个点/齿轮图标），选择 **"Format Disk..."**，分区方案选择 **GPT**，点击 Format。
4.  点击 SSD 下方的 **"+"** 号创建一个新分区：
    *   容量：默认最大值。
    *   格式：**Ext4**。
    *   名称：可填 `NVMe`。
5.  点击“播放”按钮（三角形图标）挂载该分区，并记下挂载路径（例如 `/media/<您的用户名>/NVMe`）。

### 3.2 克隆系统文件
打开终端 (`Ctrl+Alt+T`)，执行以下命令将 SD 卡内容同步到 SSD：

```bash
# 1. 更新源并安装 rsync 同步工具
sudo apt update
sudo apt install rsync

# 2. 执行系统级克隆
# ⚠️ 注意：请将 /media/jetson/NVMe 替换为您实际的挂载路径
sudo rsync -axHAWX --numeric-ids --info=progress2 --exclude={"/dev/*","/proc/*","/sys/*","/tmp/*","/run/*","/mnt/*","/media/*","/lost+found"} / /media/jetson/NVMe
```

### 3.3 修改启动引导配置
必须告诉 Bootloader 优先从 SSD 加载系统。

1.  编辑 **SD 卡上** 的引导配置文件：
    ```bash
    sudo nano /boot/extlinux/extlinux.conf
    ```
2.  找到包含 `APPEND` 的那一行，将 `root=/dev/mmcblk0p1`（指向 SD 卡）修改为 `root=/dev/nvme0n1p1`（指向 SSD）。
    *修改后示例*：`APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 ...`
3.  按 `Ctrl+O` 保存，`Enter` 确认，`Ctrl+X` 退出。
4.  **重启设备**：
    ```bash
    sudo reboot
    ```
5.  **验证迁移**：重启后打开终端，输入 `df -h /`。如果挂载点显示为 `/dev/nvme0n1p1`，则系统迁移成功。

---

## 4. 边缘计算性能调优

### 4.1 开启 MAXN 满血模式
Jetson 默认可能处于节能模式，需要手动解锁全部 40 TOPS 算力。
*   点击桌面右上角的电源/性能图标（或者使用终端：`sudo nvpmodel -m 0`），选择 **MAXN** 模式。
*   建议同时开启风扇自动控制或拉高风扇转速以应对高负载发热。

### 4.2 配置 Swap (虚拟内存)
由于 Orin Nano 是显存和内存共享架构（Unified Memory），在编译大型 TensorRT 模型或同时处理多路视频时，极易发生 Out-Of-Memory (OOM)。**强烈建议在 NVMe SSD 上分配至少 8GB 的 Swap 空间**。

```bash
# 创建 8GB 大小的 Swap 文件
sudo fallocate -l 8G /swapfile

# 设置正确的权限
sudo chmod 600 /swapfile

# 格式化为 Swap 格式并启用
sudo mkswap /swapfile
sudo swapon /swapfile

# 配置开机自动挂载 Swap
sudo bash -c 'echo "/swapfile none swap sw 0 0" >> /etc/fstab'
```

### 4.3 安装 Jtop 性能监控工具
`jtop` 是 Jetson 开发者必备的监控工具，可直观查看 CPU、GPU、内存使用率及温度。

```bash
sudo apt install python3-pip
sudo pip3 install -U jetson-stats
# 重启服务使之生效
sudo systemctl restart jetson_stats.service
# 运行监控
jtop
```

---

## 5. Docker 容器环境配置

BuildingOS Vision 完全基于容器化部署。JetPack 6.x 通常已预装 Docker，但需要进行特定配置以支持 NVIDIA GPU 穿透。

### 5.1 配置 Docker 用户组 (免 Sudo)
为了方便日常操作和脚本部署，将当前用户加入 docker 组：
```bash
sudo usermod -aG docker $USER
newgrp docker
# 测试是否成功（无需 sudo 即可运行）
docker ps
```

### 5.2 设置 NVIDIA Container Runtime 为默认
这是极其关键的一步。如果不设置，Docker 容器默认只能使用 CPU 进行计算，无法调用 Orin Nano 的 GPU 和 Tensor Core。

1.  编辑 Docker 的守护进程配置文件：
    ```bash
    sudo nano /etc/docker/daemon.json
    ```
2.  确保文件内容如下（如果文件为空，直接粘贴）：
    ```json
    {
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        },
        "default-runtime": "nvidia"
    }
    ```
3.  保存退出后，重启 Docker 服务：
    ```bash
    sudo systemctl restart docker
    ```

### 5.3 安装 Docker Compose 插件
由于 Ubuntu 默认源可能未包含最新的 Docker 插件源，请使用以下官方方式安装：

```bash
# 0. 安装 curl 工具 (如果提示 curl: command not found)
sudo apt update
sudo apt install curl -y

# 1. 下载 Docker Compose 二进制文件到插件目录
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-aarch64 -o $DOCKER_CONFIG/cli-plugins/docker-compose

# 2. 赋予执行权限
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# 3. 验证安装 (应该输出版本号)
docker compose version
```

---

至此，NVIDIA Jetson Orin Nano 的底层硬件、操作系统、性能优化及容器化环境已全部准备就绪。

### 5.4 离线环境辅助工具预装 (可选但强烈推荐)
由于设备最终将在客户内网（无互联网或受限网络）运行，虽然核心业务都在 Docker 内，但宿主机预装以下工具会在后期现场调试时帮大忙。

**1. 网络抓包与调试工具**
用于现场排查 RTSP 流拉取失败或 MQTT 连不上的问题：
```bash
sudo apt update
sudo apt install -y tcpdump nmap net-tools iputils-ping iperf3 curl wget
```

**2. 文本与系统处理工具**
用于现场快速修改配置或排查进程：
```bash
sudo apt install -y vim nano htop iotop tmux tree jq zip unzip
```

**3. Python 3 (轻量级预装)**
Ubuntu 22.04 默认自带 Python 3，但建议补齐基础依赖。这主要用于现场可能需要写个临时脚本测试网络或发个 MQTT 消息，而不是用来跑业务代码。
```bash
sudo apt install -y python3-pip python3-venv
# 预装两个最常用的现场测试库
pip3 install paho-mqtt requests
```

*(注意：Node.js **不需要**在宿主机安装，因为我们已经有 Node-RED 的 Docker 容器。如果在宿主机安装反而容易引起端口冲突或版本混乱。)*

---

## 6. 制作“黄金母盘” (Golden Image) 批量部署指南

为了避免在后续部署多台设备时重复上述繁琐的环境配置，强烈建议在此时制作一个“黄金母盘”。其他实施人员拿到该镜像后，直接刷入 SSD 即可开始跑业务代码。

### 6.1 镜像制作前的“大扫除” (瘦身)
在 Jetson 上执行以下命令清理缓存，缩小最终镜像的体积：
```bash
# 清理 apt 缓存
sudo apt-get clean
# 清理 Docker 悬空数据
docker system prune -f
# 清理命令历史（保护隐私）
history -c && history -w
```

### 6.2 制作母盘镜像 (在宿主机 PC 上操作)
为了保证文件系统完整性，建议将配置好的 NVMe SSD 拆下，插入一台运行 Ubuntu 的宿主机电脑进行克隆：

1. **识别磁盘：** 插入 SSD 后，输入 `lsblk` 确认设备名（假设为 `/dev/nvme0n1` 或 `/dev/sdb`）。
2. **全盘克隆压缩：**
   ```bash
   # 使用 dd 命令备份并压缩，bs=64K 兼顾速度和稳定性
   sudo dd if=/dev/nvme0n1 conv=sync,noerror bs=64K status=progress | gzip -c > orin_nano_base_v1.0.img.gz
   ```
   *生成的 `.img.gz` 文件即为黄金母盘。*

### 6.3 实施人员如何还原母盘
实施同事拿到 `orin_nano_base_v1.0.img.gz` 后，可通过以下方式还原到新的 SSD：

**方法 A：图形化工具 (推荐)**
在 Windows/Mac/Linux 上使用 **BalenaEtcher**，选择该压缩包，目标选新的 SSD，点击 Flash 即可。

**方法 B：命令行还原 (Ubuntu)**
```bash
gunzip -c orin_nano_base_v1.0.img.gz | sudo dd of=/dev/nvme0n1 bs=64K status=progress
```

### 6.4 实施还原后的注意事项 ⚠️
由于是从活体系统克隆，实施人员将新 SSD 插入新 Jetson 开机后，需注意：

1. **磁盘空间扩展：** 如果新 SSD 容量大于母盘 SSD，剩余空间将处于未分配状态。需执行以下命令扩容：
   ```bash
   # 假设主分区是 /dev/nvme0n1p1
   sudo resize2fs /dev/nvme0n1p1
   ```
2. **硬件版本兼容性：** 如果新 Jetson 板卡的底层 QSPI 固件过旧导致无法从 NVMe 启动，需先使用官方 SDK Manager 或 SD 卡对该板卡进行一次基础的 JetPack 引导刷写。

*(至此，基础环境彻底搭建完成，可以开始拉取 `buildingos.vision` 项目代码并进行业务服务部署了。)*