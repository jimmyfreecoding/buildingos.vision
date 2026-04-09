# 边缘网关 CI/CD 与 OTA 升级方案（ai-engine 宿主机化）

本文档定义新的生产部署策略：**仅 `ai-engine` 从 Docker 中迁出，改为宿主机常驻运行；其余服务继续 Docker 化运行**。  
目标是降低 Jetson 端构建复杂度、缩短故障恢复路径，并保持 OTA 能力。

---

## 1. 架构基线：混合部署（Host + Docker）

### 1.1 服务边界

- **宿主机运行 (systemd 托管)**
  - `ai-engine` (核心推理引擎)
  - `jtop-daemon` (硬件状态采集)
  - `llama-gemma` (Gemma 4 复核服务)
- **Docker 运行 (保持不变)**
  - `zlm`
  - `web-nginx`
  - `web-manager-backend`
  - `web-manager-frontend-deploy`
  - `dockge`
- **Docker 运行（仅回滚时启用）**
  - `ai-engine`（`legacy-ai-engine` profile）

### 1.2 这样做的原因

- 避免 `ai-engine` 在 Jetson 上反复 `docker build` 带来的高耗时、高磁盘占用与依赖不确定性。
- 避免容器内 TensorRT/PyCUDA/动态库耦合导致的构建失败或运行时崩溃。
- 保留其余服务容器化，继续享受网络隔离、统一编排和运维便利。

---

## 2. ai-engine 宿主机环境要求（详细）

### 2.1 操作系统与硬件

- Jetson Orin Nano 8GB（生产目标机型）
- JetPack 6.x（建议与当前量产机保持一致）
- 可用磁盘空间建议：
  - 系统和依赖预留：>= 10GB
  - 模型和日志预留：>= 20GB

### 2.2 运行时与推理栈

- Python 3.10（建议）
- **核心运行时依赖：**
  - `ffmpeg` (宿主机必须安装，用于 RTSP 抓拍采样)
  - `tensorrt` (由 JetPack 系统提供，需通过 `--system-site-packages` 穿透进入 venv)
  - `pycuda` (GPU 内存管理，通过 requirements.txt 安装)
- TensorRT 运行时与 `.engine` 序列化版本必须一致
  - 例如：运行时 `Current Version: 239` 时，`engine` 也必须是 `239` 生成
- CUDA 驱动由 JetPack 提供，不在项目内重复安装

### 2.3 Python 依赖策略

- 必须使用**项目独立虚拟环境**（venv）
- 依赖由 `ai_engine/requirements.txt` 管理
- 不允许在系统全局 Python 做长期运行依赖安装

### 2.4 目录与权限约定

- 项目根目录：`~/buildingos.vision`
- ai-engine 工作目录：`~/buildingos.vision/ai_engine`
- 配置文件：`~/buildingos.vision/ai_engine/config/config.json`
- 模型目录：`~/buildingos.vision/ai_engine/models`
- 运行账号需对上述目录有读写权限

---

## 3. “系统污染”定义、场景与规避

### 3.1 系统污染是什么

- 不是“中毒”，而是**宿主机运行环境被长期改写**，导致后续不可预期。

### 3.2 常见污染场景

- 同一台机上 `pip install -U` 把全局包升级，旧代码突然跑不起来。
- A 项目升级 `numpy/torch`，B 项目跟着崩（共用全局 `site-packages`）。
- `apt upgrade` 后系统库变动，某些 Python/CUDA 扩展二进制不兼容。
- 临时装的包忘了记录，半年后没人知道“当时为什么能跑”。
- 手工改环境变量（`PATH`/`LD_LIBRARY_PATH`）后，服务重启顺序一变就失效。

### 3.3 规避策略（必须执行）

- 永远用独立 `venv`。
- 用 `requirements.txt` 锁定运行依赖版本。
- `systemd` 的 `ExecStart` 只指向该 `venv` 的 Python。
- 将运行时环境变量写入 `systemd` unit，不依赖手工 `export`。

---

## 4. OTA 链路改造（从容器更新到宿主服务更新）

### 4.1 关键变化

- 以前：`docker compose pull/up -d ai-engine`
- 现在：`git pull` 后执行 `systemctl restart ai-engine`

这不是 OTA 失效，而是从“容器编排更新”变成“代码更新 + 宿主服务重启”。

### 4.2 Web Manager 的执行分支

`web-manager-backend` 目前通过 `docker.sock` 控制容器。迁移后应保留两条分支：

1. **Docker 分支（其他服务）**：继续使用 `docker compose ...`
2. **Host 分支（ai-engine）**：执行宿主机 `systemctl` 命令

推荐更新命令序列：

```bash
cd /host_project && \
git pull && \
sudo systemctl restart ai-engine && \
sudo systemctl status ai-engine --no-pager -n 50
```

---

## 5. systemd 稳定性说明与基线配置

### 5.1 systemd 脆弱吗

- 不脆弱，反而是 Linux 生产常规方案。
- 真正脆弱的是 unit 文件写得随意。

### 5.2 稳定运行最小配置

- `Restart=always`
- `RestartSec=3`
- `WorkingDirectory` 固定到项目目录
- `ExecStart` 固定到 `venv` 的 Python
- 明确 `Environment=CONFIG_PATH=...`
- 使用 journald 收敛日志并配合日志轮转

### 5.3 参考 unit 文件

```ini
[Unit]
Description=BuildingOS AI Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=buildingos
Group=buildingos
WorkingDirectory=/home/buildingos/buildingos.vision/ai_engine
Environment=CONFIG_PATH=/home/buildingos/buildingos.vision/ai_engine/config/config.json
ExecStart=/home/buildingos/buildingos.vision/ai_engine/.venv/bin/python3 /home/buildingos/buildingos.vision/ai_engine/src/main.py
Restart=always
RestartSec=3
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

---

## 6. 部署与运维流程（宿主机版）

### 6.1 首次部署

```bash
cd ~/buildingos.vision/ai_engine
# 1. 彻底删除旧 venv (如有)
rm -rf .venv
# 2. 重新创建 venv 并开启系统包穿透
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# 3. 设置 CUDA 编译环境 (关键: 防止 pycuda 编译失败)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# 4. 安装 ffmpeg 及业务依赖
sudo apt-get update && sudo apt-get install -y ffmpeg
pip install -U pip
pip install -r requirements.txt
# 5. 手动安装 pycuda (确保环境变量生效)
pip install pycuda==2024.1.2

# 6. 修复共享目录权限 (核心: 确保宿主 ai-engine 有权写入 ZLM 日志目录)
mkdir -p ~/buildingos.vision/zlm/www/occupancy_logs
sudo chown -R buildingos:buildingos ~/buildingos.vision/zlm/www/occupancy_logs
sudo chmod -R 775 ~/buildingos.vision/zlm/www/occupancy_logs

# 7. 设置每日自动清理、总结与重启定时任务 (基础部署要求)
chmod +x ~/buildingos.vision/ai_engine/src/setup_cron.sh
~/buildingos.vision/ai_engine/src/setup_cron.sh
```

### 6.2 安装并启动 systemd 服务

```bash
sudo cp /home/buildingos/buildingos.vision/deploy/ai-engine.service /etc/systemd/system/ai-engine.service
sudo systemctl daemon-reload
sudo systemctl enable ai-engine
sudo systemctl start ai-engine
```

### 6.3 日常操作

```bash
sudo systemctl status ai-engine --no-pager -n 80
sudo systemctl restart ai-engine
journalctl -u ai-engine -f
```

---

## 7. 与 Docker 网络互通注意事项

- `ai-engine` 在宿主机运行后，访问宿主本机服务可直接 `127.0.0.1`。
- 访问 Docker 内服务时，使用对外映射端口（例如宿主机 `10081` 对应 `zlm:80`）。
- **代码自适应逻辑**：代码中已内置 `get_real_url` 等助手函数，会自动处理：
  - `zlm:80` -> `127.0.0.1:10081`
  - `host.docker.internal` -> `127.0.0.1`
- 容器访问宿主的 `host.docker.internal` 规则继续保留给容器侧使用，不影响宿主侧。

---

## 8. 风险与回滚策略

### 8.1 风险
- 运维脚本若仍只会操作容器，可能出现“代码已更新但 ai-engine 未重启”。
- 若跳过 `venv` 规范，系统污染风险会重新出现。

### 8.2 回滚
如需临时回滚到容器版：
1. `sudo systemctl stop ai-engine`
2. `sudo systemctl disable ai-engine`
3. `docker compose -f docker-compose.yml --profile legacy-ai-engine up -d ai-engine`

---

## 9. 自动化维护与 AI 报告

为保证边缘网关长期稳定运行并回收存储空间，系统内置了自动化维护机制。

### 9.1 定时任务 (Cron Job)
通过 `setup_cron.sh` 配置，每天凌晨 **02:00** 自动执行以下操作：
1. **图片压缩**：将昨日生成的 `.jpg` 转换为 **WebP (50% 质量)**，并将分辨率限制在 800px 宽，可节省约 70% 磁盘空间。
2. **AI 日报生成**：本地 Python 统计全天区域占用时长与复核成功率，并调用 Gemma 生成 Markdown 格式的深度总结。
3. **系统重启**：清理内存碎片、重置底层驱动状态，确保系统每日以“零负载”状态开始新的一天。

### 9.2 日志与报告路径
- **维护日志**：`/home/buildingos/buildingos.vision/ai_engine/cleanup.log`
- **AI 总结报告**：`/home/buildingos/buildingos.vision/zlm/www/occupancy_logs/YYYY-MM-DD/daily_summary.json`

---

## 10. 一键操作指令集 (One-click Commands)

为提高运维效率，下述指令封装了混合架构下的核心操作。

### 10.1 一键全新部署 (Fresh Deployment)
适用于新设备初始化或环境重置。

```bash
cd ~/buildingos.vision && \
# 1. 彻底清理环境并拉取最新代码
deactivate 2>/dev/null || true && \
rm -rf ai_engine/.venv && \
git reset --hard HEAD && \
git pull origin main && \
# 2. 启动 Docker 服务 (不含 ai-engine)
docker compose -f docker-compose.yml up -d --build && \
# 3. 准备 ai-engine 宿主环境 (开启系统包穿透)
cd ai_engine && \
sudo apt-get update && sudo apt-get install -y ffmpeg && \
python3 -m venv --system-site-packages .venv && \
export PATH=/usr/local/cuda/bin:$$PATH && \
export CUDA_ROOT=/usr/local/cuda && \
export CPATH=/usr/local/cuda/include:$$CPATH && \
export LIBRARY_PATH=/usr/local/cuda/lib64:$$LIBRARY_PATH && \
.venv/bin/pip install -U pip && \
.venv/bin/pip install -r requirements.txt && \
.venv/bin/pip install pycuda==2024.1.2 && \
# 4. 修复共享目录权限 (提前创建目录以防 chown 失败)
mkdir -p ~/buildingos.vision/zlm/www/occupancy_logs && \
sudo chown -R buildingos:buildingos ~/buildingos.vision/zlm/www/occupancy_logs && \
sudo chmod -R 775 ~/buildingos.vision/zlm/www/occupancy_logs && \
# 5. 安装并启动 systemd 服务
sudo cp ~/buildingos.vision/deploy/ai-engine.service /etc/systemd/system/ai-engine.service && \
sudo systemctl daemon-reload && \
sudo systemctl enable ai-engine && \
sudo systemctl start ai-engine && \
# 6. 设置自动化维护定时任务 (每日 02:00 清理并重启)
chmod +x ~/buildingos.vision/ai_engine/src/setup_cron.sh && \
~/buildingos.vision/ai_engine/src/setup_cron.sh && \
sudo systemctl status ai-engine --no-pager -n 20
```

### 10.2 一键平滑升级 (OTA Update)
适用于日常代码更新。
```bash
cd ~/buildingos.vision && \
git pull && \
# 1. 更新 Docker 容器 (如有配置/代码变更)
docker compose -f docker-compose.yml up -d --build && \
# 2. 重启宿主 ai-engine 服务
sudo systemctl daemon-reload && \
sudo systemctl restart ai-engine && \
sudo systemctl status ai-engine --no-pager -n 20
```

### 10.3 一键资源清理 (System Cleanup)
回收反复构建产生的磁盘垃圾（建议每季度执行一次）。
```bash
cd ~/buildingos.vision && \
# 1. 移除孤儿容器与过时镜像
docker compose down --remove-orphans && \
docker image rm $(docker images -q buildingos-vision-ai-engine) 2>/dev/null || true && \
# 2. 深度清理构建缓存 (关键: 回收 BuildKit 占用)
docker builder prune -a -f && \
# 3. 清理未使用的卷
docker volume prune -f && \
# 4. 重新拉起服务
docker compose up -d
```

### 10.4 辅助服务一键启停 (Supporting Services)
管理硬件监控与 AI 复核服务。

**一键启动所有辅助服务：**
```bash
sudo systemctl start jtop-daemon llama-gemma && \
sudo systemctl status jtop-daemon llama-gemma --no-pager
```

**一键停止所有辅助服务：**
```bash
sudo systemctl stop jtop-daemon llama-gemma && \
sudo systemctl status jtop-daemon llama-gemma --no-pager
```

**辅助服务安装/重置（含开机自启）：**
```bash
# 复制服务文件
sudo cp ~/buildingos.vision/deploy/jtop-daemon.service /etc/systemd/system/ && \
sudo cp ~/buildingos.vision/deploy/llama-gemma.service /etc/systemd/system/ && \
# 重新加载并设置开机自启
sudo systemctl daemon-reload && \
sudo systemctl enable jtop-daemon llama-gemma && \
sudo systemctl restart jtop-daemon llama-gemma
```

---

## 11. 推荐结论

- 在当前 Jetson 边缘部署条件下，`ai-engine` 宿主机化是优先方案。
- 该方案前提是：`venv` 隔离、`systemd` 托管、OTA 流程切换到 `git pull + systemctl restart ai-engine`。
- 其余服务保持 Docker，不做架构扰动。
