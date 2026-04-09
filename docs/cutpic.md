# 采样抓拍方案演进文档 (Snapshot Strategy Evolution)

本文档详细说明了本项目从 **OpenCV 截图排队缓存方案** 演进到 **宿主机 FFmpeg 无状态采样方案** 的过程，并对比了三种备选方案的优劣。

---

## 1. 方案演进背景

在初期版本中，项目尝试使用 `cv2.VideoCapture` 进行流式抓取。由于 OpenCV 内部默认维护了一个视频帧缓冲队列（GStreamer/FFmpeg 驱动层），在低采样率（如 60 秒抓一次）的场景下，OpenCV 极易返回缓冲区中数分钟前的“旧帧”，且并发调用多路 RTSP 流时极易触发 Jetson 底层的显存竞争（Double Free）导致容器崩溃。

为了解决该问题，我们经历了从“OpenCV 加锁排队”到“宿主机 FFmpeg 独立采样”的架构转变。

---

## 2. 三种抓拍方案详细对比

| 特性 | OpenCV (`cv2.VideoCapture`) | ZLM API (`getSnap`) | **宿主机 FFmpeg (当前方案)** |
| :--- | :--- | :--- | :--- |
| **传输协议** | 默认 UDP (易花屏/丢包) | HTTP 内部获取 | **强制 TCP** (`-rtsp_transport tcp`) |
| **时效性** | **差** (存在内部缓冲区，易拿旧帧) | 好 (由 ZLM 转发层提供) | **极好** (无状态调用，实时抓取) |
| **稳定性** | **差** (易触发显存泄漏或崩溃) | 中 (受 ZLM 服务稳定性影响) | **极高** (独立进程运行，进程退出即释放资源) |
| **资源消耗** | 高 (需维持长连接/解码器) | 低 (API 调用) | **极低** (仅在采样瞬间启动进程) |
| **状态管理** | 复杂 (需管理对象生命周期) | 简单 (无状态) | **无状态** (无缓存、无排队) |
| **并发处理** | 难 (受 GStreamer 锁限制) | 容易 (HTTP 并发) | **容易** (受操作系统进程调度管理) |

---

## 3. 为什么选择“宿主机 FFmpeg”方案？

在 Jetson 边缘计算环境下，**稳定性**是第一优先级。我们最终选择了在宿主机直接调用 `ffmpeg` 进程的方案，原因如下：

1.  **彻底消除缓冲区干扰 (Zero Latency)**：
    `ffmpeg -i ... -frames:v 1` 命令每次执行都会发起一个新的无状态连接，直接获取流中的当前关键点，不存在任何历史帧缓存问题。

2.  **强制 TCP 传输 (Reliability)**：
    通过 `-rtsp_transport tcp` 参数，强制要求视频流通过可靠传输协议获取，解决了边缘网络下 UDP 采样导致的大面积花屏和解码错误。

3.  **进程级资源隔离 (Isolation)**：
    每次抓拍都是一个独立的 OS 进程。即便某次抓拍因为网络原因超时或崩溃，也不会影响 Python 主进程或其他摄像头的采样线程，进程结束后所有显存/内存立即由操作系统回收。

4.  **环境兼容性 (Jetson Optimization)**：
    宿主机安装的 `ffmpeg` 可以直接调用 Jetson 的硬件解码能力（如 `h264_nvv4l2`），相比容器内复杂的驱动穿透，宿主机直跑更加简洁可靠。

---

## 4. 代码清理确认

当前 [main.py](file:///c:/project/buildingos.vision/ai_engine/src/main.py) 已完成以下清理工作：
- [x] **移除 OpenCV 全局锁**：不再需要 `cv2_open_lock` 来防止并发崩溃。
- [x] **移除 Capture 队列**：不再使用 `Queue` 或 `List` 存储历史帧。
- [x] **移除长连接对象**：不再在线程中持有 `cv2.VideoCapture` 句柄，改为按需调用 `get_frame_from_host_ffmpeg`。
- [x] **实现错峰采样**：通过随机延迟 `stagger_delay` 避免多个摄像头同时启动 `ffmpeg` 进程导致 CPU 峰值。

---

## 5. 部署要求

为保证该方案正常运行，必须在宿主机安装 `ffmpeg`。具体步骤请参考 [cicd.md](file:///c:/project/buildingos.vision/docs/cicd.md)。

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```
