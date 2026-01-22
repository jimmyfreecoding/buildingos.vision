# 专业级吸烟检测方案与模型训练指南

本文档详细介绍了 AI Engine 中采用的 **两阶段级联（Two-Stage Cascade）** 吸烟检测算法原理，并提供了从 Roboflow 下载数据集到在本地/容器内训练模型的完整操作流程。

## 1. 算法架构原理

为了在写字楼楼梯间等复杂场景下实现高精度、低误报的吸烟检测，我们摒弃了传统的全图单级检测方案，采用了更符合人类认知逻辑的 **Pose + ROI Specialist** 级联方案。

### 1.1 核心流程
1.  **第一阶段：姿态初筛 (Pose Proposal)**
    *   **模型**: `yolov8n-pose`
    *   **逻辑**: 对全图进行人体姿态估计，实时计算手腕关键点（Left/Right Wrist）与面部关键点（Nose/Ear）的欧氏距离。
    *   **触发条件**: 当 `Distance(Wrist, Face) < Threshold`（手腕靠近嘴边）持续一定帧数时，判定为“疑似吸烟动作”。
    *   **输出**: 裁剪出该人员的上半身区域（Head + Shoulders + Hands）作为 ROI (Region of Interest)。

2.  **第二阶段：精细化确认 (Specialist Verification)**
    *   **模型**: `smoking_specialist.pt` (基于 YOLOv8n 训练的专用分类模型)
    *   **逻辑**: 将第一阶段裁剪出的 ROI 图片送入该模型进行推理。
    *   **目标类别**: 重点检测 `Cigarette` (香烟) 和 `Smoke` (烟雾)。
    *   **判定**: 只有当专用模型在 ROI 区域内明确检测到香烟或烟雾时，才最终确认为吸烟事件并触发告警。

### 1.2 方案优势
*   **抗干扰强**: 有效过滤打电话、喝水、挠头等“手部靠近脸部”的非吸烟行为（因为这些行为检测不到香烟/烟雾）。
*   **算力高效**: 只有在有人做特定动作时才启动第二阶段推理，且第二阶段只处理小图（ROI），极大节省了 Orin Nano 的推理资源。

---

## 2. 模型下载与训练全流程

为了获得第二阶段的 `smoking_specialist.pt` 权重文件，我们需要使用高质量的开源数据集进行训练。推荐使用 Roboflow 上的 `smoking-detection-3gefl` 数据集。

### 2.1 步骤一：下载数据集

我们已准备好自动化脚本。请在开发环境（本地或容器）中执行以下操作：

1.  **准备环境**:
    确保已安装 `roboflow` 库：
    ```bash
    pip install roboflow
    ```

2.  **执行下载脚本**:
    使用项目根目录下的 `download_model_local.py`（或在容器内使用 `/app/src/download_model.py`）。
    
    ```python
    # 脚本核心逻辑
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY") # 替换为您的 Key
    project = rf.workspace("kyunghee-university-ada5d").project("smoking-detection-3gefl")
    dataset = project.version(4).download("yolov8")
    ```

    **执行命令**:
    ```powershell
    python download_model_local.py
    ```
    *数据集将下载至 `ai_engine/models/dataset/Smoking-Detection-4` 目录。*

### 2.2 步骤二：训练模型

由于下载的只是图片和标签，我们需要使用 YOLOv8 框架训练出 `.pt` 权重文件。您可以直接利用 `ai-engine` 容器的环境进行训练。

**训练命令 (在宿主机执行)**:

```powershell
docker-compose -f buildingos.vision.dev.yml exec ai-engine yolo detect train \
    data=/app/models/dataset/Smoking-Detection-4/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    project=/app/models/train \
    name=smoking_run
```

**参数说明**:
*   `data=...`: 指向刚下载的数据集配置文件。
*   `model=yolov8n.pt`: 使用 YOLOv8 Nano 预训练权重作为起点（迁移学习）。
*   `epochs=50`: 训练 50 轮（通常 30-50 轮即可达到不错效果）。
*   `project=...`: 训练结果保存路径。

### 2.3 训练中断与恢复 (Resume Training)

如果训练过程中因断电或程序崩溃而中断，您不需要重新开始。YOLO 会自动保存 `last.pt`，您可以从中断点继续训练。

**恢复训练命令**:
```powershell
docker-compose -f buildingos.vision.dev.yml exec ai-engine yolo detect train \
    resume \
    model=/app/models/train/smoking_run/weights/last.pt
```
*注意：必须指定 `last.pt` 路径，并加上 `resume` 关键字。*

### 2.4 训练结果解读

训练完成后，控制台会输出最终的评估指标。关键指标解释如下：

*   **Epochs**: 训练轮数（例如 50/50）。
*   **mAP50**: 平均精度均值（IoU=0.5）。**这是最重要的指标**。
    *   `> 0.8`: 优秀，模型非常可靠。
    *   `0.6 - 0.8`: 良好，可以使用。
    *   `< 0.5`: 较差，可能需要更多数据或调整参数。
*   **mAP50-95**: 更严格的精度指标。通常比 mAP50 低，0.4-0.5 已经算不错。
*   **Speed**: 推理速度（Inference Time），越低越好。

**示例输出**:
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all        753       1144      0.824      0.769      0.832      0.488
```
*解读：mAP50 为 83.2%，说明模型性能优秀。*

### 2.5 步骤三：部署模型

训练完成后，新的权重文件会生成在容器内的 `/app/models/train/smoking_run/weights/best.pt`。

1.  **移动并重命名**:
    ```powershell
    # 进入容器操作
    docker-compose -f buildingos.vision.dev.yml exec ai-engine sh -c "mv /app/models/train/smoking_run/weights/best.pt /app/models/smoking_specialist.pt"
    ```

2.  **重启服务**:
    AI Engine 会自动加载新的 `smoking_specialist.pt`。
    ```powershell
    docker-compose -f buildingos.vision.dev.yml restart ai-engine
    ```

### 2.6 验证

1.  查看日志确认模型加载成功：
    ```bash
    docker logs buildingosvision-ai-engine-1
    # 应显示: Loading PyTorch model: /app/models/smoking_specialist.pt
    ```
2.  在摄像头前模拟吸烟动作（手持笔状物靠近嘴边），观察 Node-RED 或 MQTT 是否有 `SMOKING_DETECTED` 告警，且图片中应包含 **ROI Checked (黄框)** 和 **CONFIRMED (红框)** 标注。

---

## 3. 常见问题

*   **Q: 为什么下载后找不到 `.pt` 文件？**
    A: Roboflow 的下载接口默认只提供数据集（Images/Labels），不包含训练好的权重。必须执行步骤 2.2 进行训练。
*   **Q: 容器不断重启？**
    A: 检查 `/app/models/smoking_specialist.pt` 是否损坏。如果训练失败或文件错误，请删除该文件，系统会自动回退到使用 `yolov8n.pt` 作为占位符。
