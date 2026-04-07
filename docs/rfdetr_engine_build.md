# RF-DETR Engine 编译落地

## 1) 目标

完成第一阶段：产出可在 Jetson 上部署的 RF-DETR TensorRT Engine。

默认目标文件：

- ONNX：`/app/models/rf-detr.onnx`
- Engine：`/app/models/rf-detr-fp16.engine`

## 2) 前置条件

- Jetson 已安装 JetPack 6.x，且容器内可用 `trtexec`
- `docker compose` 中 `ai-engine` 已挂载：
  - `./ai_engine/models:/app/models`
  - `./scripts:/app/scripts:ro`

## 3) 获取 ONNX（开发机执行）

仓库已提供导出脚本：

- [export-rfdetr-onnx.py](file:///c:/project/buildingos.vision/scripts/export-rfdetr-onnx.py)

安装依赖：

```bash
pip install "rfdetr[onnx]"
```

导出 Medium 版本（推荐）：

```bash
python scripts/export-rfdetr-onnx.py --variant medium --size 640x640 --conf 0.25 --output ai_engine/models/rf-detr.onnx
```

说明：

- 权重会由 `rfdetr` 自动下载并缓存，不需要手动找 `.pt`
- RF-DETR 导出尺寸必须能被 32 整除，按你的场景推荐 `640x640`

## 4) 编译 Engine（Jetson 上执行）

仓库已提供编译脚本：

- [build-rfdetr-engine.sh](file:///c:/project/buildingos.vision/scripts/build-rfdetr-engine.sh)

标准 FP16 编译：

```bash
docker compose exec ai-engine bash -lc "chmod +x /app/scripts/build-rfdetr-engine.sh && /app/scripts/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-fp16.engine fp16"
```

默认 shape 已按 RF-DETR 设置为：

- `MIN_SHAPE=1x3x640x640`
- `OPT_SHAPE=1x3x640x640`
- `MAX_SHAPE=4x3x640x640`

## 5) 可选：自定义 shape / INT8

自定义动态 shape：

```bash
docker compose exec ai-engine bash -lc "INPUT_NAME=images MIN_SHAPE=1x3x640x640 OPT_SHAPE=2x3x640x640 MAX_SHAPE=4x3x640x640 /app/scripts/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-fp16.engine fp16"
```

INT8 编译：

```bash
docker compose exec ai-engine bash -lc "CALIB_CACHE=/app/models/rf-detr-int8.cache /app/scripts/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-int8.engine int8"
```

## 6) 验证与排错

验证产物：

```bash
docker compose exec ai-engine bash -lc "ls -lh /app/models/rf-detr*.engine"
```

常见失败项：

- `trtexec not found`
- 输入名不是 `images`（用 `INPUT_NAME` 覆盖）
- ONNX 算子不被当前 TensorRT 支持（需调整导出版本或图）
