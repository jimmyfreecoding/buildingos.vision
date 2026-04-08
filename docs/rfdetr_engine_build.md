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
- 主容器切换为 `nvcr.io/nvidia/l4t-jetpack:r36.4.0`（对齐 TensorRT 10.x）

升级容器 TensorRT 版本：

若需要 TensorRT 10.3 编译 engine，建议直接使用宿主机 `/usr/src/tensorrt/bin/trtexec` 编译，再把生成的 `.engine` 放回 `ai_engine/models`。

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
- ONNX 导出默认使用 `opset=19`，可通过 `--opset` 覆盖

## 4) 编译 Engine（Jetson 上执行）

仓库已提供编译脚本：

- [build-rfdetr-engine.sh](file:///c:/project/buildingos.vision/scripts/build-rfdetr-engine.sh)

标准 FP16 编译：

```bash
docker compose exec ai-engine bash -lc "bash /app/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-fp16.engine fp16"
```

TensorRT 10.x 建议参数（脚本默认已启用）：

- `MAX_AUX_STREAMS=0`
- `ALLOCATION_STRATEGY=runtime`
- `ENABLE_PREVIEW_PROFILE_SHARING=1`（仅在当前 trtexec 支持 `--preview` 时自动生效）
- `INPUT_NAME=input`（当前 RF-DETR ONNX 的实际输入名）

默认 shape 已按 RF-DETR 设置为：

- `MIN_SHAPE=1x3x640x640`
- `OPT_SHAPE=1x3x640x640`
- `MAX_SHAPE=2x3x640x640`
- 默认先走 `BUILD_MODE=static`、`SKIP_INFERENCE=1`、`WORKSPACE_MB=512`、`OPT_LEVEL=2`、`USE_TIMING_CACHE=0`、`USE_EXPLICIT_SHAPES=0`

## 5) 可选：自定义 shape / INT8

自定义动态 shape：

```bash
docker compose exec ai-engine bash -lc "BUILD_MODE=dynamic INPUT_NAME=images MIN_SHAPE=1x3x640x640 OPT_SHAPE=1x3x640x640 MAX_SHAPE=2x3x640x640 WORKSPACE_MB=512 OPT_LEVEL=2 USE_TIMING_CACHE=0 bash /app/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-fp16-dyn.engine fp16"
```

优先尝试 576x576（满足当前模型导出约束，且较 640 更省内存）：

```bash
docker compose exec ai-engine bash -lc "BUILD_MODE=static USE_EXPLICIT_SHAPES=0 OPT_SHAPE=1x3x576x576 WORKSPACE_MB=512 OPT_LEVEL=2 USE_TIMING_CACHE=0 bash /app/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-fp16-576.engine fp16"
```

INT8 编译：

```bash
docker compose exec ai-engine bash -lc "CALIB_CACHE=/app/models/rf-detr-int8.cache WORKSPACE_MB=512 OPT_LEVEL=2 USE_TIMING_CACHE=0 bash /app/build-rfdetr-engine.sh /app/models/rf-detr.onnx /app/models/rf-detr-int8.engine int8"
```

## 6) 验证与排错

验证产物：

```bash
docker compose exec ai-engine bash -lc "ls -lh /app/models/rf-detr*.engine"
```

常见失败项：

- `trtexec not found`
- 输入名不匹配（当前应为 `input`，可用 `INPUT_NAME` 覆盖）
- 若报 `Static model does not take explicit shapes`：静态模型请设置 `USE_EXPLICIT_SHAPES=0`
- ONNX 算子不被当前 TensorRT 支持（需调整导出版本或图）
- 若报 `aten::_upsample_bicubic2d_aa` 不支持：在更新的导出环境重试（建议 torch>=2.5, torchvision>=0.20, onnx>=1.16），并使用 `--opset 19`
- 若编译期出现 `double free or corruption`：先用 `BUILD_MODE=static OPT_LEVEL=2 SKIP_INFERENCE=1 WORKSPACE_MB=512 USE_TIMING_CACHE=0`，再尝试 `OPT_SHAPE=1x3x576x576`

若仍出现 `double free or corruption`，先做 ONNX 预处理再编译：

```bash
pip install onnx onnxsim
python scripts/prepare-rfdetr-onnx.py --input ai_engine/models/rf-detr.onnx --output ai_engine/models/rf-detr-trt.onnx --shape 1,3,576,576 --target-opset 16
docker compose exec ai-engine bash -lc "BUILD_MODE=static OPT_SHAPE=1x3x576x576 WORKSPACE_MB=512 OPT_LEVEL=1 USE_TIMING_CACHE=0 SKIP_INFERENCE=1 bash /app/build-rfdetr-engine.sh /app/models/rf-detr-trt.onnx /app/models/rf-detr-fp16-576.engine fp16"
```
