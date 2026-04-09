# Jetson Orin Nano 安装与调试 Google Gemma 4 E2B 指南

本文档基于 **2026 年 4 月 2 日发布的 Gemma 4 官方模型**重写，目标是让当前 BuildingOS Vision 所使用的 Jetson Orin Nano 8GB 能够按照**最新模型规格**完成安装、首轮调试、稳定性验证和后续接入准备。

本文统一使用官方模型名称 **Gemma 4 E2B**，对应的 Hugging Face 模型 ID 为：

```text
google/gemma-4-E2B-it
```

***

## 1. 本文档解决什么问题

这是一份偏 **How-to Guide** 的落地文档，面向已经具备 Jetson 运维能力的开发者或实施工程师，重点回答以下问题：

1. 当前最新的 Gemma E2B 到底是哪一代模型。
2. 在 Jetson Orin Nano 8GB 上，应该如何用**官方模型接口**完成安装与首测。
3. 如何在不破坏现有 YOLO、ZLM、Docker 业务的前提下完成调优和问题定位。

本文档覆盖：

1. Jetson 宿主机准备。
2. 基于 Transformers 的 **官方 Gemma 4 E2B** 安装流程。
3. 文本、图片、音频三类最小可用测试。
4. Jetson 上的调优和常见故障处理。

本文档不展开：

1. Node-RED / MQTT / RAG 的业务编排。
2. 生产环境队列设计与告警策略。
3. 与 BuildingOS 业务代码的最终接口封装。

***

## 2. 先明确最新模型事实

截至 **2026-04-02** 发布的官方 Gemma 4 资料，Gemma 4 E2B 具备以下特征：

- **模型代际**：Gemma 4
- **型号**：E2B
- **部署定位**：面向手机、边缘设备、IoT、Jetson 等本地场景
- **有效参数规模**：约 2.3B effective，含嵌入总量约 5.1B
- **上下文长度**：128K
- **支持模态**：Text、Image、Audio
- **官方对话能力**：支持 system role、thinking 模式、函数调用、结构化输出

这意味着当前文档必须从原先的 **Gemma 3n E2B** 路线切换为 **Gemma 4 E2B 官方模型路线**，不能继续把旧模型说明混用到最新实施步骤里。

***

## 3. Jetson 上的推荐实施路线

在 Jetson Orin Nano 8GB 上，推荐按以下顺序推进：

### 3.1 第一阶段：先跑通官方模型

优先目标不是一开始就做复杂多模态业务联调，而是先确保：

1. Jetson 上的 CUDA 可被 Python / PyTorch 正确识别。
2. `google/gemma-4-E2B-it` 可以正常加载。
3. 文本推理可以稳定返回结果。
4. 图片与音频的最小样例可以跑通。

### 3.2 第二阶段：再做资源收敛

Gemma 4 E2B 虽然是边缘型号，但在 Jetson Orin Nano 8GB 上依然要与以下负载共存：

- ZLMediaKit
- YOLO / TensorRT
- Docker 容器
- BuildingOS 业务服务

因此生产落地时必须遵守一个原则：

**Gemma 4 E2B 在当前项目中属于二级复核引擎，不是一级实时视频分析引擎。**

也就是说，它适合做：

- 低频复核
- 告警二次确认
- 本地问答与规则解释
- OCR / 单帧理解 / 短音频识别

而不适合：

- 长时间持续并发多路推理
- 与视频主链路抢 GPU
- 无限制地开启 128K 长上下文

***

## 4. 前置条件

开始前，建议你已经完成 [init.md](file:///c:/project/buildingos.vision/docs/init.md) 中的基础环境准备，尤其是以下部分：

- JetPack 6.x 已安装完成
- 系统已迁移到 NVMe SSD
- 已配置 Swap
- 已安装 `jtop`
- Docker 与 NVIDIA Runtime 正常

最低建议条件：

- **设备**：Jetson Orin Nano 8GB
- **系统**：JetPack 6.x
- **磁盘**：至少 25GB 可用空间
- **Swap**：建议 8GB，最好 12GB 或以上
- **网络**：首次需要访问 Hugging Face
- **Python**：建议使用系统 Python 3.10+

***

## 5. 宿主机准备

### 5.1 检查系统与 CUDA 状态

先确认 Jetson 当前环境：

```bash
uname -a
cat /etc/nv_tegra_release
nvcc --version
free -h
swapon --show
df -h
```

如果 `nvcc` 不存在，说明 CUDA 开发环境不完整，先不要继续。

### 5.2 切换到高性能模式

首次安装和压测时，建议固定到高性能模式：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 5.3 降低其他负载

在首次加载 Gemma 4 E2B 之前，建议暂时避免以下情况：

- 不要同时做 TensorRT 模型导出
- 不要同时进行多路高清视频解码压测
- 不要同时启动大批量 Docker 镜像构建
- 如非必要，先关闭图形桌面

### 5.4 先做显存预热，再启动 Gemma

在当前 BuildingOS 设备上，建议固定采用下面这条启动顺序：

1. 先启动 ZLM
2. 再启动 YOLO / TensorRT
3. 最后再启动 Gemma 4 E2B

原因是：YOLO 的 TensorRT 引擎通常会优先申请并固定一部分连续显存。如果先让 Gemma 4 把 8GB 统一内存中的大块空间占住，YOLO 后启动时可能会因为拿不到足够连续的显存而直接失败。

因此这里要明确一个实施原则：

**先让实时链路完成显存预热，再让 Gemma 4 作为二级复核引擎进入系统。**

***

## 6. Python 环境准备

### 6.1 安装基础依赖

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  git \
  wget \
  curl \
  ffmpeg \
  libsndfile1
```

### 6.2 创建虚拟环境

在 Jetson 上，**不要盲目在纯净 venv 中直接** **`pip install torch`**，因为这样很容易得到不带 Jetson CUDA 支持的通用 wheel。

更稳妥的方式是：

1. 复用 JetPack / NVIDIA 已提供的 CUDA 可用 PyTorch。
2. 虚拟环境使用 `--system-site-packages` 继承系统侧的 PyTorch。

```bash
python3 -m venv --system-site-packages ~/venvs/gemma4
source ~/venvs/gemma4/bin/activate
python -m pip install -U pip setuptools wheel
```

### 6.3 安装 Gemma 4 运行依赖

对于 **Jetson Orin Nano 8GB**，如果你要继续走 Transformers 官方模型路线，建议把 **4-bit 量化**视为默认前提，而不是可选优化。

先安装通用依赖：

```bash
python -m pip install -U \
  transformers \
  accelerate \
  sentencepiece \
  pillow \
  requests
```

再安装多模态依赖：

```bash
python -m pip install -U \
  torchvision \
  librosa
```

最后准备 4-bit 量化依赖：

```bash
python -m pip install -U bitsandbytes
```

说明：

- 在 8GB 统一内存设备上，不建议直接用全量 BF16 权重去赌首轮加载是否成功。
- 如果当前 Jetson 环境里的 `bitsandbytes` 无法直接安装成功，就应把它视为一个**阻塞问题**优先解决。
- 对于 Jetson Orin Nano 8GB，`flash_attention_2` 不是可选优化，而是宿主机 Transformers 路线的硬前提。
- 如果当前环境无法稳定启用 `flash_attention_2`，宁可先停止这条路线，也不要退回默认的 Eager Attention。
- 如果短期内无法让 `bitsandbytes` 在该 Jetson 环境稳定工作，则不建议继续推进这条宿主机 Transformers 路线，而应转向更省内存的量化部署方案。

***

## 7. 先验证 PyTorch 是否真的能用 GPU

在开始加载 Gemma 4 E2B 之前，必须先验证当前 Python 环境不是 CPU 假运行。

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
PY
```

预期结果：

- `cuda_available: True`
- `cuda_device_count: 1`
- 设备名称能正确显示 Jetson GPU

如果这里失败，先不要继续加载 Gemma 4。

在继续之前，再补一个 4-bit 量化后端检查：

```bash
python - <<'PY'
from importlib.util import find_spec
print("bitsandbytes:", find_spec("bitsandbytes") is not None)
PY
```

如果这里返回 `False`，本文后续的 8GB 优化加载脚本就不应直接执行。

***

## 8. 官方模型最小文本测试

虽然 Gemma 4 官方模型卡给出了文本场景下的 `AutoModelForCausalLM` 示例，但为了让 **文本 / 图片 / 音频** 三条测试路径共用同一套加载范式，并确保视觉与音频投影层也一起完成首轮验证，本文统一改用 **原生多模态加载类**。

推荐顺序如下：

1. 如果当前 Transformers 版本已暴露 `Gemma4ForConditionalGeneration`，优先显式使用它
2. 如果没有该类，则退回 `AutoModelForVision2Seq`
3. 不再把 `AutoModelForMultimodalLM` 作为本文默认类名；如果你当前环境里用它报错，应立即切换到前两种写法

另外，**不建议**在 8GB Jetson 上依赖 `device_map="auto"` 作为默认方案，因为它可能悄悄把一部分权重拆到 CPU，导致虽然“能跑”，但速度非常慢且不稳定。更稳妥的做法是优先尝试**单卡固定映射**。

同时要强调：`attn_implementation="flash_attention_2"` 在当前设备上属于**强制项**，不是建议项。如果环境不能稳定启用 Flash Attention 2，就不要继续执行本文的宿主机 Transformers 路线。

先从**文本模式**跑通，因为这是定位问题最简单的一步。

新建测试脚本：

`~/ai/gemma4/smoke_text.py`

```python
import transformers
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
Gemma4ForConditionalGeneration = getattr(transformers, "Gemma4ForConditionalGeneration", None)

def resolve_model_cls():
    if Gemma4ForConditionalGeneration is not None:
        return Gemma4ForConditionalGeneration
    return AutoModelForVision2Seq

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = resolve_model_cls().from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)

attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
if attn_impl != "flash_attention_2":
    raise RuntimeError("Jetson Orin Nano 8GB 必须启用 flash_attention_2，未启用时不要继续测试。")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用两句话说明 Jetson Orin Nano 为什么适合边缘 AI。"},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )

response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))
```

运行：

```bash
mkdir -p ~/ai/gemma4
python ~/ai/gemma4/smoke_text.py
```

通过标准：

- 模型可成功下载并加载
- 终端有正常文本输出
- 进程没有被系统直接杀掉
- `jtop` 中可以观察到 GPU / RAM 变化

额外要求：

- 如果启动后 CPU 占用异常高而 GPU 几乎不动，要优先怀疑是否发生了 CPU offload
- 如果 `device_map={"": 0}` 报错，再退回 `device_map="auto"` 做诊断，但不要把它当作 8GB 设备的默认建议
- 如果这里无法稳定启用 `flash_attention_2`，就不要再继续做图片和音频测试

***

## 9. 官方图片理解测试

Gemma 4 E2B 是官方支持图片输入的。图片测试建议放在文本测试通过之后再做。

针对 BuildingOS 的监控类场景，建议在进入处理器之前先**显式缩放图片**。对于 Jetson Orin Nano 8GB，首轮建议先固定到 **448x448**，这比直接把原始大图丢给视觉编码器更稳。

但仅仅缩放还不够。针对当前“抽烟检测 / 监控复核”业务，建议在 `processor` 一侧继续**显式收敛视觉展开预算**：

- 如果当前实现暴露 `max_image_tokens`、`image_seq_length` 或类似参数，就直接限制视觉 Token 数量
- 如果当前实现暴露 `patch_size`、`num_crops` 或类似参数，也要显式配置，不要完全依赖默认值
- 如果当前实现没有暴露上述参数，就至少坚持单图输入、固定 `448x448`、短问题这三条硬约束

目标不是把视觉信息压到最小，而是避免视觉 Token 默认展开到 1000+ 后，挤占 YOLO / TensorRT 的显存窗口。

新建测试脚本：

`~/ai/gemma4/smoke_image.py`

```python
import transformers
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
Gemma4ForConditionalGeneration = getattr(transformers, "Gemma4ForConditionalGeneration", None)

def resolve_model_cls():
    if Gemma4ForConditionalGeneration is not None:
        return Gemma4ForConditionalGeneration
    return AutoModelForVision2Seq

def tighten_vision_budget(processor):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "size"):
        image_processor.size = {"height": 448, "width": 448}
    if hasattr(image_processor, "num_crops"):
        image_processor.num_crops = 1
    if hasattr(image_processor, "max_image_tokens"):
        image_processor.max_image_tokens = 768

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
tighten_vision_budget(processor)
model = resolve_model_cls().from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)

attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
if attn_impl != "flash_attention_2":
    raise RuntimeError("Jetson Orin Nano 8GB 必须启用 flash_attention_2，未启用时不要继续测试。")

image = Image.open("/home/YOUR_USER/ai/gemma4/test.jpg").convert("RGB")
image = image.resize((448, 448))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "请描述图片中的场景，并判断是否有人。"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )

response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))
```

运行前准备：

1. 将一张测试图片放到 `/home/YOUR_USER/ai/gemma4/test.jpg`
2. 把脚本里的 `YOUR_USER` 替换成实际用户名

运行：

```bash
python ~/ai/gemma4/smoke_image.py
```

建议第一轮测试使用：

- 单张图片
- 分辨率固定到 `448x448`
- 如有参数入口，显式限制视觉 Token 预算
- 不要同时混入音频或视频
- 优先做“是否有人”“是否存在香烟/烟雾”“是否可读文字”这类短问题

***

## 10. 官方音频理解测试

Gemma 4 E2B 与 E4B 是官方支持音频输入的小模型版本。音频测试建议放在文本和图片都通过之后再做。

音频输入这里有一个非常关键的实践约束：**请统一使用 16,000Hz 的单声道 WAV**。不要把 44.1kHz 或立体声文件直接丢进去做首测。

新建测试脚本：

`~/ai/gemma4/smoke_audio.py`

```python
import transformers
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
Gemma4ForConditionalGeneration = getattr(transformers, "Gemma4ForConditionalGeneration", None)

def resolve_model_cls():
    if Gemma4ForConditionalGeneration is not None:
        return Gemma4ForConditionalGeneration
    return AutoModelForVision2Seq

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = resolve_model_cls().from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)

attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
if attn_impl != "flash_attention_2":
    raise RuntimeError("Jetson Orin Nano 8GB 必须启用 flash_attention_2，未启用时不要继续测试。")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/home/YOUR_USER/ai/gemma4/test.wav"},
            {
                "type": "text",
                "text": "请转写这段音频，只输出转写结果，不要加解释。"
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )

response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))
```

音频建议：

- 使用 **16,000Hz** 单声道 WAV
- 时长控制在 30 秒以内
- 第一轮先用清晰人声
- 如需转换，可先用 `ffmpeg` 统一重采样

***

## 11. Thinking 模式与对话模板注意事项

Gemma 4 与旧版 Gemma 系列相比，一个关键变化是：**thinking 模式和 system role 已成为官方推荐用法的一部分。**

### 11.1 首轮测试建议关闭 thinking

原因：

- 输出更短
- 更容易看懂
- 更容易判断是不是模型本身加载失败
- 可以减少无意义的调试变量

在 Transformers 中，直接在 `apply_chat_template` 里使用：

```python
enable_thinking=False
```

### 11.2 如果要开启 thinking

可以改成：

```python
enable_thinking=True
```

开启后，模型会输出思考通道和最终答案，日志和解析逻辑会更复杂，因此不建议作为 Jetson 上的首轮验收模式。

### 11.3 多轮对话注意事项

Gemma 4 官方建议：**历史对话中只保留最终可见答案，不要把 thought 内容回灌到下一轮。**

这点很重要，否则上下文会膨胀得非常快。

***

## 12. Jetson 调优建议

### 12.1 不要一上来就吃满 128K 上下文

Gemma 4 E2B 虽然官方支持 128K，但在 Jetson Orin Nano 8GB 上绝不建议这样起步。

首轮验证建议：

- 文本：短 prompt
- 图片：单张图片 + 简短问题
- 音频：30 秒内单条
- `max_new_tokens`：先用 `128`

等基础稳定后，再逐步增加输入规模。

更实际的工程建议是：

- 首轮运行把业务输入窗口控制在 `4K` 以内
- 稳定后再尝试 `8K`
- 在 8GB 设备上不要把官方“支持 128K”误解成“本地可直接长期稳定使用 128K”

### 12.2 推荐调优顺序

如果要逐步压测，按这个顺序调：

1. 先调输入长度
2. 再调 `max_new_tokens`
3. 最后再尝试更复杂的多模态输入

不要同时：

- 长上下文
- 大图
- 音频
- 多请求并发

如果业务侧需要长文档或长对话，应优先采用：

- 分块
- 摘要后再送入
- 外部状态机保留上下文

而不是让 Jetson 本机直接维护超长 KV Cache。

### 12.3 采样参数保持官方默认

Gemma 4 官方建议的采样参数是：

```text
temperature = 1.0
top_p = 0.95
top_k = 64
```

在调试阶段不建议随便改动这三个值，否则很难区分是模型行为变化，还是系统资源问题。

### 12.4 4-bit 量化是 8GB 路线的默认前提

对于当前这台还要同时运行 ZLM、YOLO、Docker 的 Jetson，建议把下面这个组合视为默认基线：

- `BitsAndBytesConfig(load_in_4bit=True, ...)`
- `torch_dtype=torch.bfloat16`
- `attn_implementation="flash_attention_2"`，并在加载后再次确认没有静默退回其他实现
- 单卡固定映射优先，避免不受控 CPU offload

这里要明确一点：`flash_attention_2` 在 Jetson Orin Nano 8GB 上是**必须项**，不是“有更好，没有也行”的优化项。它的价值不只是提速，更重要的是降低显存压力。如果当前环境不能稳定启用它，就不要假设宿主机 Transformers 路线还能轻松稳定运行。

### 12.5 模态顺序

Gemma 4 官方建议：**图片或音频应放在文本之前。**

因此像下面这种写法是推荐的：

```python
[
  {"type": "image", "image": "..."},
  {"type": "text", "text": "..."}
]
```

而不是先写文本再加图片。

### 12.6 与 BuildingOS 现有业务共存

当前设备上已有：

- 视频接入
- ZLM 转发
- YOLO / TensorRT
- Docker 容器

因此 Gemma 4 E2B 的运行策略建议为：

- 单请求串行
- 非实时调用
- 仅在需要复核时触发
- 空闲时常驻，忙时限流

并建议增加一个**显存锁 / 推理锁**机制：

- Gemma 开始推理前，先获取全局锁
- YOLO 大批量任务期间，不允许并发发起 Gemma 多模态请求
- Gemma 推理窗口内，尽量避免 ZLM 进入高缓存峰值状态

此外，启动顺序也应固定：

1. 先启动 ZLM
2. 再启动 YOLO / TensorRT
3. 最后再启动 Gemma 4

这样做的核心目的，是先让实时主链路完成显存预热和连续显存占位，再把 Gemma 作为二级复核负载接入，避免 YOLO 在后启动时因为显存碎片或连续显存不足而崩溃。

***

## 13. 常见问题与排查

### 13.1 `torch.cuda.is_available()` 为 False

说明当前 Python 环境没有拿到 Jetson CUDA。

优先检查：

1. 是否误用了纯 pip 安装的 CPU 版 torch
2. venv 是否没有使用 `--system-site-packages`
3. 当前 JetPack 的 PyTorch 是否本身可用

### 13.2 `bitsandbytes` 安装或导入失败

说明当前 4-bit 路线尚未打通。

优先检查：

1. 当前 Jetson 架构下的 wheel 是否可用
2. 是否需要源码编译
3. 当前 CUDA / Python / PyTorch 版本是否匹配

如果这一项没有解决，不建议继续按本文的 8GB 宿主机 Transformers 路线推进。

### 13.3 `flash_attention_2` 不可用或静默退回默认实现

对于本文这条 8GB 宿主机 Transformers 路线，这应视为**阻塞问题**。

优先检查：

1. 当前 PyTorch、Transformers 与 JetPack 版本是否匹配
2. 当前环境是否真的编译或提供了 Flash Attention 2 能力
3. 模型加载后是否通过 `model.config` 再次确认仍为 `flash_attention_2`

处理原则：

- 不要退回 Eager 模式继续“先跑起来再说”
- 不要在这种状态下继续做图片和音频测试
- 先解决环境问题，再继续本文路线

### 13.4 模型加载时直接被杀死

常见原因：

- 内存不足
- Swap 不足
- 同时运行的服务太多
- 第一次就上图片 / 音频 / 长上下文

处理顺序：

1. 先只跑文本脚本
2. 把 `max_new_tokens` 降到 64 或 128
3. 关闭桌面和其他重负载服务
4. 确认 Swap 至少 8GB

### 13.5 输出里有奇怪的 thought 标记

通常是：

- 开启了 thinking
- 或者解析方式不对

解决方式：

- 首轮调试统一使用 `enable_thinking=False`
- 最终输出统一通过 `processor.parse_response(response)` 处理

### 13.6 图片推理很慢

先排查：

1. 图片是否过大
2. 是否同时还在跑 YOLO 视频推理
3. 是否还保留图形桌面

建议：

- 第一轮图片测试只用一张普通 JPG
- 不要直接拿高清监控大图做首测
- 先固定缩放到 `448x448`
- 如有参数入口，显式压低视觉 Token 预算

### 13.7 音频推理失败

优先检查：

1. 文件格式是不是标准 WAV
2. 是否为 **16,000Hz 单声道**
3. 时长是否过长
4. `librosa`、`ffmpeg`、`libsndfile1` 是否安装完整

### 13.8 为什么不再把 llama.cpp 作为本文主线

因为这次文档要求必须**完全按照最新的 Gemma 4 官方模型**来安装和调试。

对于 Jetson 来说，`llama.cpp` / GGUF 仍然可以作为后续的低内存优化路线，但它不应该取代当前这份文档中的**官方模型首轮验收路径**。

***

## 14. BuildingOS 业务接入约束

在当前项目里，建议先把下面这条规则写死在代码或服务编排层：

```python
import threading

gemma_infer_lock = threading.Lock()

def run_gemma_review(review_fn, *args, **kwargs):
    with gemma_infer_lock:
        return review_fn(*args, **kwargs)
```

最少要落实以下约束：

1. 同一时刻只允许一个 Gemma 4 E2B 推理任务进入执行区
2. YOLO 批量检测高峰期间，不触发 Gemma 多模态复核
3. Gemma 推理任务只处理单帧、短文本、短音频，不参与长视频主链路
4. 如需和 ZLM 协调，优先在 Gemma 推理窗口内降低非必要缓存与预取压力
5. 服务启动顺序固定为 ZLM → YOLO / TensorRT → Gemma 4，不要反过来抢连续显存

***

## 15. 最小验收清单

完成本文档后，至少应满足：

- 能正确识别 `google/gemma-4-E2B-it`
- Python 环境中 `torch.cuda.is_available()` 为 True
- `bitsandbytes` 可正常导入
- `flash_attention_2` 已确认启用，且没有静默退回默认实现
- 文本脚本可正常返回
- 图片脚本可正常返回
- 音频脚本可正常返回
- 在 `jtop` 中可观测到资源变化
- ZLM 与 YOLO 已先完成启动，再接入 Gemma 4
- 与现有 YOLO / ZLM 共存时，系统未失稳

***

## 16. 下一步建议

当上述内容全部通过后，再进入 BuildingOS 的下一阶段：

1. 将 Gemma 4 E2B 封装为本地调用服务
2. 为低置信度事件增加图片复核
3. 为短语音命令或语音转写增加本地入口
4. 建立串行任务队列，避免与 YOLO 抢 GPU

---

## 17. 性能优化与内存管理建议 (Jetson Orin Nano 8GB)

在统一内存架构的 Jetson 设备上，内存（显存）是最宝贵的资源。为了确保 YOLO 视频流推理与 Gemma 4 E2B 本地大模型能够稳定共存，强烈建议进行以下优化：

### 17.1 关闭图形桌面 (GUI) 释放内存

默认开启的 Ubuntu 图形桌面（GNOME/LightDM 等）会占用约 **800MB 到 1.2GB** 的基础内存。关闭 GUI 可以将这部分内存完全释放给 AI 模型使用，极大降低 OOM 风险。

**操作方法：**
打开终端，将系统默认启动级别设置为命令行模式（multi-user），然后重启：
```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

**影响评估：**
- **不会影响**：后台服务（如 Docker 容器、Nginx、Web Manager）、AI 推理加速（CUDA/TensorRT）、硬件编解码（NVDEC）、SSH 远程登录。
- **会受影响**：无法通过连接物理显示器查看桌面；运行 Python 脚本时不能使用 `cv2.imshow()` 弹出本地窗口。

*(如需恢复图形桌面，执行 `sudo systemctl set-default graphical.target` 并重启即可)*

### 17.2 及时清理 Gemma 推理缓存 (Context Slots)

在使用 `llama.cpp` (`llama-server`) 提供单次图片/文本复核服务时，每次推理都会在内存中占用大量的上下文缓存（Context Slots）。由于我们的业务通常是“单次无状态分析”（不依赖多轮对话历史），因此**推理结束后必须主动释放缓存**。

**操作方法：**
在 Python 或 Node.js 业务代码中，在每次请求 `/v1/chat/completions` 完成后，发送一个 `DELETE` 请求到 `/slots/0` 端点（假设使用 `--parallel 1` 启动）：

```bash
# 命令行测试释放缓存
curl -X DELETE http://127.0.0.1:8080/slots/0
```
这能立刻释放掉之前图片占用的大量上下文内存，而不需要重启整个 Gemma 进程。

### 17.3 jtop 采集守护进程 systemd 落地 (开机自启 + 自动重拉)

为了让 Web Dashboard 长期稳定读取 Jetson 实时硬件状态（CPU / RAM / Swap / GPU / Power / 温度 / 引擎占用），建议将 `jtop_daemon.py` 托管给 systemd，而不是手工在终端运行。

注意：`jtop-daemon.service` 不是系统预置服务，Docker 也不会自动在宿主机创建该 unit。首次部署必须在宿主机手工创建一次，后续即可由 systemd 托管并自动拉起。

#### 17.3.1 前置检查

当前设备统一按 Jetson Nano 8G 标准环境交付，`jtop.service` 已存在。落地前仅需确认服务与脚本状态：

```bash
sudo systemctl status jtop.service
```

确保项目脚本存在：

```bash
ls -l /home/buildingos/buildingos.vision/web_manager/backend/jtop_daemon.py
```

#### 17.3.2 创建服务文件

新建：

`/etc/systemd/system/jtop-daemon.service`

推荐直接使用下面一条命令创建服务文件（可避免手工编辑格式错误）：

```bash
sudo tee /etc/systemd/system/jtop-daemon.service > /dev/null << 'EOF'
[Unit]
Description=BuildingOS jtop monitoring daemon
After=network-online.target jtop.service
Wants=network-online.target

[Service]
Type=simple
User=buildingos
WorkingDirectory=/home/buildingos/buildingos.vision/web_manager/backend
ExecStart=/usr/bin/python3 /home/buildingos/buildingos.vision/web_manager/backend/jtop_daemon.py
Restart=always
RestartSec=2
StartLimitIntervalSec=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

等价配置内容如下：

```ini
[Unit]
Description=BuildingOS jtop monitoring daemon
After=network-online.target jtop.service
Wants=network-online.target

[Service]
Type=simple
User=buildingos
WorkingDirectory=/home/buildingos/buildingos.vision/web_manager/backend
ExecStart=/usr/bin/python3 /home/buildingos/buildingos.vision/web_manager/backend/jtop_daemon.py
Restart=always
RestartSec=2
StartLimitIntervalSec=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 17.3.3 启用、启动、查看

```bash
sudo systemctl daemon-reload
sudo systemctl enable jtop-daemon.service
sudo systemctl start jtop-daemon.service
sudo systemctl status jtop-daemon.service

# 实时日志
sudo journalctl -u jtop-daemon.service -f
```

#### 17.3.4 常用运维命令

```bash
# 重启服务（修改脚本后常用）
sudo systemctl restart jtop-daemon.service

# 停止服务
sudo systemctl stop jtop-daemon.service

# 取消开机自启
sudo systemctl disable jtop-daemon.service
```

#### 17.3.5 验证桥接文件是否持续更新

```bash
ls -l /tmp/jtop_status.json
cat /tmp/jtop_status.json | head
```

如果时间戳持续变化，说明采集正常；若 `systemd` 检测到脚本异常退出，会按 `Restart=always` 自动拉起，满足“断线自动重拉”的要求。

### 17.4 jtop 采样字段对照清单 (现场排查速查)

为便于现场快速判断“性能瓶颈在 CPU / 内存 / GPU / 视频链路哪一侧”，下面给出 Dashboard 与 `jtop_daemon.py` 对应的数据字段说明。

#### 17.4.1 CPU

- `cpu.usage`：CPU 总体使用率（%）
- `cpu.cores`：CPU 核心数
- `cpu.details.<core>.usage`：单核使用率（%）
- `cpu.details.<core>.freq`：单核当前频率（MHz）

排查建议：

- `cpu.usage` 长期 > 85%：会导致请求排队，推理时延上升
- 单核长期打满：优先排查 Python 侧串行热点或日志 IO 压力

#### 17.4.2 RAM / SWAP（统一内存）

- `memory.ram.used/total/free/shared/cached/buffers`
- `memory.ram.usagePercent`
- `memory.swap.used/total/cached`
- `memory.swap.usagePercent`

排查建议：

- `ram.usagePercent` > 90%：GPU 可用统一内存被挤占
- `swap.used` 持续增长：系统进入抖动风险区，应降并发或减负载

#### 17.4.3 GPU

- `gpu.util`：GPU 利用率（%）
- `gpu.freq`：GPU 当前频率（MHz）
- `gpu.freqMax`：GPU 理论上限频率（MHz）
- `gpu.memUsed/memTotal`：按统一内存口径映射的显存占用（MB）

排查建议：

- `gpu.util` 高但 `gpu.freq` 明显下降：常见于热限制/功耗限制
- `gpu.util` 低且时延高：可能是 CPU 侧或视频链路在抢资源

#### 17.4.4 功耗 Power

- `power.total`：整机功耗（mW）
- `power.gpu` / `power.cpu`
- `power.soc` / `power.cv`
- `power.vdd_in`：输入总功耗轨（机型相关）

排查建议：

- 功耗接近平台上限且频率下降：优先检查散热、风扇、nvpmodel

#### 17.4.5 温度 Temperature

- `temperature.<sensor>`：各温区温度（°C），如 GPU、CPU、SOC 热区

排查建议：

- 温度持续 > 80°C：建议立即关注热降频与风扇策略

#### 17.4.6 硬件引擎 Engine

- `engines.NVDEC`：解码引擎占用
- `engines.NVENC`：编码引擎占用
- `engines.VIC`：图像合成/缩放引擎占用
- `engines.NVJPG`：JPEG 引擎占用
- `engines.NVDLA*`：DLA 引擎（若机型支持）

排查建议：

- `NVDEC/NVENC/VIC` 长期高占用：视频链路正在抢资源，LLM 响应会变慢

#### 17.4.7 板卡状态 Board

- `board.model`
- `board.jetpack`
- `board.nvpmodel`
- `board.jetsonClocks`
- `board.uptime`

排查建议：

- `nvpmodel` 非预期模式时，先统一切回部署基线后再比较性能

---

## 18.实际安装落地步骤

# 1. 永久修复 CUDA 环境变量

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

```

再验一次：

```bash
nvcc --version

```

返回：

```
buildingos@ubuntu:~$ nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver

Copyright (c) 2005-2024 NVIDIA Corporation

Built on Wed_Aug_14_10:14:07_PDT_2024

Cuda compilation tools, release 12.6, V12.6.68

Build cuda_12.6.r12.6/compiler.34714021_0

buildingos@ubuntu:~$ nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver

Copyright (c) 2005-2024 NVIDIA Corporation

Built on Wed_Aug_14_10:14:07_PDT_2024

Cuda compilation tools, release 12.6, V12.6.68

Build cuda_12.6.r12.6/compiler.34714021_0
```

表示环境问题已经闭环。

# 2. 安装Jetson Pytorch基座

在 Jetson 平台上，NVIDIA 官方认可的安装方式主要有三类：

## 方式 A：NVIDIA L4T PyTorch 官方论坛 Wheel (你当前的选择)

来源：由 NVIDIA Jetson 团队维护并在 NVIDIA 开发者论坛 发布的 .whl 文件。

特点：这是为了让 Python 的 venv 能直接利用 CUDA、cuDNN 和 TensorRT 加速。针对 JetPack 6.2 / CUDA 12.6，应优先使用 **JetPack 6.2 已验证可运行的专用 wheel 组合**。

结论：这是做 Gemma 4 开发最推荐的方式，因为它能让 Torch 版本、CUDA 版本和 Orin 的实际可执行内核保持一致。

## 方式 B：Jetson-containers / NVIDIA Cloud Native Stack

来源：NVIDIA 提供的 Docker 镜像。

特点：环境最稳，但对于要在宿主机运行 ZLM 和 YOLO 的你来说，容器嵌套容器会增加管理复杂度。

## 方式 C：系统预装 (JetPack 默认)

特点：版本通常较旧（如 Torch 2.1 或 2.3），往往跟不上 2026 年最新模型的需求。

当前采用方式 A，并以 **JetPack 6.2 + CUDA 12.6 + Orin 实测可用**的组合作为基线：

- `torch==2.8.0`
- `torchvision==0.23.0`

说明：

- 这组版本已经在当前 Jetson Orin Nano 上完成 CUDA 张量与矩阵乘法验证。
- 先前尝试过的 `torch 2.11.0+cu126` 虽然能识别 GPU，但会在实际 CUDA kernel 执行时报 `no kernel image is available for execution on the device`，因此不再作为本文档推荐版本。

先执行：

```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev python3-venv wget
```

然后创建专用虚拟环境：

```bash
python3 -m venv ~/venvs/gemma4
source ~/venvs/gemma4/bin/activate
python -m pip install -U pip setuptools wheel
```

清理此前试错留下的旧包，并固定基础依赖：

```bash
source ~/venvs/gemma4/bin/activate
python -m pip uninstall -y torch torchvision torchaudio triton
python -m pip cache purge
python -m pip install -U pip setuptools wheel 'numpy<2'
```

安装 Jetson 对应的 torch wheel：

```bash
source ~/venvs/gemma4/bin/activate
python -m pip install --no-cache-dir \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

装完立刻验证：

```bash
python - <<'PY'
import time
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("cc:", torch.cuda.get_device_capability(0) if torch.cuda.device_count() else "N/A")

a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()

print("result_shape:", c.shape)
print("result_dtype:", c.dtype)
print("result_device:", c.device)
print("elapsed_sec:", round(time.time() - t0, 4))
print("mean:", float(c.float().mean()))
PY
```

通过标准：

- `torch: 2.8.0`
- `cuda: 12.6`
- cuda\_available: True
- `device_count: 1`
- `device_name: Orin`
- `result_device: cuda:0`
- 没有出现 `no kernel image is available for execution on the device`

如果这里通过，说明当前 Jetson 上的 PyTorch 基座已经真正打通，可以继续进入 `transformers`、`flash_attention_2` 和 Gemma 4 文本首测阶段。

安装 `transformers` 基础依赖：

```bash
source ~/venvs/gemma4/bin/activate
python -m pip install -U \
  'numpy<2' \
  transformers \
  accelerate \
  sentencepiece \
  pillow \
  requests \
  librosa
```

先检查当前 `transformers` 是否已经暴露 Gemma 4 的原生加载类：

```bash
python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
print("Gemma4ForConditionalGeneration:", hasattr(transformers, "Gemma4ForConditionalGeneration"))
print("AutoModelForVision2Seq:", hasattr(transformers, "AutoModelForVision2Seq"))
PY
```

当前这台 Jetson 的实测结果是：

- `transformers: 5.5.0`
- `Gemma4ForConditionalGeneration: True`
- `AutoModelForVision2Seq: False`

这意味着后续脚本应优先直接使用 `Gemma4ForConditionalGeneration`。

安装 Flash Attention 2：

```bash
source ~/venvs/gemma4/bin/activate
python -m pip uninstall -y flash-attn
python -m pip cache purge
python -m pip install --no-cache-dir \
  flash-attn==2.8.2 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

安装后立刻验证：

```bash
python - <<'PY'
import flash_attn
import flash_attn_2_cuda
print("flash_attn:", flash_attn.__version__)
print("flash_attn_2_cuda_import: OK")
PY
```

```bash
python - <<'PY'
try:
    from transformers.utils import is_flash_attn_2_available
    print("flash_attn_2_available:", is_flash_attn_2_available())
except Exception as e:
    print("flash_attn_2_available_check_error:", repr(e))
PY
```

当前这台 Jetson 的实测结果是：

- `flash_attn_2_available: True`

通过标准：

- `Gemma4ForConditionalGeneration: True`
- `flash_attn_2_available: True`

如果这里通过，说明当前环境已经满足 Gemma 4 文本首测所需的模型加载类与注意力实现前提，可以继续进入文本最小测试。

Gemma 4 文本最小测试：

先补齐 4-bit 量化依赖：

```bash
source ~/venvs/gemma4/bin/activate
python -m pip uninstall -y bitsandbytes
python -m pip cache purge
python -m pip install --no-cache-dir \
  bitsandbytes==0.48.0.dev0+ff389db \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

先验证 `bitsandbytes` 是否已可导入：

```bash
python - <<'PY'
from importlib.util import find_spec
print("bitsandbytes:", find_spec("bitsandbytes") is not None)
PY
```

```bash
python - <<'PY'
import bitsandbytes as bnb
print("bitsandbytes_version:", bnb.__version__)
PY
```

```bash
python -m bitsandbytes
```

如果这里都通过，再继续执行文本首测。

用 `cat` 方式直接创建文本首测脚本：

```bash
mkdir -p ~/ai/gemma4
mkdir -p ~/ai/gemma4/offload

cat > ~/ai/gemma4/smoke_text.py <<'PY'
import os

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
import transformers
import transformers.modeling_utils as modeling_utils
from transformers import AutoProcessor, BitsAndBytesConfig

def _noop_warmup(*args, **kwargs):
    return None

modeling_utils.caching_allocator_warmup = _noop_warmup

MODEL_ID = "google/gemma-4-E2B-it"
Gemma4ForConditionalGeneration = getattr(transformers, "Gemma4ForConditionalGeneration", None)

if Gemma4ForConditionalGeneration is None:
    raise RuntimeError("当前 transformers 未暴露 Gemma4ForConditionalGeneration，不要继续本文这条文本首测路线。")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
    max_memory={0: "5GiB", "cpu": "20GiB"},
    offload_folder="/home/buildingos/ai/gemma4/offload",
    offload_state_dict=True,
    low_cpu_mem_usage=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用两句话说明 Jetson Orin Nano 为什么适合边缘 AI。"},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = processor(text=text, return_tensors="pt")
inputs = {k: v.to(model.device if hasattr(model, "device") else "cuda:0") for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))
PY
```

说明：

- `cat > 文件 <<'PY'` 会在命令行中直接创建并写入整个 Python 文件
- 结束标记 `PY` 必须单独占一行，前后不要加空格
- 这种方式比逐行编辑更适合在 Jetson 命令行里快速落地首测脚本
- 这份脚本当前按 `sdpa + device_map="auto" + offload` 方式组织，用于当前 8GB 设备上的诊断型文本首测

运行命令：

```bash
python ~/ai/gemma4/smoke_text.py
```

当前这台 Jetson 的实测结果是：

- 权重可以完整加载
- 能返回 assistant 文本结果
- 首轮加载会明显偏慢
- 生成结果可能偏短，必要时可把 `max_new_tokens` 从 `64` 提高到 `96` 或 `128`

通过标准：

- 模型可成功下载并加载
- 终端有正常文本输出
- 进程没有被系统直接杀掉
- `jtop` 中可以观察到 GPU / RAM 变化
- 允许发生 CPU / 磁盘 offload
- 首轮加载耗时较长属于当前设备上的正常现象

如果这里通过，说明当前 Jetson 宿主机路线已经完成从 CUDA、PyTorch、Transformers、4-bit 量化到 Gemma 4 文本推理的最小闭环。

## 18. 生产环境 C++ 部署

既然当前 **Jetson Orin Nano 8GB** 在宿主机 Python / Transformers 路线下只能以**诊断型、低速、带 offload** 的方式运行，那么生产环境应转向 **llama.cpp + GGUF** 路线。

这条路线的目标不是复现 Python 侧的全部调试便利，而是优先获得下面三项能力：

1. 更可控的显存与内存占用
2. 更稳定的服务启动与重启行为
3. 更适合与 YOLO / ZLM 并存的常驻推理服务

***

### 18.1 先退出 Python 推理模式并清理环境

在开始 C++ 部署前，先把之前的 Python 推理环境尽量收干净，避免残留的进程、缓存和 offload 文件继续占用内存或磁盘。

#### 18.1.1 清理运行态

```bash
pkill -f smoke_text.py || true
pkill -f "python .*gemma" || true
pkill -f "python .*transformers" || true
sync
```

如果当前 shell 还停留在 `gemma4` 虚拟环境中，先退出：

```bash
deactivate || true
```

#### 18.1.2 清理 Python 推理临时文件

```bash
rm -rf ~/ai/gemma4/offload
rm -rf ~/ai/gemma4/__pycache__
rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it
rm -rf ~/.cache/huggingface/xet
```

如果后续不再保留 Python 路线，可进一步删除整个虚拟环境：

```bash
rm -rf ~/venvs/gemma4
```

#### 18.1.3 清理完成后复查

```bash
free -h
df -h
ps -ef | grep -Ei "python|gemma" | grep -v grep
```

理想状态是：

- 不再有 Gemma 相关 Python 进程
- `~/ai/gemma4/offload` 已被清空
- Hugging Face 大模型缓存已按需移除

***

### 18.2 为什么生产环境更适合 C++ 路线

对于当前这台设备，C++ 路线的价值主要体现在下面几个方面：

- **更少运行时依赖**：不再依赖 PyTorch、Transformers、bitsandbytes、flash-attn 的组合兼容性
- **更稳的内存边界**：GGUF 量化模型更适合在 8GB 设备上做受控部署
- **更适合服务化**：`llama-server` 更容易放进 `systemd` 或 supervisor 体系中常驻运行
- **更利于与 YOLO / ZLM 共存**：更容易约束上下文、线程数和 GPU 层数

***

### 18.3 步骤一：编译 Jetson 专属 llama.cpp 后端

不要使用通用预编译包，直接在 Jetson 宿主机上本地编译。

```bash
sudo apt update
sudo apt install -y git cmake build-essential ninja-build libssl-dev

mkdir -p ~/ai
cd ~/ai
git clone https://github.com/ggerganov/llama.cpp
cd ~/ai/llama.cpp

cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build -j$(nproc)
```

当前这台 Jetson 的实测结果表明，重新配置时应重点确认下面两点：

- 输出中出现 `-- Using CMAKE_CUDA_ARCHITECTURES=87`
- 输出中出现 `-- Found OpenSSL`

这表示当前编译已经按 **Orin 的 SM 87** 显式生成 CUDA 目标，同时本地 HTTPS 相关依赖也已被正确找到。

编译完成后，至少确认下面两个二进制存在：

```bash
ls -l ~/ai/llama.cpp/build/bin/llama-cli
ls -l ~/ai/llama.cpp/build/bin/llama-server
```

***

### 18.4 步骤二：准备 GGUF 模型与视觉投影文件

生产环境不再直接使用 `.safetensors` 官方权重，而是准备已经转换好的 GGUF 文件。

当前建议的目录结构：

```bash
mkdir -p ~/ai/llama.cpp/models/gemma4
```

建议放入：

- 主模型：`~/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf`
- 视觉投影：`~/ai/llama.cpp/models/gemma4/mmproj-F16.gguf`

如果你不想在 Jetson 上直接走命令行下载，也可以：

1. 在浏览器里手动下载这两个文件
2. 通过 U 盘、SCP、SFTP 或局域网共享目录传到 Jetson
3. 最终只要把文件放到 `~/ai/llama.cpp/models/gemma4/` 即可

命令行下载方式：

```bash
mkdir -p ~/ai/llama.cpp/models/gemma4

huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  --include "gemma-4-E2B-it-Q4_K_M.gguf" \
  --include "mmproj-F16.gguf" \
  --local-dir ~/ai/llama.cpp/models/gemma4
```

落地原则：

1. 主模型与 `mmproj` 必须成对匹配
2. 优先选 4-bit 量化规格
3. 在 8GB 设备上，把多模态模型视为复核引擎，而不是高并发主推理链路

***

### 18.5 步骤三：先做本机最小启动验证

先不要一上来就做服务化，先在终端里验证 `llama-server` 能成功启动。

```bash
cd ~/ai/llama.cpp/build/bin
```

建议准备两套启动参数：

#### 18.5.1 方案 A：独占验证方案

适用场景：

- 临时停掉 YOLO / ZLM / 其他高负载任务
- 只验证 `llama.cpp + GGUF` 是否能以更激进参数启动
- 不作为当前 BuildingOS 设备上的默认常驻方案

```bash
./llama-server \
  -m /home/buildingos/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /home/buildingos/ai/llama.cpp/models/gemma4/mmproj-F16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --threads 4 \
  --parallel 1 \
  --alias buildingos_review_engine
```

参数重点：

- `--ctx-size 4096`：先以 4K 上下文验证能力边界
- `--n-gpu-layers 99`：优先尝试全 GPU 层
- `--threads 4`：限制 CPU 线程数
- `--parallel 1`：固定单请求串行

#### 18.5.2 方案 B：与 `main.py` 共存方案

适用场景：

- `src/main.py` 持续运行，视频主链路不能停
- Gemma 只作为低频复核服务
- 这是当前 BuildingOS 设备上更适合作为默认起点的方案

```bash
./llama-server \
  -m /home/buildingos/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /home/buildingos/ai/llama.cpp/models/gemma4/mmproj-F16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 40 \
  --threads 4 \
  --parallel 1 \
  --alias buildingos_review_engine
```

如果要把这组参数固化为可复用启动脚本，建议保存为：

`/home/buildingos/ai/llama.cpp/scripts/llama-gemma-start.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

exec /home/buildingos/ai/llama.cpp/build/bin/llama-server \
  -m /home/buildingos/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /home/buildingos/ai/llama.cpp/models/gemma4/mmproj-F16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 40 \
  --threads 4 \
  --parallel 1 \
  --alias buildingos_review_engine
```

写入后赋予执行权限：

```bash
chmod +x /home/buildingos/ai/llama.cpp/scripts/llama-gemma-start.sh
```

参数重点：

- `--ctx-size 2048`：先压缩上下文，减少与视频链路争抢缓存
- `--n-gpu-layers 40`：先给 Gemma 保留一部分 GPU 层，而不是直接吃满
- `--threads 4`：限制 CPU 线程，避免过度挤占系统
- `--parallel 1`：先固定单请求串行，避免与 YOLO 抢资源

#### 18.5.3 当前推荐顺序

建议按下面顺序执行，而不是一开始就使用激进参数：

1. 如果 `main.py` 不停，先跑“方案 B：与 `main.py` 共存方案”
2. 如果方案 B 稳定，再尝试把 `--n-gpu-layers 40` 提高到 `60`
3. 如果仍然稳定，再把 `--ctx-size 2048` 提高到 `4096`
4. 只有在临时停掉视频主链路时，才去跑“方案 A：独占验证方案”

如果这里启动失败，再按下面顺序收敛：

1. 把 `--n-gpu-layers 40` 降到 `30`
2. 再降到 `20`
3. 如仍失败，再把 `--ctx-size 2048` 降到 `1024`

***

### 18.6 步骤四：验证服务接口

服务启动后，先确认端口可访问：

```bash
curl http://127.0.0.1:8080/health
```

再做最小文本请求验证：

```bash
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "USER: 一句话说明 Jetson Orin Nano 的优势。\nASSISTANT:",
    "n_predict": 64,
    "temperature": 0.2,
    "top_p": 0.9
  }'
```

再做最小图片请求验证：

```bash
python3 - <<'PY'
import base64
import json
import mimetypes
import urllib.request

image_path = "/home/buildingos/ai/test.jpg"

mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "buildingos_review_engine",
    "messages": [
        {
            "role": "system",
            "content": "不要输出思考过程。请简明扼要地回答，先输出总人数，然后逐一用文字描述每个人的具体方位。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": "图片中有几个人？并且用文字描述他们具体的方位。"
                }
            ]
        }
    ],
    "chat_template_kwargs": {
        "enable_thinking": False
    },
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": False
}

req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    print("final_content:", repr(result["choices"][0]["message"]["content"]))
    print("reasoning_content:", repr(result["choices"][0]["message"].get("reasoning_content", "")))
    print("finish_reason:", result["choices"][0].get("finish_reason"))
PY
```

图片验证结果的读取规则：

- 真正业务结果优先读取 `choices[0].message.content`
- `reasoning_content` 只作为调试信息，不作为业务最终输出
- 如果 `content` 为空而 `reasoning_content` 有值，应优先检查是否还在启用 thinking

通过标准：

- `llama-server` 正常启动
- `curl` 可以拿到 JSON 返回
- 没有出现服务直接退出或 OOM

***

### 18.7 步骤五：BuildingOS 业务接入方式

在业务侧，优先把 C++ 服务当作本地 HTTP 推理端点来接入。

```python
import base64
import mimetypes
import requests

def encode_image(image_path):
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{img_b64}"

def gemma_review_image(image_path, prompt):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "model": "buildingos_review_engine",
        "messages": [
            {
                "role": "system",
                "content": "不要输出思考过程。请简明扼要地回答问题，必要时描述方位。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "stream": False,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 512,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
```

业务侧仍然建议保留串行推理锁：

- 同一时刻只允许一个 Gemma 复核请求进入执行区
- YOLO 高负载时不触发 Gemma 多模态复核
- 长任务与批量任务不要走这条本地复核链路

***

### 18.8 步骤六：把服务改成常驻进程

当终端验证通过后，再转为 `systemd` 服务。

使用以下命令一键创建并写入 `systemd` 服务文件：

```bash
sudo tee /etc/systemd/system/llama-gemma.service << 'EOF' > /dev/null
[Unit]
Description=llama.cpp Gemma 4 Review Service
After=network.target

[Service]
Type=simple
User=buildingos
WorkingDirectory=/home/buildingos/ai/llama.cpp/build/bin
ExecStart=/home/buildingos/ai/llama.cpp/scripts/llama-gemma-start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

如果你还没有创建对应的执行脚本 `llama-gemma-start.sh`，也请使用以下命令一键创建并赋予执行权限：

```bash
mkdir -p /home/buildingos/ai/llama.cpp/scripts
cat << 'EOF' > /home/buildingos/ai/llama.cpp/scripts/llama-gemma-start.sh
#!/usr/bin/env bash
set -euo pipefail

exec /home/buildingos/ai/llama.cpp/build/bin/llama-server \
  -m /home/buildingos/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /home/buildingos/ai/llama.cpp/models/gemma4/mmproj-F16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 40 \
  --threads 4 \
  --parallel 1 \
  --alias buildingos_review_engine
EOF
chmod +x /home/buildingos/ai/llama.cpp/scripts/llama-gemma-start.sh
```

然后启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-gemma
sudo systemctl start llama-gemma
sudo systemctl status llama-gemma
```

查看日志：

```bash
journalctl -u llama-gemma -f
```

***

### 18.9 运行时内存清理（释放上下文）

`llama-server` 在处理多轮对话或多张图片后，可能会在分配的 `--ctx-size`（上下文窗口）内累积 K-V 缓存，导致系统内存（或显存）占用升高。

#### 方案一：通过 API 主动释放 Cache（推荐，免重启）

如果业务只做单次图片分析（无多轮对话状态依赖），可以利用 `llama.cpp` 提供的内部端点 `/slots` 主动清空上下文槽位。

只需在每次分析图片结束后，向服务端发送一次清空请求：

```bash
curl -X DELETE http://127.0.0.1:8080/slots/0
```

*说明：如果启动参数是 `--parallel 1`，则所有请求都在 Slot 0 中处理。在 Python 业务代码中，只需发起一个 `DELETE` 请求到 `http://127.0.0.1:8080/slots/0` 即可。这能立刻释放掉之前图片占用的大量上下文内存，而不需要重启整个进程。*

#### 方案二：彻底重启服务

如果发生僵死或严重泄露，可以接受瞬间的停机时间：

```bash
sudo systemctl restart llama-gemma
```

重启后，`llama-server` 会重新加载模型权重（通常很快），但之前对话积累的 K-V 缓存会被彻底清空，内存占用将回落到初始启动状态。

***

### 18.10 生产环境调优顺序

建议按下面顺序做，而不是同时改很多参数：

1. 先固定 `--parallel 1`
2. 如果 `main.py` 不停，先固定 `--ctx-size 2048`
3. 先验证文本，再验证图片
4. 再逐步调 `--n-gpu-layers`
5. 最后再考虑业务侧并发与排队

***

### 18.10 当前阶段的最终建议

对于 BuildingOS 当前这台 Jetson Orin Nano 8GB，可以把结论写清楚：

- Python / Transformers 官方路线可作为调试与诊断参考
- 生产环境不建议继续以宿主机 Python 推理作为默认运行形态
- 真正进入商用部署时，应优先准备 `llama.cpp + GGUF + 本地服务化` 路线
- 切换到 C++ 路线前，先清理 Python 进程、offload 文件、Hugging Face 缓存和虚拟环境，给模型服务留出更干净的内存与磁盘环境
