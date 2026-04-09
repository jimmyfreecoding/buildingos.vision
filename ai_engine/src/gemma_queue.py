import threading
import queue
import time
import requests
import base64
import os

class GemmaReviewQueue:
    """
    文档 5.5 Gemma 复核门控与排队策略
    单例模式，全局唯一的复核队列，限制最大并发数为 1 (默认)。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(GemmaReviewQueue, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return
            
        config = config or {}
        # 强制队列并发上限
        self.concurrency = config.get("gemma_review_queue_concurrency", 1)
        # 获取 Gemma API 地址。
        # 重要：如果 AI 引擎运行在宿主机，直接访问 127.0.0.1。
        # 如果在 Docker 容器内，则使用 host.docker.internal。
        default_gemma_host = "127.0.0.1"
        if os.path.exists("/.dockerenv"):
            default_gemma_host = "host.docker.internal"
            
        # 获取基础 URL 并统一到 OpenAI 兼容的 v1/chat/completions 接口
        raw_api_url = config.get("gemma_api_url", f"http://{default_gemma_host}:8080/completion")
        self.base_url = raw_api_url.split("/completion")[0].split("/v1")[0].rstrip("/")
        
        self.gemma_url = f"{self.base_url}/v1/chat/completions"
        self.gemma_slots_url = f"{self.base_url}/slots/0"
        
        # 强制检查：如果在宿主机运行但配置里写了 host.docker.internal，则修正它
        if not os.path.exists("/.dockerenv") and "host.docker.internal" in self.base_url:
            self.base_url = self.base_url.replace("host.docker.internal", "127.0.0.1")
            self.gemma_url = f"{self.base_url}/v1/chat/completions"
            self.gemma_slots_url = f"{self.base_url}/slots/0"
        
        # 优先级队列 (PriorityQueue)，优先级数字越小越先执行
        self.task_queue = queue.PriorityQueue(maxsize=10) # 限制最大积压10个任务，超出的直接降级丢弃
        
        # 启动消费线程
        self.workers = []
        for i in range(self.concurrency):
            t = threading.Thread(target=self._worker_loop, name=f"Gemma-Worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)
            
        self._initialized = True
        print(f"✅ Gemma 复核队列初始化完成 (并发限制: {self.concurrency})")

    def _worker_loop(self):
        """后台线程不断从队列取任务去请求 Gemma"""
        while True:
            try:
                # 阻塞获取任务
                priority, timestamp, task = self.task_queue.get()
                
                # 如果任务排队太久 (比如超过了 30秒)，说明系统过载，直接丢弃该复核任务，执行降级策略
                if time.time() - timestamp > 30.0:
                    print(f"⚠️ Gemma 队列积压严重，任务 {task['id']} 已超时，执行默认降级")
                    task['result_event'].set() # 唤醒等待的线程，返回 None
                    self.task_queue.task_done()
                    continue
                    
                # 真正调用大模型
                result = self._call_gemma_api(task['jpg_bytes'], task['prompt'])
                
                # 回写结果并唤醒调用方
                task['result'] = result
                task['result_event'].set()
                
                self.task_queue.task_done()
                
            except Exception as e:
                print(f"❌ Gemma Worker 异常: {e}")
                time.sleep(1)

    def _call_gemma_api(self, jpg_bytes, prompt):
        """实际发起 HTTP 请求到本地 llama.cpp 部署的 Gemma 服务"""
        # --- 调试增强：保存现场以供复现 ---
        # 获取 debug 目录 (ai_engine/gemma_debug)
        debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gemma_debug"))
        if not os.path.exists(debug_dir):
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except:
                debug_dir = "/tmp/gemma_debug"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)
        
        # 使用毫秒级时间戳，防止极速请求时文件名冲突
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
        img_filename = f"gemma_{timestamp}.jpg"
        img_path = os.path.join(debug_dir, img_filename)
        sh_path = os.path.join(debug_dir, f"gemma_{timestamp}.sh")
        
        # 1. 保存原始图片
        try:
            with open(img_path, "wb") as f:
                f.write(jpg_bytes)
        except Exception as e:
            print(f"⚠️ 无法保存调试图片: {e}")

        try:
            # 1. 强制在请求前物理清理 Slot 缓存，防止模型受上一帧干扰
            try:
                requests.delete(self.gemma_slots_url, timeout=1.0)
            except:
                pass
                
            # 2. 图像转 Base64 (带 OpenAI 协议前缀)
            img_b64 = base64.b64encode(jpg_bytes).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{img_b64}"
            
            # 3. 构造 OpenAI 兼容的消息体
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": f"{prompt}\nConstraint: Answer ONLY 'YES' or 'NO'. No reasoning. No explanations."}
                    ]
                }
            ]
            
            # 4. 组装请求体 (使用 v1/chat/completions 接口)
            payload = {
                "model": "buildingos_review_engine",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16,
                "stream": False,
                "stop": ["<end_of_turn>", "<eos>", "model\n"]
            }
            
            # --- 调试增强：生成并保存复测脚本 ---
            import json
            # 构造用于脚本的 Payload (不带巨大的 Base64)
            payload_for_sh = payload.copy()
            payload_for_sh["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "IMAGE_URL_PLACEHOLDER"}},
                        {"type": "text", "text": payload["messages"][0]["content"][1]["text"]}
                    ]
                }
            ]
            
            payload_json_safe = json.dumps(payload_for_sh, ensure_ascii=False).replace("'", "'\\''")
            
            sh_content = f"""#!/bin/bash
# Gemma 复核手动复现脚本 (由 AI-Engine 自动生成 - OpenAI 协议版)
# 时间: {timestamp}
# 提示词: {prompt}

# 获取当前脚本所在目录
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
IMG_FILE="$DIR/{img_filename}"
GEMMA_URL="{self.gemma_url}"

echo "🚀 正在使用本地图片重新发起 Gemma 复核 (OpenAI 协议)..."
echo "📸 图片: $IMG_FILE"
echo "❓ 提示词: {prompt}"

# 使用 heredoc 传递 Python 代码，彻底避免 shell 参数过长限制
python3 - << 'EOF' "$IMG_FILE" '{payload_json_safe}' "$GEMMA_URL"
import sys, json, requests, base64

img_path = sys.argv[1]
payload = json.loads(sys.argv[2])
gemma_url = sys.argv[3]

try:
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    # 注入 OpenAI 格式的图片 URL
    payload['messages'][0]['content'][0]['image_url']['url'] = f"data:image/jpeg;base64,{{img_b64}}"
except Exception as e:
    print(f"❌ 无法读取图片文件: {{e}}")
    sys.exit(1)

try:
    print(f"\\n📡 正在请求 Gemma API: {{gemma_url}} ...")
    resp = requests.post(gemma_url, json=payload, timeout=20.0)
    print(f'\\n[Response Status] {{resp.status_code}}')
    
    if resp.status_code == 200:
        data = resp.json()
        content = data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
        print(f'[Response Content] "{{content}}"')
    else:
        print(f'[Response Body] {{resp.text}}')
except Exception as e:
    print(f'\\n[Error] {{e}}')
EOF
"""
            try:
                with open(sh_path, "w", encoding="utf-8") as f:
                    f.write(sh_content)
                try:
                    os.chmod(sh_path, 0o755)
                except:
                    pass
            except Exception as e:
                print(f"⚠️ 无法保存调试脚本: {e}")
            # -----------------------------------

            # 5. 发起请求
            resp = requests.post(self.gemma_url, json=payload, timeout=15.0)
            
            if resp.status_code == 200:
                data = resp.json()
                # OpenAI 协议返回在 choices[0].message.content
                raw_answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                
                # 兼容性兜底：有些 llama-server 版本可能还在 content 里直接返回
                if not raw_answer:
                    raw_answer = data.get("content", "").strip().upper()
                
                print(f"DEBUG: Gemma raw response: '{raw_answer}'")
                
                # --- 强力清洗逻辑 ---
                # 即使模型复读了指令或包含了 Prompt 片段，我们也只看最后的有效输出
                cleaned = raw_answer
                for stop_word in ["ANSWER:", "MODEL\n", "[IMG-1]", "INSTRUCTION", "YES OR NO"]:
                    if stop_word in cleaned:
                        cleaned = cleaned.split(stop_word)[-1].strip()
                
                # 去除前缀标点
                cleaned = cleaned.lstrip(":： ").strip()
                
                print(f"DEBUG: Gemma cleaned response: '{cleaned}' (Timestamp: {timestamp})")
                print(f"DEBUG: 现场已保存至: {img_path}")
                print(f"DEBUG: 手动复测脚本: {sh_path}")
                
                # 最终决策：寻找关键词
                if "YES" in cleaned or "是的" in cleaned or "确认" in cleaned:
                    return "YES"
                elif "NO" in cleaned or "无人" in cleaned or "没有" in cleaned:
                    return "NO"
                else:
                    # 最后的兜底：如果在整个 raw_answer 中能找到 YES/NO 也行
                    if "YES" in raw_answer: return "YES"
                    if "NO" in raw_answer: return "NO"
                    return "NO" # 默认保守处理
            else:
                print(f"❌ Gemma API 状态码异常: {resp.status_code}")
                return "UNKNOWN"
                
        except Exception as e:
            print(f"❌ Gemma 调用过程中发生崩溃: {e}")
            return "UNKNOWN"
        finally:
            try:
                requests.delete(self.gemma_slots_url, timeout=1.0)
            except:
                pass

    def submit_review(self, task_id, task_type, jpg_bytes, prompt, yolo_conf=1.0):
        """
        提交复核任务并阻塞等待结果。
        为了完全杜绝 OpenCV 多线程引发的 C++ 内存崩溃，
        这里接收的是纯 Python bytes (jpg_bytes) 而不是 numpy array。
        task_type: 'presence' 或 'smoking'
        yolo_conf: YOLO 给出的置信度，用于决定优先级
        """
        # 1. 计算优先级 (数字越小越优先)
        # 文档 5.5 优先级：
        # 1: YOLO 判定“无人”（防误关灯，这是最高优的，如果有任何可疑，马上复核）
        # 2: YOLO 边界低置信样本 (0.25 - 0.4 之间)
        # 3: 普通 Presence 确认
        # 4: Smoking 复核 (吸烟对时效性要求没那么高，可以稍微排后)
        priority = 3
        if task_type == 'presence':
            if yolo_conf < 0.2:
                priority = 1
            elif yolo_conf < 0.4:
                priority = 2
        elif task_type == 'smoking':
            priority = 4

        # 2. 组装任务对象
        task = {
            'id': task_id,
            'jpg_bytes': jpg_bytes,
            'prompt': prompt,
            'result': None,
            'result_event': threading.Event()
        }

        # 3. 尝试放入队列
        try:
            self.task_queue.put_nowait((priority, time.time(), task))
        except queue.Full:
            print(f"⚠️ Gemma 队列已满，直接拒绝任务 {task_id}，执行降级")
            # 降级规则 (文档 7.2)：
            # Presence 降级按“有人”处理 (YES)；Smoking 降级按“不告警”处理 (UNKNOWN/NO)
            return "YES" if task_type == 'presence' else "UNKNOWN"

        # 4. 阻塞等待结果 (最多等 20 秒)
        # print(f"⏳ 任务 {task_id} 排队中 (优先级 {priority})...")
        waited = task['result_event'].wait(timeout=20.0)
        
        if not waited or task['result'] is None:
            print(f"⚠️ 任务 {task_id} 等待结果超时，执行降级")
            return "YES" if task_type == 'presence' else "UNKNOWN"
            
        return task['result']

# 全局单例实例
gemma_queue = GemmaReviewQueue()
