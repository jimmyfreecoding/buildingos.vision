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
            
        self.gemma_url = config.get("gemma_api_url", f"http://{default_gemma_host}:8080/completion")
        self.gemma_slots_url = self.gemma_url.replace("/completion", "/slots/0")
        
        # 强制检查：如果在宿主机运行但配置里写了 host.docker.internal，则修正它
        if not os.path.exists("/.dockerenv") and "host.docker.internal" in self.gemma_url:
            self.gemma_url = self.gemma_url.replace("host.docker.internal", "127.0.0.1")
            self.gemma_slots_url = self.gemma_slots_url.replace("host.docker.internal", "127.0.0.1")
        
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
        try:
            # 1. 强制在请求前物理清理 Slot 缓存，防止模型受上一帧干扰
            try:
                requests.delete(self.gemma_slots_url, timeout=1.0)
            except:
                pass
                
            # 2. 图像转 Base64
            img_b64 = base64.b64encode(jpg_bytes).decode('utf-8')
            
            # 3. 极简指令：采用 Completion 引导模式，强制关闭推理
            # 这种格式在 llama.cpp 中比 Chat 格式更能逼迫模型直接给出答案
            引导式提示词 = (
                f"<start_of_turn>user\n"
                f"Image: [img-1]\n"
                f"Instruction: {prompt}\n"
                f"Constraint: Answer ONLY 'YES' or 'NO'. No reasoning. No explanations.\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"Answer: "
            )
            
            # 4. 组装请求体：极致精简参数以关闭推理模式并提速
            payload = {
                "prompt": 引导式提示词,
                "image_data": [{"id": 1, "data": img_b64}],
                "temperature": 0.0,   # 绝对确定性
                "n_predict": 5,       # 极短生成
                "stream": False,
                "cache_prompt": False,
                "echo": False,        # 关键：禁止回显 Prompt
                "include_reasoning": False,  # 【核心修改】显式关闭推理/思维链模式
                "stop": ["<end_of_turn>", "user", "model", "Answer:"] 
            }
            
            # 5. 发起请求
            resp = requests.post(self.gemma_url, json=payload, timeout=10.0)
            
            if resp.status_code == 200:
                data = resp.json()
                # 优先获取 content，这是响应的核心部分
                raw_answer = data.get("content", "").strip().upper()
                
                # 兼容性处理
                if not raw_answer and "choices" in data:
                    raw_answer = data["choices"][0].get("text", "").strip().upper()
                
                print(f"DEBUG: Gemma raw response: '{raw_answer}'")
                
                # --- 强力清洗逻辑 ---
                # 即使模型复读了指令或包含了 Prompt 片段，我们也只看最后的有效输出
                cleaned = raw_answer
                for stop_word in ["ANSWER:", "MODEL\n", "[IMG-1]", "INSTRUCTION", "YES OR NO"]:
                    if stop_word in cleaned:
                        cleaned = cleaned.split(stop_word)[-1].strip()
                
                # 去除前缀标点
                cleaned = cleaned.lstrip(":： ").strip()
                
                print(f"DEBUG: Gemma cleaned response: '{cleaned}'")
                
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
