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
        try:
            # 1. 强制在请求前物理清理 Slot 缓存，防止模型受上一帧干扰
            try:
                requests.delete(self.gemma_slots_url, timeout=1.0)
            except:
                pass
                
            # 2. 图像转 Base64 (带 OpenAI 协议前缀)
            img_b64 = base64.b64encode(jpg_bytes).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{img_b64}"
            
            # 3. 构造 OpenAI 兼容的消息体 (要求输出 JSON)
            system_prompt = (
                "You are a professional image analyzer. You MUST output a JSON object ONLY. "
                "Structure: {\"result\": \"YES\" or \"NO\", \"analysis\": \"brief description\"}"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": f"{prompt}"}
                    ]
                }
            ]
            
            # 4. 组装请求体
            payload = {
                "model": "buildingos_review_engine",
                "messages": messages,
                "chat_template_kwargs": {
                    "enable_thinking": False  # 禁用思维链，直接输出 JSON
                },
                "temperature": 0.0,
                "max_tokens": 256, 
                "stream": False,
                "stop": ["<end_of_turn>", "<eos>"]
            }
            
            # 5. 发起请求
            resp = requests.post(self.gemma_url, json=payload, timeout=15.0)
            
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get('choices', [{}])[0].get('message', {})
                content = msg.get('content', '').strip()
                
                # 尝试解析 JSON
                import json
                try:
                    # 清理可能存在的 Markdown 代码块
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    res_json = json.loads(clean_content)
                    
                    raw_result = str(res_json.get("result", "UNKNOWN")).upper()
                    analysis = res_json.get("analysis", "")
                    
                    # 归一化结果
                    final_res = "UNKNOWN"
                    if "YES" in raw_result: final_res = "YES"
                    elif "NO" in raw_result: final_res = "NO"
                    
                    return {
                        "result": final_res,
                        "prompt": prompt,
                        "llm_response": content,
                        "reasoning": analysis
                    }
                except Exception as je:
                    print(f"⚠️ Gemma JSON 解析失败: {je}, Content: {content}")
                    # 备选方案
                    final_res = "UNKNOWN"
                    if "YES" in content.upper()[:50]: final_res = "YES"
                    elif "NO" in content.upper()[:50]: final_res = "NO"
                    
                    return {
                        "result": final_res,
                        "prompt": prompt,
                        "llm_response": content,
                        "reasoning": "JSON Parse Error"
                    }
            else:
                print(f"❌ Gemma API 状态码异常: {resp.status_code}")
                return { "result": "UNKNOWN", "prompt": prompt, "llm_response": f"HTTP {resp.status_code}", "reasoning": "" }
                
        except Exception as e:
            print(f"❌ Gemma 调用过程中发生崩溃: {e}")
            return { "result": "UNKNOWN", "prompt": prompt, "llm_response": str(e), "reasoning": "" }
        finally:
            try:
                requests.delete(self.gemma_slots_url, timeout=1.0)
            except:
                pass

    def submit_review(self, task_id, task_type, jpg_bytes, prompt, yolo_conf=1.0):
        """
        提交复核任务并阻塞等待结果。
        返回: dict { "result": "YES/NO/UNKNOWN", "prompt": "...", "llm_response": "...", "reasoning": "..." }
        """
        # 1. 计算优先级
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
            print(f"⚠️ Gemma 队列已满，直接拒绝任务 {task_id}")
            return { "result": "UNKNOWN", "prompt": prompt, "llm_response": "Queue Full", "reasoning": "" }

        # 4. 阻塞等待结果
        waited = task['result_event'].wait(timeout=20.0)
        
        if not waited or task['result'] is None:
            print(f"⚠️ 任务 {task_id} 等待结果超时")
            return { "result": "UNKNOWN", "prompt": prompt, "llm_response": "Timeout", "reasoning": "" }
            
        return task['result']

# 全局单例实例
gemma_queue = GemmaReviewQueue()
