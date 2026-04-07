import threading
import queue
import time
import requests
import base64

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
        # 重要：因为 AI 引擎运行在 Docker 容器内，如果 Gemma 运行在宿主机的 8080 端口，
        # 不能使用 127.0.0.1 (它指向容器自己内部)，而应该使用 host.docker.internal 
        self.gemma_url = config.get("gemma_api_url", "http://host.docker.internal:8080/completion")
        self.gemma_slots_url = self.gemma_url.replace("/completion", "/slots/0")
        
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
            # 1. 图像转 Base64
            img_b64 = base64.b64encode(jpg_bytes).decode('utf-8')
            
            # 2. 组装 llama.cpp completion API 的请求体
            payload = {
                "prompt": f"[img-1]{prompt}",
                "image_data": [{"id": 1, "data": img_b64}],
                "temperature": 0.1,  # 极低温度，追求确定性回答
                "n_predict": 64,     # 限制输出长度，我们只需要简短的 YES/NO/UNKNOWN
                "stream": False
            }
            
            # 3. 发起请求
            # print(f"🚀 发送请求至 Gemma: {prompt[:30]}...")
            start_t = time.time()
            resp = requests.post(self.gemma_url, json=payload, timeout=15.0)
            
            if resp.status_code == 200:
                answer = resp.json().get("content", "").strip().upper()
                # print(f"⏱️ Gemma 耗时 {time.time()-start_t:.1f}s, 原始返回: {answer}")
                
                # 解析标准化结果
                if "YES" in answer or "确认" in answer or "TRUE" in answer:
                    return "YES"
                elif "NO" in answer or "不" in answer or "FALSE" in answer:
                    return "NO"
                else:
                    return "UNKNOWN"
                    
            else:
                print(f"❌ Gemma API 错误: {resp.status_code}")
                return "UNKNOWN"
                
        except requests.exceptions.Timeout:
            print("❌ Gemma 请求超时 (>15s)")
            return "UNKNOWN"
        except Exception as e:
            print(f"❌ Gemma 调用失败: {e}")
            return "UNKNOWN"
        finally:
            # 释放上下文缓存机制 (项目记忆 01KNFKT7RT1JDQF7YPEB9XR8GT)
            try:
                requests.delete(self.gemma_slots_url, timeout=2.0)
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
