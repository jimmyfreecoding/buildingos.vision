import cv2
import threading
import time
import os
import json
import urllib.request
import urllib.parse
from datetime import datetime
import paho.mqtt.client as mqtt

# 导入我们新写的双轨制底层驱动与业务大脑
from yolo_infer import YoloTensorRTEngine
from state_machine import PresenceStateMachine, SmokingStateMachine
from gemma_queue import gemma_queue

# --- Load Configuration ---
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/ai_engine/config/config.json")
DEFAULT_CONFIG_PATH = "/app/ai_engine/config/config.default.json"

def load_config():
    # --- 自动初始化配置文件机制 ---
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: {CONFIG_PATH} not found. Initializing from default config...")
        try:
            import shutil
            if os.path.exists(DEFAULT_CONFIG_PATH):
                shutil.copy(DEFAULT_CONFIG_PATH, CONFIG_PATH)
                print(f"Successfully copied {DEFAULT_CONFIG_PATH} to {CONFIG_PATH}")
            else:
                # 连 default 都找不到时的保底方案
                with open(CONFIG_PATH, 'w') as f:
                    json.dump({"streams": {"occupancy": [], "smoking": []}}, f, indent=4)
                print(f"Created empty config at {CONFIG_PATH}")
        except Exception as e:
            print(f"Error initializing config file: {e}")
            
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

config = load_config()

# --- Unified Camera Config Parser ---
def get_unified_cameras(cfg):
    """将旧版 streams.smoking 和 streams.occupancy 配置统一转换为 cameras 字典"""
    cameras = {}
    streams = cfg.get("streams", {})
    
    # 解析人员感知流 (Presence)
    for cam in streams.get("occupancy", []):
        cam_id = cam.get("id")
        if not cam_id: continue
        cameras[cam_id] = {
            "source_url": cam.get("source_url"),
            "url": cam.get("url"), # 这是内部 zlm 的代理地址
            "areaCode": cam.get("areaCode", "UNKNOWN"),
            "enabled": True,
            "tasks": ["presence"]
        }
        
    # 解析吸烟检测流 (Smoking)
    for cam in streams.get("smoking", []):
        cam_id = cam.get("id")
        if not cam_id: continue
        if cam_id in cameras:
            cameras[cam_id]["tasks"].append("smoking")
            if not cameras[cam_id].get("areaCode") or cameras[cam_id].get("areaCode") == "UNKNOWN":
                cameras[cam_id]["areaCode"] = cam.get("areaCode", "UNKNOWN")
        else:
            cameras[cam_id] = {
                "source_url": cam.get("source_url"),
                "url": cam.get("url"), # 这是内部 zlm 的代理地址
                "areaCode": cam.get("areaCode", "UNKNOWN"),
                "enabled": True,
                "tasks": ["smoking"]
            }
    return cameras

camera_config = get_unified_cameras(config)

# AI Engine specific configuration (与 ZLM 共用)
ai_config = config.get("ai_engine", {})
ZLM_HTTP_PORT = ai_config.get("zlm_http_port", 10081)
ZLM_RTSP_PORT = ai_config.get("zlm_rtsp_port", 10554)

# 从环境变量获取 API Secret，ZLM_API_SECRET 已经在 docker-compose 中统一定义
ZLM_API_SECRET = os.getenv("ZLM_API_SECRET", "buildingos_edge_secret_2026")

# --- Global Dictionaries ---
presence_machines = {} # 存储每个摄像头的 Presence 状态机
smoking_machines = {}  # 存储每个摄像头的 Smoking 状态机
mqtt_cooldowns = {}    # MQTT 发送冷却时间戳 (去重键)

# 全局模型占位符 (延迟加载)
pose_model = None
smoking_model = None

# --- Init TensorRT Engines ---
def init_tensorrt_models():
    global pose_model, smoking_model
    if pose_model is not None and smoking_model is not None:
        return
        
    print("Initializing TensorRT Models (yolo_infer)...")
    try:
        pose_model = YoloTensorRTEngine("/app/models/yolo26m-pose.engine", conf_thres=0.25)
        smoking_model = YoloTensorRTEngine("/app/models/smoking_26m.engine", conf_thres=0.3)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load TensorRT engines: {e}")
        print("Did you compile them to /app/models/yolo26m-pose.engine and smoking_26m.engine ?")

# --- MQTT Setup ---
MQTT_BROKER = config.get("mqtt", {}).get("broker", "127.0.0.1")
MQTT_PORT = config.get("mqtt", {}).get("port", 1883)
MQTT_KEEPALIVE = 60

try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    # 兼容老版本 paho-mqtt
    mqtt_client = mqtt.Client()

try:
    print(f"Connecting to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}...")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE)
    mqtt_client.loop_start()
    print("MQTT Connected.")
except Exception as e:
    print(f"Error connecting to MQTT: {e}")
    print("WARNING: MQTT connection failed, but AI Engine will continue to run without publishing events.")

def publish_mqtt_event(cam_id, area_code, event_type, payload, frame=None):
    """带冷却去重机制的 MQTT 发布，同时持久化到本地日志供 Web 查阅"""
    # 强制去重键: areaCode/camera + eventType
    dedup_key = f"{area_code}_{cam_id}_{event_type}"
    now = time.time()
    
    # 强制冷却时间：180 秒 (3分钟)
    cooldown = config.get("mqtt_alert_cooldown_seconds", 180)
    
    if dedup_key in mqtt_cooldowns:
        if now - mqtt_cooldowns[dedup_key] < cooldown:
            print(f"[{cam_id}] MQTT {event_type} 在 {cooldown}s 冷却期内，跳过发送")
            return
            
    # 执行发送
    topic = "buildingos/presence/result" if event_type == "presence" else "buildingos/smoking/alert"
    try:
        mqtt_client.publish(topic, json.dumps(payload))
        mqtt_cooldowns[dedup_key] = now
        print(f"[{cam_id}] => MQTT 已发布 {topic}: {payload['result']}")
        
        # --- 本地持久化 (供 Web 界面场景检测结果展示) ---
        try:
            log_dir_base = config.get("storage_quota", {}).get("occupancy_log_dir", "/app/www/occupancy_logs")
            today_str = datetime.now().strftime("%Y-%m-%d")
            # 清理 area_code 中的非法路径字符
            safe_area = str(area_code).replace('/', '_').replace('\\', '_')
            target_dir = os.path.join(log_dir_base, today_str, safe_area)
            os.makedirs(target_dir, exist_ok=True)
            
            timestamp_ms = int(now * 1000)
            
            # 1. 保存截图
            image_path = ""
            if frame is not None:
                img_name = f"{cam_id}_{event_type}_{timestamp_ms}.jpg"
                img_full_path = os.path.join(target_dir, img_name)
                cv2.imwrite(img_full_path, frame)
                # 记录相对路径，Web 端会自动拼接 ZLM 端口
                image_path = f"occupancy_logs/{today_str}/{safe_area}/{img_name}"
            
            # 2. 构造日志结构 (兼容旧版 UI)
            log_entry = {
                "event": "Smoking Alert" if event_type == "smoking" else "Presence Update",
                "areaCode": area_code,
                "is_occupied": payload.get("result") == "occupied" or payload.get("result") == "confirmed_smoking",
                "person_count": 1 if payload.get("result") in ["occupied", "confirmed_smoking"] else 0,
                "timestamp": payload.get("timestamp"),
                "scores": {
                    "visual": payload.get("windowMinutes", 0),
                    "total": 1.0,
                    "time_bias": payload.get("sampleIntervalSeconds", 0)
                },
                "threshold_used": "Gemma E2B Verified",
                "images": [image_path] if image_path else [],
                "raw_payload": payload
            }
            
            # 3. 保存 JSON
            json_path = os.path.join(target_dir, f"{cam_id}_{event_type}_{timestamp_ms}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"[{cam_id}] 本地日志持久化失败: {e}")

    except Exception as e:
        print(f"[{cam_id}] MQTT 发送失败: {e}")

# --- Camera Processing Thread (每 60s/20s 抓拍) ---
def process_camera(cam_id, cam_info):
    rtsp_url = cam_info.get("url")
    area_code = cam_info.get("areaCode", "UNKNOWN")
    enabled = cam_info.get("enabled", True)

    if not enabled:
        print(f"[{cam_id}] Camera is disabled in config.")
        return

    print(f"[{cam_id}] Starting timer-based inference thread...")

    # 初始化两套状态机
    if cam_id not in presence_machines:
        presence_machines[cam_id] = PresenceStateMachine(cam_id, config)
    if cam_id not in smoking_machines:
        smoking_machines[cam_id] = SmokingStateMachine(cam_id, config)
        
    p_sm = presence_machines[cam_id]
    s_sm = smoking_machines[cam_id]
    
    tasks = cam_info.get("tasks", [])
    has_presence_task = "presence" in tasks
    has_smoking_task = "smoking" in tasks

    # 抓拍间隔配置
    p_interval = config.get("presence_sample_interval_seconds", 60)
    s_interval = config.get("smoke_sample_interval_seconds", 20)
    
    # 我们用一个主循环，每秒检查一次是否到了该抓拍的时间
    last_p_time = 0
    last_s_time = 0

    # 尝试连接视频流
    # 我们应该等 ZLM 代理配置生效后再去连接，否则一启动马上连接会报 404
    # 在主循环里处理连接
    cap = cv2.VideoCapture()
    
    while True:
        try:
            now = time.time()
            need_p_sample = has_presence_task and (now - last_p_time) >= p_interval
            
            # Smoking 仅在小窗口激活时才执行采样 (如果没有开启 Presence，默认持续激活 Smoking，因为没有门控)
            if has_presence_task:
                is_smoke_active = s_sm.check_window_active()
            else:
                is_smoke_active = True
                
            need_s_sample = has_smoking_task and is_smoke_active and ((now - last_s_time) >= s_interval)

            if need_p_sample or need_s_sample:
                # 如果连接断开，尝试重连
                if not cap.isOpened():
                    # 取消硬编码 cv2.CAP_FFMPEG 后端。
                    # l4t-ml OpenCV 版本支持 GStreamer，FFMPEG在处理网络流名字时可能会报错 "can't be used to capture by name"
                    cap.open(rtsp_url) 
                    if not cap.isOpened():
                        print(f"[{cam_id}] 无法连接到视频流: {rtsp_url}")
                        time.sleep(5)
                        continue
                
                # 读取一帧 (抛弃缓存中堆积的旧帧，获取最新)
                # cv2.CAP_PROP_BUFFERSIZE = 1 在某些后端无效，我们连读几帧清空 buffer
                try:
                    for _ in range(3): 
                        ret, frame = cap.read()
                except Exception as e:
                    print(f"[{cam_id}] cap.read() 崩溃: {e}")
                    ret = False
                    frame = None
                
                if not ret or frame is None:
                    print(f"[{cam_id}] 读取视频流失败，准备重连...")
                    cap.release()
                    continue
                
                # --- 延迟加载模型 ---
                # 在确保流能读出第一帧之后，才去初始化 TensorRT 模型
                # 避免 OpenCV 的 GStreamer 初始化与 TensorRT C++ 引擎抢占底层内存指针导致 double free
                init_tensorrt_models()
                
                # --- 1. Presence (人员存在) 综合判定流程 ---
                if need_p_sample and pose_model:
                    last_p_time = now
                    
                    # YOLO 一级判定
                    boxes = pose_model.predict(frame)
                    
                    has_person = False
                    if len(boxes) > 0:
                        # 有候选框，送 Gemma 二级复核
                        # 获取最高置信度用于队列优先级
                        max_conf = max([b['conf'] for b in boxes])
                        
                        # 构造 Gemma Prompt
                        prompt = "这幅办公场景图像中，是否有真实的、活着的人？请回答 YES 或 NO。"
                        
                        # 为了防止 cv2 在多个线程间争抢内存引发 corrupted size vs. prev_size，必须复制一份独立的内存
                        frame_copy = frame.copy()
                        
                        # 提交复核 (阻塞等待)
                        gemma_res = gemma_queue.submit_review(f"{cam_id}_P_{now}", "presence", frame_copy, prompt, yolo_conf=max_conf)
                        
                        if gemma_res == "YES":
                            has_person = True
                            print(f"[{cam_id}] Presence: Gemma 确认有人 (YOLO框: {len(boxes)}个)")
                        else:
                            print(f"[{cam_id}] Presence: Gemma 否决了 YOLO 的判定 (误报过滤)")
                    
                    # 无论有没有人，送入状态机处理时间窗口
                    event_triggered, final_status, window_mins, time_period = p_sm.update(has_person_this_frame=has_person)
                    
                    # 如果状态机决定收敛，触发 MQTT
                    if event_triggered:
                        payload = {
                            "areaCode": area_code,
                            "result": final_status, # occupied / empty
                            "windowMinutes": window_mins,
                            "timePeriod": time_period,
                            "source": "yolo26m+gemma",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # 由于 MQTT 和日志保存需要一帧图像，如果本次有框就用当前帧，没有就不传或者传空
                        publish_mqtt_event(cam_id, area_code, "presence", payload, frame if has_person else None)
                        
                        # 如果确认为有人闯入，激活吸烟小窗口
                        if final_status == "occupied":
                            s_sm.trigger_presence()

                # --- 2. Smoking (吸烟检测) 综合判定流程 ---
                if need_s_sample and smoking_model:
                    last_s_time = now
                    
                    # YOLO 一级判定
                    boxes = smoking_model.predict(frame)
                    
                    if len(boxes) > 0:
                        # 发现可疑吸烟动作
                        max_conf = max([b['conf'] for b in boxes])
                        prompt = "这幅图像中，是否有人正在抽烟？(包括拿着烟、嘴里叼着烟、吐烟圈)。请排除吃东西、喝水、拿笔或托腮等动作。请回答 YES 或 NO。"
                        
                        frame_copy = frame.copy()
                        gemma_res = gemma_queue.submit_review(f"{cam_id}_S_{now}", "smoking", frame_copy, prompt, yolo_conf=max_conf)
                        
                        if gemma_res == "YES":
                            # Gemma 确认吸烟
                            alert_triggered = s_sm.confirm_smoke()
                            
                            if alert_triggered:
                                # 保存一张截图作为证据 (可选)
                                evidence_url = "http://buildingos.local/placeholder.jpg" # 实际应上传 OSS
                                
                                payload = {
                                    "cameraId": cam_id,
                                    "areaCode": area_code,
                                    "result": "confirmed_smoking",
                                    "windowMinutes": config.get("smoke_window_minutes", 2),
                                    "sampleIntervalSeconds": s_interval,
                                    "source": "yolo26m+gemma",
                                    "evidenceImageUrl": evidence_url,
                                    "timestamp": datetime.now().isoformat()
                                }
                                publish_mqtt_event(cam_id, area_code, "smoking", payload, frame)

            # 休眠 1 秒，防止死循环跑满 CPU
            time.sleep(1)

        except Exception as e:
            print(f"[{cam_id}] 线程发生严重异常: {e}")
            time.sleep(5)

# --- ZLMediaKit Auto-Proxy Setup ---
def register_cameras_to_zlm():
    print("Waiting for ZLMediaKit to start...")
    # 把等待时间拉长，因为容器启动有先后，ZLM 可能还没就绪
    time.sleep(10) 
    
    for cam_id, cam_info in camera_config.items():
        # 这里必须使用摄像头的原始 RTSP 物理地址注册到 ZLM 中
        rtsp_source = cam_info.get("source_url")
        if not rtsp_source:
            # 兼容处理：尝试从 config 原始数据里捞
            for stream_type in ["smoking", "occupancy"]:
                for stream in config.get("streams", {}).get(stream_type, []):
                    if stream.get("id") == cam_id:
                        rtsp_source = stream.get("source_url")
                        break
                if rtsp_source: break
        
        if not rtsp_source:
            print(f"[{cam_id}] Cannot find physical source_url for ZLM proxy. Skipping.")
            continue
            
        enabled = cam_info.get("enabled", True)
        
        if not enabled:
            continue
            
        # 注意：这里调用的是宿主机或者 Docker 内部网络名
        # 因为我们是在 docker-compose 里跑的，所以使用容器名 `zlm` 作为 HTTP 地址，而不是 127.0.0.1
        api_url = f"http://zlm:80/index/api/addStreamProxy"
        params = {
            "secret": ZLM_API_SECRET,
            "vhost": "__defaultVhost__",
            "app": "live",
            "stream": cam_id,
            "url": rtsp_source,
            "enable_rtmp": 1,
            "enable_rtsp": 1,
            "enable_hls": 1,
            "enable_mp4": 0
        }
        query_string = urllib.parse.urlencode(params)
        full_url = f"{api_url}?{query_string}"
        
        try:
            req = urllib.request.Request(full_url, method="POST")
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode())
                if res_data.get("code") == 0:
                    print(f"[{cam_id}] ZLM Proxy configured. Live at rtsp://zlm:554/live/{cam_id}")
                else:
                    print(f"[{cam_id}] ZLM Proxy failed: {res_data.get('msg')}")
        except Exception as e:
            print(f"[{cam_id}] Could not connect to ZLM API ({api_url}): {e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    print("Starting AI Engine (Dual-Stage Architecture)...")
    
    # 注册 ZLM 代理 (项目核心记忆: 动态拉流)
    zlm_thread = threading.Thread(target=register_cameras_to_zlm)
    zlm_thread.start()

    # 启动摄像头定时采样线程
    threads = []
    for cam_id, cam_info in camera_config.items():
        t = threading.Thread(target=process_camera, args=(cam_id, cam_info))
        t.start()
        threads.append(t)

    # 保持主线程运行
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down AI Engine...")
        mqtt_client.loop_stop()
        os._exit(0)
