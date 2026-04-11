import cv2
import threading
import time
import os
import json
import urllib.request
import urllib.parse
import base64
import numpy as np
from datetime import datetime
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify

# 导入我们新写的双轨制底层驱动与业务大脑
from yolo_infer import YoloTensorRTEngine
from rfdetr_trt_infer import RFDETRTensorRTEngine
from state_machine import PresenceStateMachine, SmokingStateMachine
from gemma_queue import gemma_queue
import paho.mqtt.client as mqtt

# --- Flask App for Single Image Test ---
flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def api_predict():
    """
    接收 Base64 图片和参数，执行 AI 推理并返回结果。
    用于前端“测试图”功能，支持实时调整参数。
    """
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # 1. 解码图片
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # 2. 获取参数
        conf_thres = data.get('conf_thres')
        if conf_thres is not None:
            conf_thres = float(conf_thres)
            
        # 3. 执行推理 (确保模型已初始化)
        init_tensorrt_models()
        
        # 默认使用人员感知模型 (RF-DETR 或 YOLO)
        results = []
        if pose_model:
            results = pose_model.predict(frame, conf_thres=conf_thres)
            
        # 4. 绘制结果图 (用于直观观测)
        annotated_frame = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            conf = res['conf']
            cls_id = res['class_id']
            # 获取类别名
            cls_name = "person" if cls_id == 0 else f"cls_{cls_id}"
            if hasattr(pose_model, 'classes') and cls_id < len(pose_model.classes):
                cls_name = pose_model.classes[cls_id]
                
            color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 5. 编码结果图为 Base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "results": results,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "detector_source": presence_detector_source
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def run_flask():
    log_info("Starting Flask API server for AI testing on port 5000...")
    flask_app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# --- Environment Detection & Path/URL Translation Helpers ---
def is_in_container():
    return os.path.exists("/.dockerenv")

def get_real_path(p):
    """
    自适应路径转换：
    如果检测到不在容器内运行 (没有 /.dockerenv)，
    则将容器内的标准路径 /app/... 映射到宿主机的物理路径。
    """
    if is_in_container():
        return p
    
    # 获取宿主机项目根目录 (假设在 ~/buildingos.vision)
    home = os.path.expanduser("~")
    project_root = os.path.join(home, "buildingos.vision")
    
    if p.startswith("/app/www"):
        return p.replace("/app/www", os.path.join(project_root, "zlm/www"))
    if p.startswith("/app/models"):
        return p.replace("/app/models", os.path.join(project_root, "ai_engine/models"))
    if p.startswith("/app/ai_engine/config"):
        return p.replace("/app/ai_engine/config", os.path.join(project_root, "ai_engine/config"))
    if p.startswith("/app/"):
        # 通用映射：将容器根目录映射到宿主机的 ai_engine 目录
        return p.replace("/app/", os.path.join(project_root, "ai_engine/"))
    return p

def get_real_url(url, zlm_http_port=10081):
    """
    自适应 URL 转换：
    如果在宿主机运行，将容器名 'zlm:80' 替换为 '127.0.0.1:10081'。
    """
    if is_in_container():
        return url
    
    # 替换 API URL
    if "zlm:80" in url:
        return url.replace("zlm:80", f"127.0.0.1:{zlm_http_port}")
    
    # 额外处理 rtsp 地址
    if "rtsp://zlm:554" in url:
        # 这里的 10554 是宿主机映射出的端口
        return url.replace("zlm:554", "127.0.0.1:10554")
        
    return url

# --- Load Configuration ---
CONFIG_PATH = os.getenv("CONFIG_PATH", get_real_path("/app/ai_engine/config/config.json"))
DEFAULT_CONFIG_PATH = get_real_path("/app/ai_engine/config/config.default.json")

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
            "url": get_real_url(cam.get("url"), config.get("ai_engine", {}).get("zlm_http_port", 10081)), # 需自适应
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
                "url": get_real_url(cam.get("url"), config.get("ai_engine", {}).get("zlm_http_port", 10081)), # 需自适应
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

def log_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --- Global Dictionaries ---
presence_machines = {} # 存储每个摄像头的 Presence 状态机
smoking_machines = {}  # 存储每个摄像头的 Smoking 状态机
mqtt_cooldowns = {}    # MQTT 发送冷却时间戳 (去重键)

# 全局模型占位符 (延迟加载)
pose_model = None
smoking_model = None
presence_detector_source = "detector"

# 确保 TensorRT 初始化的锁，防止多个摄像头线程同时触发初始化
trt_init_lock = threading.Lock()

# --- Init TensorRT Engines ---
def init_tensorrt_models():
    global pose_model, smoking_model, presence_detector_source
    
    with trt_init_lock:
        if pose_model is not None and smoking_model is not None:
            return
            
        print("Initializing detection models...")
        try:
            detector_cfg = ai_config.get("detector", {})
            presence_backend = detector_cfg.get("presence_backend", "yolo").lower()
            presence_conf = float(detector_cfg.get("presence_conf", 0.25))
            fallback_yolo_path = get_real_path(detector_cfg.get("fallback_yolo_engine_path", "/app/models/yolo26m-pose.engine"))
            if presence_backend == "rfdetr_trt":
                try:
                    presence_engine_path = get_real_path(detector_cfg.get("presence_engine_path", "/app/models/rf-detr-fp16-576.engine"))
                    person_class_id = int(detector_cfg.get("person_class_id", 0))
                    max_det = int(detector_cfg.get("max_det", 100))
                    pose_model = RFDETRTensorRTEngine(
                        presence_engine_path,
                        conf_thres=presence_conf,
                        person_class_id=person_class_id,
                        max_det=max_det
                    )
                    presence_detector_source = "rf-detr"
                except Exception as e:
                    print(f"RF-DETR init failed, fallback to YOLO: {e}")
                    pose_model = YoloTensorRTEngine(fallback_yolo_path, conf_thres=presence_conf)
                    presence_detector_source = "yolo26m"
            else:
                presence_engine_path = get_real_path(detector_cfg.get("presence_engine_path", fallback_yolo_path))
                pose_model = YoloTensorRTEngine(presence_engine_path, conf_thres=presence_conf)
                presence_detector_source = "yolo26m"

            smoking_engine_path = get_real_path(detector_cfg.get("smoking_engine_path", "/app/models/smoking_26m.engine"))
            smoking_conf = float(detector_cfg.get("smoking_conf", 0.3))
            smoking_model = YoloTensorRTEngine(smoking_engine_path, conf_thres=smoking_conf)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Failed to load TensorRT engines: {e}")
            print("Please check detector config and engine files")

# --- MQTT Setup ---
MQTT_BROKER = config.get("mqtt", {}).get("broker", "127.0.0.1")
# 如果在宿主机运行，且 broker 依然是容器名，则强制修正为 127.0.0.1
if not is_in_container() and "buildingos-emqx-prod" in MQTT_BROKER:
    MQTT_BROKER = "127.0.0.1"

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
    print("✅ MQTT Connected.")
except Exception as e:
    print(f"❌ Error connecting to MQTT: {e}")
    if "[Errno 111]" in str(e):
        print("   💡 TIP: Connection refused. This usually means the MQTT Broker (EMQX/Mosquitto) is not running.")
        print("   💡 Try starting it: 'sudo systemctl start emqx' or 'sudo docker compose up -d' if using Docker.")
    print("   WARNING: AI Engine will continue to run without publishing events.")

def save_minute_log_for_frontend(cam_id, area_code, has_person, raw_payload=None, images=None, decision_chain=None, yolo_count=0, gemma_details=None):
    """
    不管是否触发 MQTT 报警，每一分钟（或每一个采样周期）都将原始判定结果
    追加保存到本地的 JSON 中，以保证前端的 Heatmap 有细粒度的数据点。
    """
    if not cam_id or not area_code or area_code == "UNKNOWN":
        log_info(f"⚠️ 跳过保存无效日志: cam_id='{cam_id}', area_code='{area_code}'")
        return

    try:
        log_dir_base = get_real_path(config.get("storage_quota", {}).get("occupancy_log_dir", "/app/www/occupancy_logs"))
        today_str = datetime.now().strftime("%Y-%m-%d")
        safe_area = str(area_code).replace('/', '_').replace('\\', '_')
        target_dir = os.path.join(log_dir_base, today_str, safe_area)
        os.makedirs(target_dir, exist_ok=True)
        
        # 强制填充 decision_chain，防止前端显示“无日志”
        if not decision_chain:
            decision_chain = ["AI 引擎默认状态更新"]
        
        # 写入图片
        image_paths = []
        timestamp_ms = int(time.time() * 1000)
        
        # 修复 numpy truth value ambiguous 报错: 不要直接使用 `if images:`
        if isinstance(images, list) and len(images) > 0:
            for i, img in enumerate(images):
                if img is not None:
                    img_name = f"{cam_id}_sample_{timestamp_ms}_{i}.jpg"
                    img_path = os.path.join(target_dir, img_name)
                    cv2.imwrite(img_path, img)
                    rel_path = f"occupancy_logs/{today_str}/{safe_area}/{img_name}"
                    image_paths.append(rel_path)
        elif images is not None: # 直接用 is not None 检查 numpy array
            img_name = f"{cam_id}_sample_{timestamp_ms}.jpg"
            img_path = os.path.join(target_dir, img_name)
            cv2.imwrite(img_path, images)
            rel_path = f"occupancy_logs/{today_str}/{safe_area}/{img_name}"
            image_paths.append(rel_path)

        log_entry = {
            "id": f"{cam_id}_{timestamp_ms}",
            "date": today_str,
            "timestamp": datetime.now().isoformat(),
            "camera_id": cam_id,
            "areaCode": area_code,
            "event": "Presence Sample",
            "threshold_used": "1-minute sample",
            "images": image_paths,
            "raw_payload": raw_payload or {
                "result": "occupied" if has_person else "empty",
                "source": f"{presence_detector_source}+gemma",
                "decision_chain": decision_chain,
                "yolo_count": yolo_count
            }
        }
        
        if gemma_details:
            log_entry["gemma"] = gemma_details
        
        json_path = os.path.join(target_dir, f"{cam_id}_sample_{timestamp_ms}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        log_info(f"❌ 保存前端日志失败: {e}")

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
            log_dir_base = get_real_path(config.get("storage_quota", {}).get("occupancy_log_dir", "/app/www/occupancy_logs"))
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
import subprocess
import random

# --- Host FFmpeg Snapshot Helper ---
def get_frame_from_host_ffmpeg(cam_id):
    """
    直接在宿主机调用 ffmpeg 进程抓取 ZLM 转发的 RTSP 流。
    这种方式无状态、无缓存，且强制使用 TCP 传输，能保证 100% 画面完整。
    """
    # 宿主机上 ZLM 转发的 RTSP 地址
    local_rtsp_url = f"rtsp://127.0.0.1:{ZLM_RTSP_PORT}/live/{cam_id}"
    tmp_snap_path = f"/tmp/snap_{cam_id}_{int(time.time())}.jpg"
    
    # 构造 ffmpeg 命令
    # -rtsp_transport tcp: 强制使用 TCP，防止 UDP 丢包导致花屏
    # -y: 覆盖输出文件
    # -i: 输入流
    # -frames:v 1: 只截取一帧
    # -f image2: 输出格式为图片
    # 注意：FFmpeg 必须已安装在宿主机 PATH 中
    cmd = [
        "ffmpeg", 
        "-rtsp_transport", "tcp", 
        "-y", 
        "-i", local_rtsp_url, 
        "-frames:v", "1", 
        "-f", "image2", 
        tmp_snap_path
    ]
    
    try:
        # 执行抓拍，设置 15 秒超时防止卡死
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        
        if result.returncode == 0 and os.path.exists(tmp_snap_path):
            frame = cv2.imread(tmp_snap_path)
            # 及时清理临时文件
            try:
                os.remove(tmp_snap_path)
            except:
                pass
            return frame
        else:
            log_info(f"[{cam_id}] FFmpeg 抓拍失败: {result.stderr.decode() if result.stderr else 'Unknown error'}")
            return None
    except subprocess.TimeoutExpired:
        log_info(f"[{cam_id}] FFmpeg 抓拍超时 (15s)")
        return None
    except Exception as e:
        log_info(f"[{cam_id}] FFmpeg 抓拍异常: {e}")
        return None

def process_camera(cam_id, cam_info):
    """
    针对每个摄像头运行的独立采样线程。
    核心逻辑：使用宿主机 FFmpeg 进行定时无状态抓拍。
    """
    global presence_detector_source
    
    area_code = cam_info.get("areaCode", "UNKNOWN")
    enabled = cam_info.get("enabled", True)

    if not enabled:
        log_info(f"[{cam_id}] Camera is disabled in config.")
        return

    log_info(f"[{cam_id}] Starting host-ffmpeg sampling thread...")

    # 初始化状态机
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
    
    # 错峰采样延迟，防止并发调用 ffmpeg 进程导致 CPU 瞬间爆表
    stagger_delay = random.uniform(0, 5)
    log_info(f"[{cam_id}] 采样错峰延迟: {stagger_delay:.2f}s")
    time.sleep(stagger_delay)

    last_p_time = 0
    last_s_time = 0

    while True:
        try:
            now = time.time()
            need_p_sample = has_presence_task and (now - last_p_time) >= p_interval
            
            # Smoking 仅在窗口激活时执行
            is_smoke_active = s_sm.check_window_active() if has_presence_task else True
            need_s_sample = has_smoking_task and is_smoke_active and ((now - last_s_time) >= s_interval)

            if need_p_sample or need_s_sample:
                # 使用宿主机本地 FFmpeg 抓拍
                frame = get_frame_from_host_ffmpeg(cam_id)
                
                if frame is None:
                    time.sleep(5)
                    continue
                
                # 预热初始化 TensorRT (如果是首次)
                init_tensorrt_models()
                
                # --- 1. Presence (人员存在) 综合判定流程 ---
                if need_p_sample:
                    last_p_time = now
                    
                    # RF-DETR/YOLO 一级判定
                    boxes = pose_model.predict(frame)
                    
                    # 过滤出“人”类别的框 (person_class_id 通常是 0)
                    person_boxes = [b for b in boxes if b.get('class_id') == 0]
                    
                    has_person = False
                    decision_chain = []
                    yolo_count = len(person_boxes)
                    annotated_frame = frame.copy()
                    
                    # 绘制时间戳
                    cv2.putText(annotated_frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 绘制所有检测到的目标（用于调试分析）
                    if len(boxes) > 0:
                        for b in boxes:
                            x1, y1, x2, y2 = b['bbox']
                            cls_name = b.get('class_name', 'unknown')
                            conf = b['conf']
                            # 人用红色，其他用蓝色
                            color = (0, 0, 255) if cls_name == 'person' else (255, 0, 0)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if yolo_count > 0:
                        decision_chain.append(f"Detector 检测到 {yolo_count} 个候选人员")
                        max_conf = max([b['conf'] for b in person_boxes])
                    else:
                        decision_chain.append("Detector 未检测到人员，准备全图复核")
                        max_conf = 0.0

                    # --- 核心改进逻辑 ---
                    # 1. 如果 Detector 信心极高 (>= 70%)，直接通过，不麻烦 Gemma
                    if max_conf >= 0.70:
                        gemma_res = "YES"
                        decision_chain.append(f"Detector 高置信度({max_conf:.2f})直接确认有人")
                    else:
                        # 2. 否则，送给 Gemma 做最终裁决
                        prompt = "检测图片中是否有活人存在，仔细鉴别头肩和肢体等人体要输，如果有人回答YES，并且告知在什么位置。没有则回答NO"
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            jpg_bytes = buffer.tobytes()
                            # submit_review 现在返回 dict {result, prompt, llm_response, reasoning}
                            gemma_data = gemma_queue.submit_review(f"{cam_id}_P_{now}", "presence", jpg_bytes, prompt, yolo_conf=max_conf)
                            gemma_res = gemma_data.get("result", "UNKNOWN")
                            gemma_details = gemma_data
                            
                            if gemma_res == "UNKNOWN":
                                # 异常降级保护：如果 Gemma 挂了或超时，降级采信 Detector 的原始结果
                                if yolo_count > 0:
                                    gemma_res = "YES"
                                    decision_chain.append("Gemma 响应异常，降级采信 Detector 结果: YES")
                                else:
                                    gemma_res = "NO"
                                    decision_chain.append("Gemma 响应异常，降级采信 Detector 结果: NO")
                            else:
                                decision_chain.append(f"Gemma 二级裁决结果: {gemma_res}")
                        else:
                            log_info(f"[{cam_id}] OpenCV JPEG 编码失败，降级采信 Detector")
                            gemma_res = "YES" if yolo_count > 0 else "NO"
                            decision_chain.append("图像编码失败，降级采信 Detector")
                            gemma_details = None
                    
                    if gemma_res == "YES":
                        has_person = True
                        if max_conf < 0.70:
                            if yolo_count > 0:
                                decision_chain.append("Gemma 复核: 确认图中存在真实人员")
                            else:
                                decision_chain.append("Gemma 复核: Detector漏报，但Gemma在全图中发现了人员")
                        log_info(f"[{cam_id}] Presence: 确认有人 (YOLO框: {yolo_count}个, MaxConf: {max_conf:.2f})")
                    else:
                        has_person = False # 确保明确赋值
                        if yolo_count > 0:
                            decision_chain.append("Gemma 复核: 否决 (认定疑似目标为误报/假人)")
                        else:
                            decision_chain.append("Gemma 复核: 确认全图确实无人")
                        log_info(f"[{cam_id}] Presence: 判定无人")
                    
                    # 无论有没有人，送入状态机处理时间窗口
                    event_triggered, final_status, window_mins, time_period = p_sm.update(has_person_this_frame=has_person)
                    
                    # 【核心优化】反馈到前端页面：
                    save_minute_log_for_frontend(
                        cam_id, 
                        area_code, 
                        has_person, 
                        images=[annotated_frame, frame], 
                        decision_chain=decision_chain, 
                        yolo_count=yolo_count,
                        gemma_details=gemma_details
                    )
                    
                    # 如果状态机决定收敛，触发 MQTT
                    if event_triggered:
                        payload = {
                            "areaCode": area_code,
                            "result": final_status, # occupied / empty
                            "windowMinutes": window_mins,
                            "timePeriod": time_period,
                            "source": f"{presence_detector_source}+gemma",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # 【核心优化】发布报警时，也使用标注过的图作为证据
                        publish_mqtt_event(cam_id, area_code, "presence", payload, annotated_frame)
                        
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
                        
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            jpg_bytes = buffer.tobytes()
                            gemma_data = gemma_queue.submit_review(f"{cam_id}_S_{now}", "smoking", jpg_bytes, prompt, yolo_conf=max_conf)
                            gemma_res = gemma_data.get("result", "UNKNOWN")
                        else:
                            log_info(f"[{cam_id}] OpenCV JPEG 编码失败，跳过 Gemma 复核")
                            gemma_res = "NO"
                        
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
                                    "source": f"smoking_specialist+gemma",
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
        # 使用 get_real_url 实现自适应：如果在宿主机运行，将 zlm:80 替换为 127.0.0.1:10081
        zlm_api_root = get_real_url(config.get("zlm", {}).get("api_url", "http://zlm:80/index/api"), ZLM_HTTP_PORT)
        
        # 彻底解决 URL 拼接可能导致的 404 问题：确保路径包含 addStreamProxy
        if not zlm_api_root.endswith("/addStreamProxy"):
            api_url = f"{zlm_api_root.rstrip('/')}/addStreamProxy"
        else:
            api_url = zlm_api_root

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
                    # 播放地址同样需要自适应转换显示
                    play_url = get_real_url(f"rtsp://zlm:554/live/{cam_id}")
                    print(f"[{cam_id}] ZLM Proxy configured. Live at {play_url}")
                elif "already exists" in res_data.get("msg", ""):
                    # 忽略已存在的报错，静默成功
                    pass
                else:
                    print(f"[{cam_id}] ZLM Proxy failed: {res_data.get('msg')}")
        except Exception as e:
            print(f"[{cam_id}] Could not connect to ZLM API ({api_url}): {e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    print("Starting AI Engine (Dual-Stage Architecture)...")
    
    # 检查 ffmpeg 是否存在，防止后续抓拍静默失败
    import shutil
    if not shutil.which("ffmpeg"):
        print("\n" + "!"*60)
        print("CRITICAL ERROR: 'ffmpeg' not found in system PATH!")
        print("This AI Engine requires FFmpeg to capture snapshots from RTSP streams.")
        print("Please follow the setup guide in docs/cicd.md to install it:")
        print("  sudo apt-get update && sudo apt-get install -y ffmpeg")
        print("!"*60 + "\n")
    
    # 注册 ZLM 代理 (项目核心记忆: 动态拉流)
    zlm_thread = threading.Thread(target=register_cameras_to_zlm)
    zlm_thread.start()

    # 重点：为了彻底消除多线程并发初始化导致的 double free，
    # 我们在主线程中先行预热初始化 TensorRT，然后再开启各个摄像头的处理线程。
    # 这样所有的 OpenCV GStreamer 实例化都会发生在模型加载完毕之后，避免内存抢占。
    # 由于需要连接视频流，这里我们可以做个短暂的等待，或者直接在主线程加载。
    print("Pre-loading TensorRT Engines sequentially in main thread...")
    init_tensorrt_models()

    # 启动摄像头定时采样线程
    threads = []
    
    # 启动单图测试 HTTP 服务
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
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
