import os
import sys
import time
import json
import base64
import threading
import multiprocessing
from datetime import datetime
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify

# --- 强制写死宿主机运行路径 ---
current_dir = "/home/buildingos/buildingos.vision/ai_engine/src"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 直接从当前目录导入
from yolo_infer import YoloTensorRTEngine
from rfdetr_trt_infer import RFDETRTensorRTEngine

# --- 核心工具函数写死 ---
def load_config(path):
    real_path = "/home/buildingos/buildingos.vision/ai_engine/config/config.json"
    try:
        with open(real_path, 'r') as f:
            config = json.load(f)
            # 彻底写死：宿主机访问 ZLM 必须使用 localhost:10554
            if "streams" in config:
                for stream_type in ["smoking", "occupancy"]:
                    if stream_type in config["streams"]:
                        for stream in config["streams"][stream_type]:
                            # 替换域名 zlm -> localhost
                            if "url" in stream:
                                stream["url"] = stream["url"].replace("rtsp://zlm:", "rtsp://localhost:")
                                # 强制补全端口，防止 FFmpeg 报 Port missing
                                if "rtsp://localhost/" in stream["url"]:
                                    stream["url"] = stream["url"].replace("rtsp://localhost/", "rtsp://localhost:10554/")
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def log_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}")

def log_error(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] ❌ {msg}")

# --- Global Configurations ---
# 同样写死
config = load_config(None)
ai_config = config.get("ai_engine", {})

# --- Global Configurations ---
config_path = os.path.join(os.path.dirname(__file__), '../config/config.json')
config = load_config(config_path)
ai_config = config.get("ai_engine", {})

# --- Flask App for Single Image Test ---
flask_app = Flask(__name__)

# 模型状态追踪 (移到这里以便 api_status 访问)
model_status = {
    "presence": {
        "status": "Initializing",
        "active_backend": "None",
        "primary_model": "None",
        "primary_status": "None",
        "fallback_model": "None",
        "fallback_status": "None"
    },
    "smoking": {
        "status": "Initializing",
        "model": "None",
        "error": ""
    }
}

@flask_app.route('/status', methods=['GET'])
def api_status():
    """返回 AI 引擎的详细运行状态"""
    return jsonify({
        "engine": "BuildingOS Vision AI Engine",
        "timestamp": datetime.now().isoformat(),
        "models": model_status
    })

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
    如果在宿主机运行，且 URL 指向容器名 zlm，则将其替换为 localhost
    """
    if is_in_container():
        return url
    
    if "://zlm:" in url:
        return url.replace("://zlm:", f"://localhost:{zlm_http_port}")
    return url

# --- AI Task Workers ---
# 全局模型占位符 (延迟加载)
pose_model = None
smoking_model = None
presence_detector_source = "detector"

# 确保 TensorRT 初始化的锁，防止多个摄像头线程同时触发初始化
trt_init_lock = threading.Lock()

# --- Init TensorRT Engines ---
def init_tensorrt_models():
    global pose_model, smoking_model, presence_detector_source, model_status
    
    with trt_init_lock:
        if pose_model is not None and smoking_model is not None:
            return
            
        print("Initializing detection models...")
        try:
            detector_cfg = ai_config.get("detector", {})
            presence_backend = detector_cfg.get("presence_backend", "yolo").lower()
            presence_conf = float(detector_cfg.get("presence_conf", 0.25))
            fallback_yolo_path = get_real_path(detector_cfg.get("fallback_yolo_engine_path", "/app/models/yolo26m-pose.engine"))
            
            # 更新 Presence 状态元数据
            model_status["presence"]["primary_model"] = detector_cfg.get("presence_engine_path", "N/A")
            model_status["presence"]["fallback_model"] = fallback_yolo_path

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
                    model_status["presence"]["status"] = "Running"
                    model_status["presence"]["active_backend"] = "RF-DETR"
                    model_status["presence"]["primary_status"] = "Active"
                    model_status["presence"]["fallback_status"] = "Standby"
                except Exception as e:
                    print(f"RF-DETR init failed, fallback to YOLO: {e}")
                    model_status["presence"]["primary_status"] = f"Failed: {str(e)[:50]}"
                    try:
                        pose_model = YoloTensorRTEngine(fallback_yolo_path, conf_thres=presence_conf)
                        presence_detector_source = "yolo26m"
                        model_status["presence"]["status"] = "Running (Fallback)"
                        model_status["presence"]["active_backend"] = "YOLO (Fallback)"
                        model_status["presence"]["fallback_status"] = "Active"
                    except Exception as fe:
                        model_status["presence"]["status"] = "Failed"
                        model_status["presence"]["fallback_status"] = f"Failed: {str(fe)[:50]}"
            else:
                try:
                    presence_engine_path = get_real_path(detector_cfg.get("presence_engine_path", fallback_yolo_path))
                    pose_model = YoloTensorRTEngine(presence_engine_path, conf_thres=presence_conf)
                    presence_detector_source = "yolo26m"
                    model_status["presence"]["status"] = "Running"
                    model_status["presence"]["active_backend"] = "YOLO"
                    model_status["presence"]["primary_status"] = "Active"
                except Exception as e:
                    model_status["presence"]["status"] = "Failed"
                    model_status["presence"]["primary_status"] = f"Failed: {str(e)[:50]}"

            # Smoking 模型加载
            smoking_engine_path = get_real_path(detector_cfg.get("smoking_engine_path", "/app/models/smoking_26m.engine"))
            model_status["smoking"]["model"] = smoking_engine_path
            try:
                smoking_conf = float(detector_cfg.get("smoking_conf", 0.3))
                smoking_model = YoloTensorRTEngine(smoking_engine_path, conf_thres=smoking_conf)
                model_status["smoking"]["status"] = "Running"
                print("Models loaded successfully.")
            except Exception as e:
                model_status["smoking"]["status"] = "Failed"
                model_status["smoking"]["error"] = str(e)
                print(f"Failed to load Smoking engine: {e}")

        except Exception as e:
            print(f"Failed to load TensorRT engines: {e}")
            model_status["presence"]["status"] = "Error"
            model_status["smoking"]["status"] = "Error"

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
            for idx, img in enumerate(images):
                img_name = f"{timestamp_ms}_{idx}.jpg"
                img_path = os.path.join(target_dir, img_name)
                cv2.imwrite(img_path, img)
                image_paths.append(img_name)

        # 构造日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "has_person": has_person,
            "yolo_count": yolo_count,
            "images": image_paths,
            "decision_chain": decision_chain,
            "gemma_details": gemma_details,
            "raw_payload": raw_payload
        }
        
        log_file = os.path.join(target_dir, "minute_logs.json")
        
        # 追加写入
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        existing_data.append(log_entry)
        
        # 只保留最近 1440 条（一天的分钟数）
        if len(existing_data) > 1440:
            existing_data = existing_data[-1440:]
            
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        log_error(f"Error saving minute log: {e}")

def occupancy_task(cam_config):
    """
    人员占用检测任务线程：
    1. 间隔采样图片
    2. YOLO 预判
    3. 若有人，则调用 Gemma 复核 (如果配置了复核逻辑)
    4. 发布结果到 MQTT
    """
    cam_id = cam_config.get("id")
    url = get_real_url(cam_config.get("url"))
    area_code = cam_config.get("areaCode", "UNKNOWN")
    
    log_info(f"Starting Occupancy Task for {cam_id} ({url})...")
    
    # 初始化模型 (如果尚未初始化)
    init_tensorrt_models()
    
    while True:
        try:
            # 模拟间隔采样 (根据配置，如 1 分钟采样一次)
            # 实际上应该从 RTSP 流拉取最新帧
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                log_error(f"Failed to grab frame from {cam_id}")
                time.sleep(10)
                continue
            
            # 1. YOLO 预判
            results = pose_model.predict(frame)
            person_count = len(results)
            has_person = person_count > 0
            
            decision_chain = [f"Detector ({presence_detector_source}) found {person_count} persons."]
            
            # 2. 如果有人，发布消息 (此处简化了复核逻辑)
            # ... 复核逻辑 ...
            
            payload = {
                "cam_id": cam_id,
                "area_code": area_code,
                "has_person": has_person,
                "person_count": person_count,
                "timestamp": datetime.now().isoformat()
            }
            mqtt_client.publish(f"buildingos/occupancy/{cam_id}", json.dumps(payload))
            
            # 3. 保存分钟日志
            save_minute_log_for_frontend(cam_id, area_code, has_person, raw_payload=payload, images=[frame], decision_chain=decision_chain, yolo_count=person_count)
            
            time.sleep(60)
            
        except Exception as e:
            log_error(f"Error in occupancy task {cam_id}: {e}")
            time.sleep(10)

def smoking_task(cam_config):
    """
    吸烟检测任务线程
    """
    cam_id = cam_config.get("id")
    url = get_real_url(cam_config.get("url"))
    
    log_info(f"Starting Smoking Task for {cam_id} ({url})...")
    
    # 初始化模型
    init_tensorrt_models()
    
    while True:
        try:
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                time.sleep(10)
                continue
                
            # 执行吸烟检测
            if smoking_model:
                results = smoking_model.predict(frame)
                if len(results) > 0:
                    log_info(f"🔥 Smoking detected in {cam_id}!")
                    payload = {
                        "cam_id": cam_id,
                        "event": "smoking",
                        "timestamp": datetime.now().isoformat()
                    }
                    mqtt_client.publish(f"buildingos/events/smoking", json.dumps(payload))
            
            time.sleep(5) # 吸烟检测频率稍高
            
        except Exception as e:
            log_error(f"Error in smoking task {cam_id}: {e}")
            time.sleep(10)

def main():
    log_info("BuildingOS AI Engine Starting...")
    
    # 启动 Flask 服务
    threading.Thread(target=run_flask, daemon=True).start()
    
    # 预加载模型
    init_tensorrt_models()
    
    # 启动任务线程
    occupancy_streams = config.get("streams", {}).get("occupancy", [])
    for cam in occupancy_streams:
        threading.Thread(target=occupancy_task, args=(cam,), daemon=True).start()
        
    smoking_streams = config.get("streams", {}).get("smoking", [])
    for cam in smoking_streams:
        threading.Thread(target=smoking_task, args=(cam,), daemon=True).start()
        
    log_info("All AI tasks initiated. System running.")
    
    # 保持主线程运行
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
