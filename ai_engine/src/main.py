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

# --- 强制写死宿主机运行环境 ---
PROJECT_ROOT = "/home/buildingos/buildingos.vision"
AI_SRC_DIR = os.path.join(PROJECT_ROOT, "ai_engine/src")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "ai_engine/config/config.json")
ZLM_WWW_DIR = os.path.join(PROJECT_ROOT, "zlm/www")

if AI_SRC_DIR not in sys.path:
    sys.path.insert(0, AI_SRC_DIR)

# 直接导入业务模块
from yolo_infer import YoloTensorRTEngine
from rfdetr_trt_infer import RFDETRTensorRTEngine
from state_machine import PresenceStateMachine, SmokingStateMachine
from gemma_queue import gemma_queue

# --- 核心工具函数 ---
def log_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}")

def log_error(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] ❌ {msg}")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
            # 彻底写死：宿主机访问 ZLM 必须使用 localhost
            # RTSP 使用 10554, HTTP 使用 10081 (对应 docker-compose 映射)
            if "streams" in config:
                for stream_type in ["smoking", "occupancy"]:
                    if stream_type in config["streams"]:
                        for stream in config["streams"][stream_type]:
                            if "url" in stream:
                                # 1. 替换域名
                                url = stream["url"].replace("rtsp://zlm:", "rtsp://localhost:")
                                url = url.replace("rtsp://zlm/", "rtsp://localhost/")
                                
                                # 2. 修正端口
                                if "localhost:554" in url:
                                    url = url.replace("localhost:554", "localhost:10554")
                                elif "localhost/" in url:
                                    url = url.replace("localhost/", "localhost:10554/")
                                
                                stream["url"] = url
                                log_info(f"Stream {stream.get('id')} URL mapped to: {url}")
            
            # 彻底写死：宿主机访问 MQTT 必须使用 localhost
            if "mqtt" in config:
                config["mqtt"]["broker"] = "localhost"
                
            return config
    except Exception as e:
        log_error(f"Error loading config: {e}")
        sys.exit(1)

# 全局配置
config = load_config()
ai_config = config.get("ai_engine", {})

# --- 模型状态追踪 ---
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

# --- Flask App ---
flask_app = Flask(__name__)

@flask_app.route('/status', methods=['GET'])
def api_status():
    return jsonify({
        "engine": "BuildingOS Vision AI Engine",
        "timestamp": datetime.now().isoformat(),
        "models": model_status
    })

@flask_app.route('/predict', methods=['POST'])
def api_predict():
    """用于前端测试图功能"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        conf_thres = data.get('conf_thres')
        if conf_thres is not None:
            conf_thres = float(conf_thres)
            
        init_tensorrt_models()
        
        results = []
        if pose_model:
            results = pose_model.predict(frame, conf_thres=conf_thres)
            
        annotated_frame = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            conf = res['conf']
            cls_id = res['class_id']
            color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"person {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "results": results,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "detector_source": presence_detector_source
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    flask_app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# --- 模型加载 ---
pose_model = None
smoking_model = None
presence_detector_source = "None"
trt_init_lock = threading.Lock()

def init_tensorrt_models():
    global pose_model, smoking_model, presence_detector_source, model_status
    with trt_init_lock:
        if pose_model is not None:
            return
            
        log_info("Initializing detection models...")
        detector_cfg = ai_config.get("detector", {})
        presence_backend = detector_cfg.get("presence_backend", "yolo").lower()
        presence_conf = float(detector_cfg.get("presence_conf", 0.25))
        
        # 模型路径写死
        rf_engine_path = os.path.join(PROJECT_ROOT, "ai_engine/models/rf-detr-presence.engine")
        yolo_engine_path = os.path.join(PROJECT_ROOT, "ai_engine/models/yolo26m-pose.engine")
        smoke_engine_path = os.path.join(PROJECT_ROOT, "ai_engine/models/smoking_26m.engine")

        # Presence 模型
        model_status["presence"]["primary_model"] = rf_engine_path
        model_status["presence"]["fallback_model"] = yolo_engine_path

        if presence_backend == "rfdetr_trt":
            try:
                pose_model = RFDETRTensorRTEngine(rf_engine_path, conf_thres=presence_conf)
                presence_detector_source = "rf-detr"
                model_status["presence"]["status"] = "Running"
                model_status["presence"]["active_backend"] = "RF-DETR"
                model_status["presence"]["primary_status"] = "Active"
            except Exception as e:
                log_error(f"RF-DETR init failed: {e}. Trying Fallback YOLO...")
                model_status["presence"]["primary_status"] = f"Failed: {str(e)[:50]}"
                try:
                    pose_model = YoloTensorRTEngine(yolo_engine_path, conf_thres=presence_conf)
                    presence_detector_source = "yolo-fallback"
                    model_status["presence"]["status"] = "Running (Fallback)"
                    model_status["presence"]["active_backend"] = "YOLO (Fallback)"
                except Exception as fe:
                    model_status["presence"]["status"] = "Failed"
                    log_error(f"Fallback YOLO also failed: {fe}")
        else:
            try:
                pose_model = YoloTensorRTEngine(yolo_engine_path, conf_thres=presence_conf)
                presence_detector_source = "yolo26m"
                model_status["presence"]["status"] = "Running"
                model_status["presence"]["active_backend"] = "YOLO"
            except Exception as e:
                model_status["presence"]["status"] = "Failed"
                log_error(f"YOLO init failed: {e}")

        # Smoking 模型
        model_status["smoking"]["model"] = smoke_engine_path
        try:
            smoking_conf = float(detector_cfg.get("smoking_conf", 0.3))
            smoking_model = YoloTensorRTEngine(smoke_engine_path, conf_thres=smoking_conf)
            model_status["smoking"]["status"] = "Running"
        except Exception as e:
            model_status["smoking"]["status"] = "Failed"
            model_status["smoking"]["error"] = str(e)
            log_error(f"Failed to load Smoking engine: {e}")

# --- MQTT 客户端 ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
try:
    mqtt_client.connect("localhost", 1883, keepalive=60)
    mqtt_client.loop_start()
    log_info("✅ MQTT Connected to localhost.")
except Exception as e:
    log_error(f"MQTT Connection failed: {e}")

# --- 业务逻辑：日志保存 ---
def save_minute_log_for_frontend(cam_id, area_code, has_person, frame=None, decision_chain=None, yolo_count=0, gemma_details=None):
    """保存前端 Heatmap 所需的 JSON 和切图"""
    if not cam_id or not area_code: return

    try:
        # 路径强制写死到 ZLM www 目录下
        log_dir_base = os.path.join(ZLM_WWW_DIR, "occupancy_logs")
        today_str = datetime.now().strftime("%Y-%m-%d")
        safe_area = str(area_code).replace('/', '_').replace('\\', '_')
        target_dir = os.path.join(log_dir_base, today_str, safe_area)
        os.makedirs(target_dir, exist_ok=True)
        
        timestamp_ms = int(time.time() * 1000)
        image_name = ""
        
        if frame is not None:
            image_name = f"{timestamp_ms}.jpg"
            cv2.imwrite(os.path.join(target_dir, image_name), frame)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "has_person": has_person,
            "yolo_count": yolo_count,
            "images": [image_name] if image_name else [],
            "decision_chain": decision_chain or ["AI 引擎状态更新"],
            "gemma_details": gemma_details
        }
        
        log_file = os.path.join(target_dir, "minute_logs.json")
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except: pass
        
        existing_data.append(log_entry)
        if len(existing_data) > 1440: existing_data = existing_data[-1440:]
            
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        log_info(f"📊 日志保存成功: {log_file} (条目数: {len(existing_data)})")
            
    except Exception as e:
        log_error(f"Error saving minute log to {target_dir}: {e}")

# --- 业务逻辑：状态机存储 ---
camera_state_machines = {}

# --- 核心任务：人员感知 ---
def occupancy_task(cam_config):
    cam_id = cam_config.get("id")
    url = cam_config.get("url")
    area_code = cam_config.get("areaCode", "UNKNOWN")
    
    log_info(f"Starting Occupancy Task: {cam_id} -> {url}")
    
    # 初始化该摄像头的状态机
    if cam_id not in camera_state_machines:
        camera_state_machines[cam_id] = {
            "presence": PresenceStateMachine(cam_id, config.get("areas", [{}])[0]),
            "smoking": SmokingStateMachine(cam_id, {})
        }
    
    psm = camera_state_machines[cam_id]["presence"]
    ssm = camera_state_machines[cam_id]["smoking"]
    
    while True:
        try:
            # 1. 抓图
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                log_error(f"[{cam_id}] Failed to grab frame, retrying in 10s...")
                time.sleep(10)
                continue
            
            # 2. 一级检测 (RF-DETR/YOLO)
            results = pose_model.predict(frame)
            person_count = len(results)
            yolo_has_person = person_count > 0
            
            decision_chain = [f"一级检测 ({presence_detector_source}): 发现 {person_count} 个目标"]
            final_has_person = yolo_has_person
            gemma_details = None
            
            # 3. 二级复核 (Gemma) - 如果 YOLO 判定有人
            if yolo_has_person:
                _, img_encoded = cv2.imencode('.jpg', frame)
                gemma_res = gemma_queue.submit_review(
                    task_id=f"{cam_id}_{int(time.time())}",
                    task_type="presence",
                    jpg_bytes=img_encoded.tobytes(),
                    prompt="Is there any real person in this office area? Answer YES or NO.",
                    yolo_conf=results[0]['conf'] if results else 1.0
                )
                
                gemma_confirmed = (gemma_res.get("result") == "YES")
                gemma_details = gemma_res.get("reasoning", "")
                decision_chain.append(f"Gemma 二级复核: {gemma_res.get('result')} ({gemma_details})")
                
                # 最终判定：遵循 Gemma 结果 (或者 Gemma 不确定时保守认为有人)
                final_has_person = (gemma_res.get("result") != "NO")
            
            # 4. 更新状态机
            evt_triggered, final_status, window_min, period = psm.update(final_has_person)
            
            # 5. 如果状态机触发了窗口收敛 (occupied/empty)，发布 MQTT
            if evt_triggered:
                topic = f"buildingos/presence/result"
                payload = {
                    "camId": cam_id,
                    "areaCode": area_code,
                    "result": final_status,
                    "windowMinutes": window_min,
                    "timePeriod": period,
                    "timestamp": datetime.now().isoformat()
                }
                mqtt_client.publish(topic, json.dumps(payload))
                log_info(f"[{cam_id}] Presence Event: {final_status} (Window: {window_min}m, Period: {period})")
                
                # 如果确认有人，触发吸烟检测窗口
                if final_status == "occupied":
                    ssm.trigger_presence()
            
            # 6. 保存本地日志 (Heatmap 所需)
            save_minute_log_for_frontend(
                cam_id, area_code, final_has_person, 
                frame=frame, decision_chain=decision_chain, 
                yolo_count=person_count, gemma_details=gemma_details
            )
            
            # 每分钟采样一次
            time.sleep(60)
            
        except Exception as e:
            log_error(f"Error in occupancy task {cam_id}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

# --- 核心任务：吸烟检测 ---
def smoking_task(cam_config):
    cam_id = cam_config.get("id")
    url = cam_config.get("url")
    
    log_info(f"Starting Smoking Task: {cam_id} -> {url}")
    
    if cam_id not in camera_state_machines:
        # 这里理论上应该由 occupancy_task 初始化，如果没有则补齐
        camera_state_machines[cam_id] = {
            "presence": PresenceStateMachine(cam_id, config.get("areas", [{}])[0]),
            "smoking": SmokingStateMachine(cam_id, {})
        }
    
    ssm = camera_state_machines[cam_id]["smoking"]
    
    while True:
        try:
            # 只有在吸烟窗口激活时才执行检测 (由 Presence 触发)
            if not ssm.check_window_active():
                time.sleep(10)
                continue
                
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                time.sleep(5)
                continue
                
            if smoking_model:
                results = smoking_model.predict(frame)
                if len(results) > 0:
                    # YOLO 发现可疑吸烟，调用 Gemma 复核
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    gemma_res = gemma_queue.submit_review(
                        task_id=f"smoke_{cam_id}_{int(time.time())}",
                        task_type="smoking",
                        jpg_bytes=img_encoded.tobytes(),
                        prompt="Is the person in this image smoking? Answer YES or NO.",
                        yolo_conf=results[0]['conf']
                    )
                    
                    if gemma_res.get("result") == "YES":
                        log_info(f"🔥 Smoking CONFIRMED in {cam_id}!")
                        ssm.confirm_smoke()
                        
                        payload = {
                            "cameraId": cam_id,
                            "event": "confirmed_smoking",
                            "timestamp": datetime.now().isoformat(),
                            "details": gemma_res.get("reasoning", "")
                        }
                        mqtt_client.publish(f"buildingos/smoking/alert", json.dumps(payload))
            
            # 吸烟检测采样频率 20s
            time.sleep(20)
            
        except Exception as e:
            log_error(f"Error in smoking task {cam_id}: {e}")
            time.sleep(10)

def main():
    log_info("BuildingOS AI Engine (Host Mode) Starting...")
    
    # 1. 启动 Flask
    threading.Thread(target=run_flask, daemon=True).start()
    
    # 2. 预加载模型
    init_tensorrt_models()
    
    # 3. 启动任务线程
    occupancy_streams = config.get("streams", {}).get("occupancy", [])
    for cam in occupancy_streams:
        threading.Thread(target=occupancy_task, args=(cam,), daemon=True).start()
        
    smoking_streams = config.get("streams", {}).get("smoking", [])
    for cam in smoking_streams:
        threading.Thread(target=smoking_task, args=(cam,), daemon=True).start()
        
    log_info("All AI tasks (State Machine + Gemma) initiated.")
    
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
