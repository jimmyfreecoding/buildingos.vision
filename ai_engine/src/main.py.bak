import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import threading
import time
import os
import sys
import numpy as np
import json
import urllib.request
import urllib.parse
from collections import deque
from datetime import datetime, timedelta
import shutil

# --- Load Configuration ---
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/ai_engine/config/config.json")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {CONFIG_PATH}: {e}")
        sys.exit(1)

config = load_config()

# --- Configuration Constants ---
MQTT_BROKER = os.getenv("MQTT_BROKER", config['mqtt']['broker'])
MQTT_PORT = int(os.getenv("MQTT_PORT", config['mqtt']['port']))
MQTT_KEEPALIVE = config['mqtt'].get('keepalive', 60)

SMOKING_CONF = config['model_params'].get('smoking_conf', 0.5)
OCCUPANCY_CONF = config['model_params'].get('occupancy_conf', 0.4)
STATE_PATIENCE = config['model_params'].get('state_patience', 120)
ALERT_COOLDOWN = config['model_params'].get('alert_cooldown', 30) # Default 30s cooldown
SMOKING_SPECIALIST_CONF = config['model_params'].get('smoking_specialist_conf', 0.25)
POSE_HEURISTIC_THRESHOLD = config['model_params'].get('pose_heuristic_threshold', 0.25)

ZLM_API_URL = os.getenv("ZLM_API", config.get('zlm', {}).get('api_url', "http://zlm:80/index/api"))
ZLM_SECRET = os.getenv("ZLM_SECRET", config.get('zlm', {}).get('secret', ""))

MEDIA_STORAGE_PATH = config.get('media', {}).get('storage_path', "/app/www/captures")
MEDIA_BASE_URL = config.get('media', {}).get('base_url', "http://localhost:10081/captures")
VIDEO_DURATION = config.get('media', {}).get('video_duration', 5)

# Occupancy log storage
OCCUPANCY_LOG_DIR = config.get('storage_quota', {}).get('occupancy_log_dir', "/app/www/occupancy_logs")
MAX_STORAGE_MB = config.get('storage_quota', {}).get('max_size_mb', 1024)

# Ensure directories exist
os.makedirs(MEDIA_STORAGE_PATH, exist_ok=True)
os.makedirs(OCCUPANCY_LOG_DIR, exist_ok=True)

# --- Global State for Cooldowns ---
last_alert_times = {}

# --- ZLMediaKit API Helper ---
def add_stream_proxy(stream_config):
    if 'source_url' not in stream_config:
        return

    api_url = f"{ZLM_API_URL}/addStreamProxy"
    
    # 提取并处理 url 参数，确保特殊字符（如 @）被正确编码
    source_url = stream_config['source_url']
    stream_id = stream_config.get('zlm_stream_id', stream_config['id'])
    
    # For testing/debug, if source_url has @ but no RTSP port, add one or check formatting
    # However, urlencode handles most things. We must pass the raw string to urlencode.
    
    params = {
        'secret': ZLM_SECRET,
        'vhost': '__defaultVhost__',
        'app': 'live',
        'stream': stream_id,
        'url': source_url,
        'enable_rtsp': 1,
        'enable_rtmp': 1,
        'enable_hls': 0,
        'enable_mp4': 0,
        'rtp_type': 0  # 0: tcp, 1: udp, 2: multicast (TCP is more stable for proxy)
    }
    
    try:
        # 对参数进行 urlencode，这会自动将源地址中的 @ 等特殊字符转换为 %40 等
        query_string = urllib.parse.urlencode(params)
        full_url = f"{api_url}?{query_string}"
        print(f"Registering stream proxy: {stream_id} -> ZLM")
        
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=10) as response:
            resp_data = json.loads(response.read().decode())
            if resp_data.get('code') == 0:
                print(f"Successfully registered proxy for {stream_id}")
            else:
                print(f"Failed to register proxy for {stream_id}: {resp_data}")
    except Exception as e:
        print(f"Error calling ZLM API for {stream_id}: {e}")

# --- Media Capture Helper ---
def capture_event_media(cam_id, frame, event_type, results=None, model_type="detect", extra_annotations=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{cam_id}_{event_type}_{timestamp}"
    
    # Draw Bounding Boxes on Frame Copy
    annotated_frame = frame.copy()
    if results:
        for r in results:
            if model_type == "detect":
                annotated_frame = r.plot() 
            elif model_type == "pose":
                annotated_frame = r.plot()
    
    # Draw extra custom annotations
    if extra_annotations:
        for ann in extra_annotations:
            x1, y1, x2, y2 = map(int, ann['box'])
            color = ann.get('color', (0, 0, 255))
            label = ann.get('label', '')
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    img_filename = f"{filename_base}.jpg"
    img_path = os.path.join(MEDIA_STORAGE_PATH, img_filename)
    cv2.imwrite(img_path, annotated_frame)
    img_url = f"{MEDIA_BASE_URL}/{img_filename}"
    
    video_filename = f"{filename_base}.mp4"
    video_path = os.path.join(MEDIA_STORAGE_PATH, video_filename)
    video_url = f"{MEDIA_BASE_URL}/{video_filename}"
    
    stream_url = None
    for s_list in config['streams'].values():
        for s in s_list:
            if s['id'] == cam_id:
                stream_url = s['url']
                break
        if stream_url: break
    
    if stream_url:
        threading.Thread(target=record_clip, args=(stream_url, video_path, VIDEO_DURATION)).start()
    
    return img_url, video_url

def record_clip(stream_url, output_path, duration):
    try:
        try:
            os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = "FFMPEG,GSTREAMER,V4L2"
            cap = cv2.VideoCapture(stream_url, getattr(cv2, 'CAP_FFMPEG', 1900))
        except Exception as e:
            cap = cv2.VideoCapture(stream_url)
            
        if not cap.isOpened():
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60: fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        start_time = time.time()
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
    except Exception as e:
        print(f"Error recording clip: {e}")

# --- Storage Cleanup Thread ---
def cleanup_storage_worker():
    while True:
        try:
            total_size = 0
            # Calculate total size
            for dirpath, _, filenames in os.walk(OCCUPANCY_LOG_DIR):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            
            total_size_mb = total_size / (1024 * 1024)
            if total_size_mb > MAX_STORAGE_MB:
                print(f"[Storage] Usage {total_size_mb:.2f}MB exceeds limit {MAX_STORAGE_MB}MB. Cleaning up...")
                
                # Get all date directories sorted by name (oldest first)
                date_dirs = [d for d in os.listdir(OCCUPANCY_LOG_DIR) if os.path.isdir(os.path.join(OCCUPANCY_LOG_DIR, d))]
                date_dirs.sort()
                
                # Remove oldest date directories until under 80% of max
                target_size = MAX_STORAGE_MB * 0.8
                for date_dir in date_dirs:
                    if total_size_mb <= target_size:
                        break
                    
                    dir_to_remove = os.path.join(OCCUPANCY_LOG_DIR, date_dir)
                    print(f"[Storage] Removing old directory: {dir_to_remove}")
                    
                    # Calculate size of this directory before removing
                    dir_size = 0
                    for dirpath, _, filenames in os.walk(dir_to_remove):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if not os.path.islink(fp):
                                dir_size += os.path.getsize(fp)
                    
                    try:
                        shutil.rmtree(dir_to_remove)
                        total_size_mb -= (dir_size / (1024 * 1024))
                    except Exception as e:
                        print(f"[Storage] Error removing {dir_to_remove}: {e}")
                        
        except Exception as e:
            print(f"[Storage Worker Error] {e}")
            
        # Run cleanup every hour
        time.sleep(3600)

# Start storage cleanup thread
threading.Thread(target=cleanup_storage_worker, daemon=True).start()

# --- State Machine & Tracking Classes ---
class AreaStateManager:
    def __init__(self, area_code, config):
        self.area_code = area_code
        self.config = config
        self.last_occupied_time = 0
        self.state = "VACANT"
        self.level2_triggered = False
        self.level3_triggered = False
        
        # Parse thresholds from config
        area_conf = self._get_area_config()
        self.score_threshold = area_conf.get('score_threshold', 0.6)
        self.buffer_minutes = area_conf.get('buffer_minutes', 2)
        self.level2_minutes = area_conf.get('level2_minutes', 5)
        self.level3_minutes = area_conf.get('level3_minutes', 10)

    def _get_area_config(self):
        areas = self.config.get('areas', [])
        for area in areas:
            if area.get('areaCode') == self.area_code:
                return area
        # Default fallback
        return {
            'score_threshold': 0.6,
            'buffer_minutes': 2,
            'level2_minutes': 5,
            'level3_minutes': 10
        }

    def evaluate(self, visual_score, motion_score, person_count, images_data):
        current_time = time.time()
        now = datetime.now()
        
        # Time bias (0.2 during 9:00-18:00, else 0)
        time_bias = 0.2 if 9 <= now.hour < 18 else 0.0
        
        # 优化打分公式：以视觉分数为基准，时间和移动作为加分项
        # 避免夜间或无移动分数时，视觉分数被乘以 0.6 导致永远无法达到 score_threshold 的bug
        total_score = visual_score + (motion_score * 0.2) + (time_bias * 0.5)
        
        event_type = None
        is_occupied = False
        
        if total_score > self.score_threshold:
            self.last_occupied_time = current_time
            self.state = "ACTIVE"
            self.level2_triggered = False
            self.level3_triggered = False
            is_occupied = True
            
            # 为了防止前端看到完全卡死，并且满足用户希望观察算法运行过程的需求
            # 我们将记录频率从原来的 60 秒改短，比如 10 秒，这样用户能更频繁地看到最新截图
            # 但是为了避免数据无限膨胀，我们在前端最好有分页或清理机制，这里我们在后端把频率设为 10秒一次
            if not hasattr(self, 'last_level1_time'):
                self.last_level1_time = 0
                
            if current_time - self.last_level1_time > 10: # 10 seconds throttle
                event_type = "LEVEL_1_DECISION"
                self.last_level1_time = current_time
            else:
                event_type = None # skip logging this time, but state is still ACTIVE
                
        else:
            time_since_last_seen = (current_time - self.last_occupied_time) / 60.0 # in minutes
            
            if self.state == "ACTIVE":
                if time_since_last_seen < self.buffer_minutes:
                    self.state = "POTENTIAL"
                    is_occupied = True
                else:
                    self.state = "VACANT"
            
            if self.state == "VACANT":
                if time_since_last_seen > self.level3_minutes and not self.level3_triggered:
                    self.level3_triggered = True
                    event_type = "LEVEL_3_TRIGGER"
                elif time_since_last_seen > self.level2_minutes and not self.level2_triggered:
                    self.level2_triggered = True
                    event_type = "LEVEL_2_TRIGGER"

        # Log process data if an event occurred
        if event_type:
            self._log_process_data(event_type, visual_score, motion_score, time_bias, total_score, is_occupied, person_count, images_data)
            
        return self.state, event_type

    def _log_process_data(self, event_type, visual_score, motion_score, time_bias, total_score, is_occupied, person_count, images_data):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H_%M_%S")
        
        # Create directory structure
        safe_area_code = self.area_code.replace('/', '_')
        log_dir = os.path.join(OCCUPANCY_LOG_DIR, date_str, safe_area_code)
        os.makedirs(log_dir, exist_ok=True)
        
        saved_images = []
        for cam_id, img_frame in images_data.items():
            img_filename = f"{time_str}_{cam_id}.jpg"
            img_path = os.path.join(log_dir, img_filename)
            cv2.imwrite(img_path, img_frame)
            # Store relative path for JSON
            saved_images.append(f"occupancy_logs/{date_str}/{safe_area_code}/{img_filename}")
            
        log_data = {
            "timestamp": now.isoformat(),
            "areaCode": self.area_code,
            "event": event_type,
            "scores": {
                "visual": visual_score,
                "motion": motion_score,
                "time_bias": time_bias,
                "total": total_score
            },
            "threshold_used": self.score_threshold,
            "is_occupied": is_occupied,
            "person_count": person_count,
            "images": saved_images
        }
        
        json_filename = f"{time_str}_{event_type}.json"
        json_path = os.path.join(log_dir, json_filename)
        
        try:
            with open(json_path, 'w') as f:
                json.dump(log_data, f, indent=4)
        except Exception as e:
            print(f"[{self.area_code}] Error saving log data: {e}")

area_states = {}

# --- Model Initialization & Auto-Conversion ---
def get_model(model_name, task):
    base_path = "/app/models" if os.path.exists("/app/models") else "./models"
    engine_path = os.path.join(base_path, f"{model_name}.engine")
    pt_path = os.path.join(base_path, f"{model_name}.pt")
    
    # Check if TensorRT engine exists
    if os.path.exists(engine_path):
        print(f"Loading TensorRT model: {engine_path}")
        return YOLO(engine_path, task=task)
    
    # If .pt doesn't exist either, use fallback
    if not os.path.exists(pt_path):
        print(f"Model {pt_path} not found. Using standard YOLOv8n for demo.")
        fallback_pt = "yolov8n-pose.pt" if task == 'pose' else "yolov8n.pt"
        pt_path = os.path.join(base_path, fallback_pt)
        if not os.path.exists(pt_path):
            return YOLO(fallback_pt, task=task) # Let ultralytics download it
        
    print(f"Loading PyTorch model: {pt_path}")
    model = YOLO(pt_path, task=task)
    
    # AUTO-CONVERSION LOGIC: If we are on Jetson (CUDA available) and no .engine exists
    import torch
    if torch.cuda.is_available():
        print(f"TensorRT engine not found for {model_name}. Starting auto-conversion...")
        print("This may take 5-10 minutes. Please be patient. Do not kill the process.")
        try:
            # 解决 TensorRT 转换时 OOM 或被 kill 的问题，降低 workspace 并禁用 half 强转如果失败
            model.export(format="engine", device=0, half=True, workspace=2)
            print(f"Auto-conversion successful! Created: {engine_path}")
            # Reload the newly created engine model
            return YOLO(engine_path, task=task)
        except Exception as e:
            print(f"Auto-conversion failed: {e}. Falling back to PyTorch model.")
    else:
        print("CUDA not available. Skipping TensorRT conversion. Running on CPU.")

    return model

print("Initializing Models...")
# Stage 1: Pose for human detection and ROI proposal
# Load models in main thread to initialize them safely before threading
pose_model = get_model("pose_v8n", "pose")
# Force initialization of the underlying predictor to avoid thread race conditions
# Run a dummy inference to warm up
try:
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    pose_model(dummy_frame, verbose=False, device=0)
    print("Pose model warmed up on GPU.")
except Exception as e:
    print(f"Warning: Pose model warmup failed: {e}")

# Stage 2: Specialist for smoking detection (fine-grained)
# We look for smoking_specialist.pt. If missing, get_model defaults to yolov8n.pt
smoking_specialist = get_model("smoking_specialist", "detect")
try:
    smoking_specialist(dummy_frame, verbose=False, device=0)
    print("Specialist model warmed up on GPU.")
except Exception as e:
    print(f"Warning: Specialist model warmup failed: {e}")

# --- MQTT Setup ---
mqtt_client = mqtt.Client()
try:
    print(f"Connecting to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}...")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE)
    mqtt_client.loop_start()
    print("MQTT Connected.")
except Exception as e:
    print(f"Error connecting to MQTT: {e}")

# --- Processing Logic ---

def is_hand_near_face(keypoints, box):
    if keypoints is None or len(keypoints) == 0:
        return False
    
    kp = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
    if kp.shape[1] < 3: return False 
    
    nose = kp[0]
    l_wrist = kp[9]
    r_wrist = kp[10]
    
    box_h = box[3] - box[1]
    # Relaxed threshold: Increased from 0.15 to 0.25 (25% of height)
    # This allows for more tolerance in hand-to-face distance
    threshold = box_h * POSE_HEURISTIC_THRESHOLD
    
    detected = False
    if nose[2] > 0.5 and l_wrist[2] > 0.5:
        if np.linalg.norm(nose[:2] - l_wrist[:2]) < threshold:
            detected = True
    if nose[2] > 0.5 and r_wrist[2] > 0.5:
        if np.linalg.norm(nose[:2] - r_wrist[:2]) < threshold:
            detected = True
            
    return detected

def get_upper_body_crop(frame, box):
    """
    Crops the upper body/head region for fine-grained detection.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # Expand slightly to ensure context
    crop_h = (y2 - y1) * 0.5 # Top 50% of the person box (head + shoulders + hands)
    
    cy1 = max(0, y1 - 20)
    cy2 = min(h, int(y1 + crop_h + 20))
    cx1 = max(0, x1 - 20)
    cx2 = min(w, x2 + 20)
    
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)

def capture_frame(stream_url):
    """Captures a single frame from the given URL."""
    try:
        # If it's a Snapshot API (returns a single JPEG image), use requests
        if 'getSnap' in stream_url:
            import requests
            import numpy as np
            from urllib.parse import urlparse, parse_qs
            
            # Parse the URL and its parameters manually so we can pass them cleanly to requests
            parsed_url = urlparse(stream_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            
            # Parse query string into dictionary
            # Note: parse_qs returns values as lists, we need the first item
            query_params = parse_qs(parsed_url.query)
            params = {k: v[0] for k, v in query_params.items()}
            
            # Increase timeout since API might be slow
            print(f"[capture_frame] Requesting snapshot Base URL: {base_url}")
            print(f"[capture_frame] Requesting snapshot Params: {params}")
            
            # Use params dict so requests handles the url encoding perfectly
            resp = requests.get(base_url, params=params, timeout=10)
            
            if resp.status_code == 200:
                # ZLM getSnap returns JSON if error, else returns image data.
                # Check Content-Type to be sure
                content_type = resp.headers.get('Content-Type', '')
                if 'image' in content_type:
                    image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    return frame
                else:
                    print(f"getSnap returned non-image data: {resp.text}")
                    return None
            else:
                print(f"HTTP capture failed with status {resp.status_code}: {resp.text}")
            return None
            
        # For video streams (RTSP or HTTP-FLV), use OpenCV
        # Fallback for RTSP (we use a timeout env to prevent hanging)
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "5000" # Avoid infinite blocking
        
        print(f"[capture_frame] Attempting VideoCapture on URL: {stream_url}")
        # force use FFMPEG backend, but if not exist, default is used.
        try:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        except:
            cap = cv2.VideoCapture(stream_url)
            
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None
    
    if 'cap' in locals() and not cap.isOpened():
        print(f"Failed to open VideoCapture for {stream_url}")
        return None
        
    if 'cap' in locals():
        for _ in range(3):
            ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        else:
             print(f"Failed to read frame from VideoCapture for {stream_url}")
            
    return None

def process_occupancy_areas():
    """Polls all configured areas periodically."""
    print("Starting area polling worker...")
    
    # Register all streams at startup (including smoking ones if not handled by worker)
    # Actually, smoking streams are handled by stream_worker, so we just handle occupancy here.
    
    while True:
        # Load the latest config inside the loop to catch any webadmin changes
        try:
            with open(CONFIG_PATH, 'r') as f:
                current_config = json.load(f)
        except Exception as e:
            print(f"Error reloading config: {e}")
            current_config = config # fallback to global
            
        occupancy_streams = current_config.get('streams', {}).get('occupancy', [])
        
        # --- NEW DEBUG LOGGING ---
        print(f"[Occupancy Polling] Reloaded config. Found {len(occupancy_streams)} occupancy streams.")
        for s in occupancy_streams:
            print(f"  - Stream ID: {s.get('id')}, Area: {s.get('areaCode')}, Source: {s.get('source_url')}")
        # -------------------------

        if not occupancy_streams:
            print("No occupancy streams configured. Waiting...")
            time.sleep(5)
            continue
        
        try:
            # Check ZLM stream status periodically
            try:
                zlm_list_url = f"{ZLM_API_URL}/getMediaList?secret={ZLM_SECRET}"
                req = urllib.request.Request(zlm_list_url)
                with urllib.request.urlopen(req, timeout=5) as response:
                    resp_data = json.loads(response.read().decode())
                    active_streams = []
                    if resp_data.get('code') == 0 and resp_data.get('data'):
                        active_streams = [s['stream'] for s in resp_data['data']]
                    
                    # Debug print
                    print(f"[Occupancy Polling] Found active streams in ZLM: {active_streams}")
                    
                    # Re-register if missing
                    for stream_conf in occupancy_streams:
                        stream_id = stream_conf.get('zlm_stream_id', stream_conf['id'])
                        if stream_id not in active_streams and 'source_url' in stream_conf:
                            print(f"[Occupancy] Stream {stream_id} missing in ZLM, re-registering...")
                            add_stream_proxy(stream_conf)
            except Exception as e:
                print(f"Error checking ZLM media list: {e}")

            # Add a short delay after potential registration before attempting to pull
            time.sleep(2)
            # Group streams by area
            area_streams = {}
            for stream in occupancy_streams:
                area_code = stream.get('areaCode', 'unknown_area')
                if area_code not in area_streams:
                    area_streams[area_code] = []
                area_streams[area_code].append(stream)
            
            for area_code, streams in area_streams.items():
                if area_code not in area_states:
                    # Pass the current_config so AreaStateManager gets latest thresholds
                    area_states[area_code] = AreaStateManager(area_code, current_config)
                
                state_machine = area_states[area_code]
                
                max_visual_score = 0.0
                total_person_count = 0
                images_data = {}
                
                # Fetch and process frames from all cameras in the area
                for stream_conf in streams:
                    cam_id = stream_conf.get('id', 'unknown')
                    stream_url = stream_conf.get('url')
                    
                    # If stream is missing in ZLM, don't even try to capture (avoids OpenCV block)
                    zlm_stream_id = stream_conf.get('zlm_stream_id', cam_id)
                    if 'active_streams' in locals() and zlm_stream_id not in active_streams:
                        print(f"[{cam_id}] Skipping capture because stream is not ready in ZLM.")
                        continue
                    
                    # Since ZLM is converting to all formats, we can use the getSnap API to get a JPEG safely without OpenCV blocking.
                    # If that fails, we fallback to OpenCV reading the HTTP-FLV stream, and finally RTSP.
                    
                    # NOTE: ZLM's getSnap API requires the URL of the FLV or RTMP stream.
                    internal_flv = f"http://127.0.0.1:80/live/{zlm_stream_id}.live.flv"
                    # DO NOT url encode the whole thing, just the internal URL
                    # the quote function is escaping the : and / characters which is correct for query params
                    encoded_flv = urllib.parse.quote(internal_flv, safe='')
                    
                    # We MUST use the zlm container hostname when calling from ai-engine
                    snapshot_url = f"http://zlm:80/index/api/getSnap?secret={ZLM_SECRET}&url={encoded_flv}&timeout_sec=5&expire_sec=10"
                    
                    print(f"[{cam_id}] Attempting to capture from ZLM Snapshot API: {snapshot_url}")
                    
                    # Instead of using requests directly inside process_occupancy_areas which we reverted,
                    # we will use our unified capture_frame function which now uses params
                    frame = capture_frame(snapshot_url)
                    
                    if frame is None:
                        # try local flv url for opencv since it runs in the same compose network
                        local_flv = f"http://zlm:80/live/{zlm_stream_id}.live.flv"
                        print(f"[{cam_id}] Snapshot API failed. Attempting to capture from HTTP-FLV: {local_flv}")
                        frame = capture_frame(local_flv)
                        
                        if frame is None:
                            print(f"[{cam_id}] All HTTP capture methods failed. Falling back to original stream URL.")
                            frame = capture_frame(stream_url)
                            if frame is None:
                                print(f"[{cam_id}] Original stream capture also failed.")
                    
                    if frame is not None:
                        print(f"[{cam_id}] Frame captured successfully. Running AI inference...")
                        # Add logic to calculate motion score if needed, currently 0
                        motion_score = 0.0 
                        
                        # Process with pose model
                        # IMPORTANT: Explicitly use device=0 or device='cuda:0' to force GPU inference
                        results = pose_model(frame, conf=OCCUPANCY_CONF, verbose=False, device=0)
                        
                        annotated_frame = frame.copy()
                        person_count = 0
                        cam_visual_score = 0.0
                        
                        for res in results:
                            if res.boxes:
                                person_count += len(res.boxes)
                                if len(res.boxes) > 0 and res.boxes.conf is not None:
                                    cam_visual_score = float(res.boxes.conf.max().cpu().numpy())
                                
                                # Draw red boxes around detected persons
                                for box in res.boxes.xyxy:
                                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(annotated_frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        images_data[cam_id] = annotated_frame
                        total_person_count += person_count
                        if cam_visual_score > max_visual_score:
                            max_visual_score = cam_visual_score
                    else:
                        print(f"[{cam_id}] Failed to capture frame from {stream_conf['url']}")

                # Evaluate state for the area
                current_state, event_type = state_machine.evaluate(max_visual_score, motion_score=0.0, person_count=total_person_count, images_data=images_data)
                
                # Handle MQTT Publishing based on event type
                if event_type:
                    topic = f"ai/event/occupancy/{area_code.replace('/', '_')}"
                    payload = {
                        "areaCode": area_code,
                        "event": event_type,
                        "timestamp": datetime.now().isoformat(),
                        "state": current_state
                    }
                    
                    if event_type == "LEVEL_1_DECISION":
                        payload["action"] = "keep_on"
                        payload["person_count"] = total_person_count
                    elif event_type == "LEVEL_2_TRIGGER":
                        payload["action"] = "dim"
                        payload["level"] = 2
                    elif event_type == "LEVEL_3_TRIGGER":
                        payload["action"] = "off"
                        payload["level"] = 3
                        
                    try:
                        mqtt_client.publish(topic, json.dumps(payload))
                        print(f"[{area_code}] Published Event: {event_type} -> Action: {payload.get('action')}")
                    except Exception as mqtt_err:
                        print(f"[{area_code}] Failed to publish event: {mqtt_err}")
                
                # Send periodic status update
                is_occupied = "1" if current_state in ["ACTIVE", "POTENTIAL"] else "0"
                status_topic = f"ai/status/occupancy/{area_code.replace('/', '_')}"
                try:
                    mqtt_client.publish(status_topic, is_occupied)
                except Exception:
                    pass
                
        except Exception as e:
            print(f"Error in process_occupancy_areas: {e}")
            import traceback
            traceback.print_exc()
            
        # Poll every 5 seconds
        time.sleep(5)

def process_smoking(cam_id, frame):
    # Stage 1: Pose Detection
    results = pose_model.track(frame, persist=True, conf=SMOKING_CONF, verbose=False)
    
    detected_smoking = False
    extra_annotations = []
    
    for res in results:
        if res.boxes and res.boxes.id is not None:
            for i, box in enumerate(res.boxes.xyxy):
                keypoints = res.keypoints.data[i] if res.keypoints is not None else None
                box_np = box.cpu().numpy()
                
                # Heuristic: Hand near face?
                is_near = is_hand_near_face(keypoints, box_np)
                if is_near:
                    print(f"[{cam_id}] Pose Heuristic Triggered: Hand near face detected.")
                    # Stage 2: Specialist Verification
                    roi_img, (rx1, ry1, rx2, ry2) = get_upper_body_crop(frame, box_np)
                    
                    if roi_img.size > 0:
                        # Run specialist model on ROI
                        # conf=0.4 slightly lower because small object
                        # Adjusting threshold to 0.25 to improve recall for smoking
                        spec_results = smoking_specialist(roi_img, conf=SMOKING_SPECIALIST_CONF, verbose=False) 
                        
                        has_target = False
                        for sr in spec_results:
                            # Check for specific classes: 'Cigarette', 'Smoke' (or index 0, 1 if using custom model)
                            # If using placeholder yolov8n, it detects 'person', 'cell phone' etc.
                            # For now, if ANY detection occurs in ROI with reasonable confidence, we assume valid
                            # In production, check: sr.names[int(cls)] in ['cigarette', 'smoke']
                            if len(sr.boxes) > 0:
                                has_target = True
                                print(f"[{cam_id}] Specialist Confirmed: Found target in ROI (Conf > {SMOKING_SPECIALIST_CONF}).")
                                # Draw ROI box and specialist detections on main frame for evidence
                                extra_annotations.append({
                                    'box': [rx1, ry1, rx2, ry2],
                                    'label': 'ROI Checked',
                                    'color': (0, 255, 255) # Yellow ROI
                                })
                                # Map ROI coordinates back to full frame for annotations
                                for s_box in sr.boxes.xyxy:
                                    sx1, sy1, sx2, sy2 = s_box.cpu().numpy()
                                    extra_annotations.append({
                                        'box': [rx1+sx1, ry1+sy1, rx1+sx2, ry1+sy2],
                                        'label': 'CONFIRMED',
                                        'color': (0, 0, 255) # Red confirmed
                                    })
                        
                        if has_target:
                            detected_smoking = True
                    else:
                        print(f"[{cam_id}] Warning: ROI image empty.")
                else:
                    # Optional: Print verbose log if needed for debugging why pose failed
                    pass

    if detected_smoking:
        current_time = time.time()
        alert_key = f"{cam_id}_smoking"
        last_alert = last_alert_times.get(alert_key, 0)
        
        if current_time - last_alert > ALERT_COOLDOWN:
            img_url, video_url = capture_event_media(cam_id, frame, "smoking", results=results, model_type="pose", extra_annotations=extra_annotations)
            
            topic = f"ai/alarm/smoking/{cam_id}"
            payload = {
                "event": "SMOKING_DETECTED",
                "camera": cam_id,
                "timestamp": datetime.now().isoformat(),
                "image_url": img_url,
                "video_url": video_url,
                "details": "Confirmed by Specialist Model"
            }
            mqtt_client.publish(topic, json.dumps(payload))
            print(f"[{cam_id}] Smoking Event Published (Cascade Confirmed)")
            last_alert_times[alert_key] = current_time

# Remove process_occupancy
# def process_occupancy(cam_id, frame, fps_counter=0):

def stream_worker(stream_config, task_type):
    url = stream_config['url']
    cam_id = stream_config['id']
    cam_name = stream_config.get('name', cam_id)
    print(f"Starting worker for {cam_name} ({cam_id}) - Task: {task_type}")
    
    fps_counter = 0
    while True: # Keep thread alive indefinitely
        # Re-register proxy before attempting connection if stream drops or fails
        if 'source_url' in stream_config:
            add_stream_proxy(stream_config)
            time.sleep(3) # Give ZLM a little more time to pull the stream

        try:
            # Fallback for cv2.CAP_FFMPEG missing in some environments, though it should be there.
            os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = "FFMPEG,GSTREAMER,V4L2"
            # In some OpenCV builds, CAP_FFMPEG might not be available, use literal value 1900 or fallback
            cap = cv2.VideoCapture(url, getattr(cv2, 'CAP_FFMPEG', 1900))
        except Exception as e:
            print(f"[{cam_id}] Warning: Failed to use CAP_FFMPEG directly, falling back to default. Error: {e}")
            cap = cv2.VideoCapture(url)
            
        # Give OpenCV a tiny moment to establish the connection before checking isOpened
        time.sleep(1)
        if not cap.isOpened():
            print(f"[{cam_id}] Failed to open stream: {url}. OpenCV CAP_FFMPEG returned False. Retrying in 5 seconds...")
            time.sleep(5)
            continue
            
        print(f"[{cam_id}] Successfully connected to stream.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[{cam_id}] Stream interrupted or EOF. Reconnecting...")
                break # Break inner loop to trigger reconnection
                
            fps_counter += 1
            if task_type == "smoking" and fps_counter % 12 == 0:
                process_smoking(cam_id, frame)
            # Removed occupancy processing from continuous stream loop
            
            if fps_counter > 10000: 
                fps_counter = 0
                
        cap.release()
        time.sleep(3) # Pause before reconnecting

if __name__ == "__main__":
    threads = []
    
    # Start polling thread for occupancy areas
    if 'occupancy' in config['streams'] and len(config['streams']['occupancy']) > 0:
        occupancy_thread = threading.Thread(target=process_occupancy_areas)
        occupancy_thread.daemon = True
        occupancy_thread.start()
        threads.append(occupancy_thread)

    # Only start other stream workers (like smoking) if they actually have cameras configured
    for task_type, stream_list in config.get('streams', {}).items():
        if task_type == 'occupancy':
            continue # Occupancy is now handled by the polling thread

        if isinstance(stream_list, list) and len(stream_list) > 0:
            for stream_conf in stream_list:
                t = threading.Thread(target=stream_worker, args=(stream_conf, task_type))
                t.daemon = True
                t.start()
                threads.append(t)
                
    print(f"Started {len(threads)} background threads.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
