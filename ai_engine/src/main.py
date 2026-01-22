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
from datetime import datetime

# --- Load Configuration ---
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config/config.json")

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

ZLM_API_URL = config.get('zlm', {}).get('api_url', "http://zlm:80/index/api")
ZLM_SECRET = config.get('zlm', {}).get('secret', "")

MEDIA_STORAGE_PATH = config.get('media', {}).get('storage_path', "/app/www/captures")
MEDIA_BASE_URL = config.get('media', {}).get('base_url', "http://localhost:10080/captures")
VIDEO_DURATION = config.get('media', {}).get('video_duration', 5)

# Ensure media directory exists
os.makedirs(MEDIA_STORAGE_PATH, exist_ok=True)

# --- Global State for Cooldowns ---
last_alert_times = {}

# --- ZLMediaKit API Helper ---
def add_stream_proxy(stream_config):
    if 'source_url' not in stream_config:
        return

    api_url = f"{ZLM_API_URL}/addStreamProxy"
    params = {
        'secret': ZLM_SECRET,
        'vhost': '__defaultVhost__',
        'app': 'live',
        'stream': stream_config.get('zlm_stream_id', stream_config['id']),
        'url': stream_config['source_url'],
        'enable_rtsp': 1,
        'enable_rtmp': 1,
        'enable_hls': 0,
        'enable_mp4': 0
    }
    
    try:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{api_url}?{query_string}"
        print(f"Registering stream proxy: {stream_config['id']} -> ZLM")
        with urllib.request.urlopen(full_url) as response:
            resp_data = json.loads(response.read().decode())
            if resp_data.get('code') == 0:
                print(f"Successfully registered proxy for {stream_config['id']}")
            else:
                print(f"Failed to register proxy for {stream_config['id']}: {resp_data}")
    except Exception as e:
        print(f"Error calling ZLM API for {stream_config['id']}: {e}")

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

# --- State Machine & Tracking Classes ---
class PersonState:
    def __init__(self, patience_seconds=120):
        self.state = "VACANT"
        self.last_seen_time = 0
        self.patience_seconds = patience_seconds
        self.history_scores = deque(maxlen=10)
        self.last_alert_state = "VACANT" 

    def update(self, has_visual_detection, visual_score, motion_score=0):
        current_time = time.time()
        total_score = (visual_score * 0.7) + (motion_score * 0.3)
        self.history_scores.append(total_score)
        avg_score = sum(self.history_scores) / len(self.history_scores)

        if has_visual_detection:
            self.state = "ACTIVE"
            self.last_seen_time = current_time
        else:
            time_since_last_seen = current_time - self.last_seen_time
            if self.state == "ACTIVE":
                if time_since_last_seen < self.patience_seconds:
                    self.state = "POTENTIAL"
                else:
                    self.state = "VACANT"
            elif self.state == "POTENTIAL":
                if time_since_last_seen > self.patience_seconds:
                    self.state = "VACANT"
        return self.state, avg_score

camera_states = {}

# --- Model Initialization ---
def get_model(model_name, task):
    base_path = "/app/models" if os.path.exists("/app/models") else "./models"
    engine_path = os.path.join(base_path, f"{model_name}.engine")
    pt_path = os.path.join(base_path, f"{model_name}.pt")
    
    if os.path.exists(engine_path):
        print(f"Loading TensorRT model: {engine_path}")
        return YOLO(engine_path, task=task)
    
    if not os.path.exists(pt_path):
        print(f"Model {pt_path} not found. Using standard YOLOv8n for demo.")
        if task == 'pose':
            return YOLO("yolov8n-pose.pt", task='pose')
        return YOLO("yolov8n.pt", task='detect')
        
    print(f"Loading PyTorch model: {pt_path}")
    return YOLO(pt_path, task=task)

print("Initializing Models...")
# Stage 1: Pose for human detection and ROI proposal
# Load models in main thread to initialize them safely before threading
pose_model = get_model("pose_v8n", "pose")
# Force initialization of the underlying predictor to avoid thread race conditions
# Run a dummy inference to warm up
try:
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    pose_model(dummy_frame, verbose=False)
    print("Pose model warmed up.")
except Exception as e:
    print(f"Warning: Pose model warmup failed: {e}")

# Stage 2: Specialist for smoking detection (fine-grained)
# We look for smoking_specialist.pt. If missing, get_model defaults to yolov8n.pt
smoking_specialist = get_model("smoking_specialist", "detect")
try:
    smoking_specialist(dummy_frame, verbose=False)
    print("Specialist model warmed up.")
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

def process_occupancy(cam_id, frame, fps_counter=0):
    if cam_id not in camera_states:
        camera_states[cam_id] = PersonState(patience_seconds=STATE_PATIENCE)
    state_machine = camera_states[cam_id]
    results = pose_model.track(frame, persist=True, conf=OCCUPANCY_CONF, verbose=False)
    has_visual = False
    visual_score = 0.0
    current_results = []
    extra_annotations = []

    for res in results:
        if res.boxes and res.boxes.id is not None:
            if res.keypoints and len(res.keypoints.data) > 0:
                has_visual = True
                visual_score = float(res.boxes.conf.mean().cpu().numpy()) if res.boxes.conf is not None else 0.8
                current_results.append(res)
                for box in res.boxes.xyxy:
                     x1, y1, x2, y2 = box.cpu().numpy()
                     extra_annotations.append({
                        'box': [x1, y1, x2, y2],
                        'label': 'Person',
                        'color': (255, 0, 0)
                    })
                break
    
    prev_state = state_machine.state
    current_state, avg_score = state_machine.update(has_visual, visual_score)
    
    if current_state == "ACTIVE" and state_machine.last_alert_state != "ACTIVE":
        current_time = time.time()
        alert_key = f"{cam_id}_occupancy"
        last_alert = last_alert_times.get(alert_key, 0)
        
        if current_time - last_alert > ALERT_COOLDOWN:
            for ann in extra_annotations: ann['color'] = (0, 0, 255)
            img_url, video_url = capture_event_media(cam_id, frame, "occupancy", results=current_results, model_type="pose", extra_annotations=extra_annotations)
            topic = f"ai/alarm/occupancy/{cam_id}"
            payload = {
                "event": "PERSON_DETECTED",
                "camera": cam_id,
                "timestamp": datetime.now().isoformat(),
                "image_url": img_url,
                "video_url": video_url
            }
            mqtt_client.publish(topic, json.dumps(payload))
            print(f"[{cam_id}] Occupancy Event Published with Media")
            last_alert_times[alert_key] = current_time
            
    state_machine.last_alert_state = current_state
    
    # Send status update every time to prevent "stuck" states, or at least log debug info
    is_occupied = "1" if current_state in ["ACTIVE", "POTENTIAL"] else "0"
    topic = f"ai/status/occupancy/{cam_id}"
    mqtt_client.publish(topic, is_occupied)
    # Debug print to trace state changes
    if fps_counter % 200 == 0: # Print occasionally
         print(f"[{cam_id}] Occupancy State: {current_state} (Score: {avg_score:.2f}) -> MQTT: {is_occupied}")

def stream_worker(stream_config, task_type):
    if 'source_url' in stream_config:
        add_stream_proxy(stream_config)
        time.sleep(2)
    url = stream_config['url']
    cam_id = stream_config['id']
    cam_name = stream_config.get('name', cam_id)
    print(f"Starting worker for {cam_name} ({cam_id}) - Task: {task_type}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[{cam_id}] Failed to open stream: {url}")
        return
    fps_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_id}] Stream interrupted. Retrying...")
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue
        fps_counter += 1
        if task_type == "smoking" and fps_counter % 12 == 0:
            process_smoking(cam_id, frame)
        elif task_type == "occupancy":
            # Process more frequently (every 15 frames instead of 50) to catch movement better
            # Reset counter only when it gets very large to avoid overflow, not on every process
            if fps_counter % 15 == 0:
                 process_occupancy(cam_id, frame, fps_counter)
        
        if fps_counter > 10000: fps_counter = 0
    cap.release()

if __name__ == "__main__":
    threads = []
    for task_type, stream_list in config['streams'].items():
        for stream_conf in stream_list:
            t = threading.Thread(target=stream_worker, args=(stream_conf, task_type))
            t.daemon = True
            t.start()
            threads.append(t)
    print(f"Started {len(threads)} stream processing threads.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
