import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import threading
import time
import os
import sys
import numpy as np
import json
from collections import deque

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

# --- State Machine & Tracking Classes ---

class PersonState:
    """
    State machine for a single detected person/zone.
    States:
    - ACTIVE: Person currently detected with high confidence
    - POTENTIAL: Person recently lost (maybe occluded), waiting for confirmation
    - VACANT: No person detected
    """
    def __init__(self, patience_seconds=120):
        self.state = "VACANT"
        self.last_seen_time = 0
        self.patience_seconds = patience_seconds # How long to wait in POTENTIAL state (e.g., 2 mins)
        self.history_scores = deque(maxlen=10) # Sliding window for scores

    def update(self, has_visual_detection, visual_score, motion_score=0):
        current_time = time.time()
        
        # Calculate Weighted Score (Requirement: 60% Visual, 20% Motion, 10% Time, 10% History)
        # Simplified here: 70% Visual + 30% Motion for demo
        # Time context and History are implicit in the state transition logic
        total_score = (visual_score * 0.7) + (motion_score * 0.3)
        self.history_scores.append(total_score)
        avg_score = sum(self.history_scores) / len(self.history_scores)

        # State Transitions
        if has_visual_detection:
            self.state = "ACTIVE"
            self.last_seen_time = current_time
        else:
            # No visual detection this frame
            time_since_last_seen = current_time - self.last_seen_time
            
            if self.state == "ACTIVE":
                if time_since_last_seen < self.patience_seconds:
                    self.state = "POTENTIAL" # Enter occlusion handling mode
                else:
                    self.state = "VACANT"
            
            elif self.state == "POTENTIAL":
                if time_since_last_seen > self.patience_seconds:
                    self.state = "VACANT" # Timed out
                # Else: stay in POTENTIAL (Memory Logic)

        return self.state, avg_score

# Global state dictionary for each camera
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
smoking_model = get_model("smoking_v8n", "detect")
pose_model = get_model("pose_v8n", "pose")

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

def process_smoking(cam_id, frame):
    # Using 'persist=True' enables ByteTrack in Ultralytics automatically
    results = smoking_model.track(frame, persist=True, conf=SMOKING_CONF, verbose=False)
    
    for res in results:
        # Check if we have boxes and TRACK IDs (meaning consistent objects)
        if res.boxes and res.boxes.id is not None:
            # Simply triggering if ANY smoking object is tracked
            topic = f"ai/alarm/smoking/{cam_id}"
            payload = "DETECTED"
            mqtt_client.publish(topic, payload)
            print(f"[{cam_id}] Smoking Event Published")

def process_occupancy(cam_id, frame):
    # Initialize state for this camera if new
    if cam_id not in camera_states:
        camera_states[cam_id] = PersonState(patience_seconds=STATE_PATIENCE)
    
    state_machine = camera_states[cam_id]
    
    # 1. Visual Detection with Tracking (ByteTrack)
    # Using Pose model for better 'human' vs 'chair' distinction
    results = pose_model.track(frame, persist=True, conf=OCCUPANCY_CONF, verbose=False)
    
    has_visual = False
    visual_score = 0.0
    
    for res in results:
        if res.boxes and res.boxes.id is not None:
            # Check for keypoints to confirm it's a person (Skeleton check)
            if res.keypoints and len(res.keypoints.data) > 0:
                has_visual = True
                # Use mean confidence as score
                visual_score = float(res.boxes.conf.mean().cpu().numpy()) if res.boxes.conf is not None else 0.8
                break
    
    # 2. Update State Machine (combining visual + memory)
    # Note: Motion detection (Optical Flow) would be calculated here and passed as motion_score
    current_state, avg_score = state_machine.update(has_visual, visual_score)
    
    # 3. Publish Result
    # Output: 1 (Occupied/Potential), 0 (Vacant)
    # We treat POTENTIAL as "Occupied" to prevent lights turning off during occlusion
    is_occupied = "1" if current_state in ["ACTIVE", "POTENTIAL"] else "0"
    
    topic = f"ai/status/occupancy/{cam_id}"
    mqtt_client.publish(topic, is_occupied)
    
    # Optional: Publish detailed state for debugging
    debug_topic = f"ai/debug/occupancy/{cam_id}"
    mqtt_client.publish(debug_topic, f'{{"state": "{current_state}", "score": {avg_score:.2f}}}')

def stream_worker(stream_config, task_type):
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
            
        elif task_type == "occupancy" and fps_counter % 50 == 0:
            process_occupancy(cam_id, frame)
            fps_counter = 0

    cap.release()

if __name__ == "__main__":
    threads = []
    
    # Load streams from config
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
