import cv2
from ultralytics import YOLO
import threading
import time
import os
import sys
import numpy as np
import json
from flask import Flask, Response, render_template_string

# --- Configuration ---
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config/config.json")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

config = load_config()

SMOKING_CONF = config['model_params'].get('smoking_conf', 0.5)
OCCUPANCY_CONF = config['model_params'].get('occupancy_conf', 0.4)
SMOKING_SPECIALIST_CONF = config['model_params'].get('smoking_specialist_conf', 0.25)
POSE_HEURISTIC_THRESHOLD = config['model_params'].get('pose_heuristic_threshold', 0.25)

# --- Global Models (Lazy Load) ---
pose_model = None
smoking_specialist = None

def get_models():
    global pose_model, smoking_specialist
    if pose_model is None:
        print("Loading Pose Model...")
        pose_model = YOLO("/app/models/pose_v8n.pt", task='pose')
    if smoking_specialist is None:
        print("Loading Specialist Model...")
        path = "/app/models/smoking_specialist.pt"
        if not os.path.exists(path): path = "/app/models/yolov8n.pt"
        smoking_specialist = YOLO(path, task='detect')
    return pose_model, smoking_specialist

# --- Detection Logic (Simplified copy from main.py) ---
def is_hand_near_face(keypoints, box):
    if keypoints is None or len(keypoints) == 0: return False
    kp = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
    if kp.shape[1] < 3: return False
    nose = kp[0]
    l_wrist = kp[9]
    r_wrist = kp[10]
    box_h = box[3] - box[1]
    # Sync with main.py: Relaxed threshold to 0.25
    threshold = box_h * POSE_HEURISTIC_THRESHOLD
    if nose[2] > 0.5 and l_wrist[2] > 0.5:
        if np.linalg.norm(nose[:2] - l_wrist[:2]) < threshold: return True
    if nose[2] > 0.5 and r_wrist[2] > 0.5:
        if np.linalg.norm(nose[:2] - r_wrist[:2]) < threshold: return True
    return False

def process_frame(frame, models):
    pose_model, specialist = models
    
    # 1. Pose Detection
    # Using full resolution to match main.py logic exactly
    results = pose_model(frame, verbose=False, conf=SMOKING_CONF)
    annotated_frame = results[0].plot() # Draw skeleton
    
    for res in results:
        # Loop through boxes regardless of ID (inference only)
        if res.boxes:
            for i, box in enumerate(res.boxes.xyxy):
                kp = res.keypoints.data[i] if res.keypoints is not None else None
                box_np = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, box_np)
                
                # Check heuristic
                # Draw debug info for distance
                nose = kp[0]
                l_wrist = kp[9]
                r_wrist = kp[10]
                box_h = box_np[3] - box_np[1]
                threshold = box_h * POSE_HEURISTIC_THRESHOLD
                
                # Debug Text
                dist_l = np.linalg.norm(nose[:2] - l_wrist[:2]) if l_wrist[2] > 0.3 else 9999
                dist_r = np.linalg.norm(nose[:2] - r_wrist[:2]) if r_wrist[2] > 0.3 else 9999
                
                debug_txt = f"H:{int(box_h)} T:{int(threshold)} L:{int(dist_l)}({l_wrist[2]:.2f}) R:{int(dist_r)}({r_wrist[2]:.2f})"
                # Ensure y1-30 is visible
                text_y = max(30, y1 - 30)
                cv2.putText(annotated_frame, debug_txt, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if is_hand_near_face(kp, box_np):
                    # Draw Yellow Box for ROI Trigger
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated_frame, "Pose Trigger", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 2. Specialist Check
                    # Crop ROI
                    h, w = frame.shape[:2]
                    crop_h = (y2 - y1) * 0.5
                    cy1, cy2 = max(0, y1-20), min(h, int(y1+crop_h+20))
                    cx1, cx2 = max(0, x1-20), min(w, x2+20)
                    roi = frame[cy1:cy2, cx1:cx2]
                    
                    if roi.size > 0:
                        spec_results = specialist(roi, conf=SMOKING_SPECIALIST_CONF, verbose=False)
                        for sr in spec_results:
                            if len(sr.boxes) > 0:
                                # Found Smoking!
                                # Map back coords
                                for sbox in sr.boxes.xyxy:
                                    sx1, sy1, sx2, sy2 = sbox.cpu().numpy()
                                    fx1, fy1 = cx1 + int(sx1), cy1 + int(sy1)
                                    fx2, fy2 = cx1 + int(sx2), cy1 + int(sy2)
                                    
                                    # Draw Red Box
                                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                                    cv2.putText(annotated_frame, "SMOKING CONFIRMED", (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated_frame

# --- Flask App ---
app = Flask(__name__)

def generate_frames(camera_id):
    # Find camera URL
    stream_url = None
    for s_list in config['streams'].values():
        for s in s_list:
            if s['id'] == camera_id:
                stream_url = s['url']
                break
        if stream_url: break
    
    if not stream_url:
        print(f"Camera {camera_id} not found")
        return

    cap = cv2.VideoCapture(stream_url)
    models = get_models()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize for speed if needed
        # frame = cv2.resize(frame, (640, 480))
        
        # Process
        output_frame = process_frame(frame, models)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # List available cameras
    links = []
    for task, s_list in config['streams'].items():
        for s in s_list:
            links.append(f'<li><a href="/video_feed/{s["id"]}">{s["name"]} ({task})</a></li>')
            
    html = f"""
    <html>
      <head><title>AI Debug Stream</title></head>
      <body>
        <h1>AI Debug Console</h1>
        <p>Select a camera to view real-time inference:</p>
        <ul>
          {''.join(links)}
        </ul>
        <hr>
        <h3>Legend:</h3>
        <p><span style="color:blue">Blue</span>: Skeleton (Pose)</p>
        <p><span style="color:gold">Yellow</span>: Heuristic Triggered (Hand near face)</p>
        <p><span style="color:red">Red</span>: Specialist Confirmed (Smoking)</p>
      </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run on 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
