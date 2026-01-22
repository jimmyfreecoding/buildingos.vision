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
    
    # Collect all debug thumbnails to draw them at the end
    all_debug_thumbnails = []

    for res in results:
        # Loop through boxes regardless of ID (inference only)
        if res.boxes:
            for i, box in enumerate(res.boxes.xyxy):
                kp = res.keypoints.data[i] if res.keypoints is not None else None
                box_np = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, box_np)
                
                # New Logic: Direct Hand ROI Detection
                # Ignore hand-to-face distance, check all hands
                nose = kp[0]
                l_wrist = kp[9]
                r_wrist = kp[10]
                
                hands_to_check = []
                if l_wrist[2] > 0.3: hands_to_check.append((l_wrist[:2], "Left Hand"))
                if r_wrist[2] > 0.3: hands_to_check.append((r_wrist[:2], "Right Hand"))
                
                for hand_pt, hand_name in hands_to_check:
                    hx, hy = map(int, hand_pt)
                    
                    # Crop ROI around hand (320x320)
                    # Increased to 320 to provide more context and handle occlusions
                    roi_size = 320
                    h, w = frame.shape[:2]
                    x1_roi = max(0, hx - roi_size//2)
                    y1_roi = max(0, hy - roi_size//2)
                    x2_roi = min(w, hx + roi_size//2)
                    y2_roi = min(h, hy + roi_size//2)
                    
                    # IMPORTANT: We crop from 'frame' (the clean original image)
                    # NOT 'annotated_frame' (which has the lines drawn on it)
                    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                    
                    if roi.size > 0:
                        # Create a debug thumbnail
                        thumb = cv2.resize(roi, (150, 150))
                        # Add label with Person ID
                        label = f"P{i+1} {hand_name.split()[0]}" 
                        cv2.putText(thumb, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        all_debug_thumbnails.append(thumb)

                        # Draw ROI debug box (Cyan)
                        cv2.rectangle(annotated_frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 255, 0), 1)
                    
                        # Use config confidence and enable agnostic NMS for better recall
                        spec_results = specialist(roi, conf=SMOKING_SPECIALIST_CONF, verbose=False, agnostic_nms=True)
                        for sr in spec_results:
                            if len(sr.boxes) > 0:
                                # Found Smoking!
                                for sbox in sr.boxes.xyxy:
                                    sx1, sy1, sx2, sy2 = sbox.cpu().numpy()
                                    fx1, fy1 = x1_roi + int(sx1), y1_roi + int(sy1)
                                    fx2, fy2 = x1_roi + int(sx2), y1_roi + int(sy2)
                                    
                                    # Draw Red Box
                                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                                    cv2.putText(annotated_frame, f"SMOKING ({hand_name})", (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw ALL debug thumbnails on the main frame
    # Vertical stack on the left, wrap to next column if needed
    y_offset = 50
    x_offset = 10
    for thumb in all_debug_thumbnails:
        th, tw = thumb.shape[:2]
        if y_offset + th < annotated_frame.shape[0]:
            # Draw white border
            cv2.rectangle(annotated_frame, (x_offset, y_offset), (x_offset+tw, y_offset+th), (255,255,255), 2)
            annotated_frame[y_offset:y_offset+th, x_offset:x_offset+tw] = thumb
            y_offset += th + 10
        else:
             # Wrap to next column
             x_offset += tw + 10
             y_offset = 50
             if x_offset + tw < annotated_frame.shape[1] and y_offset + th < annotated_frame.shape[0]:
                cv2.rectangle(annotated_frame, (x_offset, y_offset), (x_offset+tw, y_offset+th), (255,255,255), 2)
                annotated_frame[y_offset:y_offset+th, x_offset:x_offset+tw] = thumb
                y_offset += th + 10
                
                # Old logic commented out for reference
                # if is_hand_near_face(kp, box_np):
                # ...

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
