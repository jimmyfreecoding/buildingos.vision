import os
from roboflow import Roboflow

rf = Roboflow(api_key="om4YycxjpHh2z5jnKV1e")
project = rf.workspace("kyunghee-university-ada5d").project("smoking-detection-3gefl")
version = project.version(4)

print("Attempting to download model weights...")
try:
    # Try to download the model (weights)
    # This usually downloads to a folder like 'Smoking-Detection-4' or similar, 
    # and might contain 'weights/best.pt' if it's a YOLOv8 training.
    # Note: 'yolov8' format in download() usually refers to dataset export format.
    # To get weights, we might need a different call or it might be included if we are lucky.
    # Let's try to see if there is a specific method for weights.
    # For now, we stick to the user's snippet but inspect more.
    dataset = version.download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Check if weights are hidden somewhere
    for root, dirs, files in os.walk(dataset.location):
        for file in files:
            if file.endswith(".pt"):
                print(f"FOUND WEIGHTS: {os.path.join(root, file)}")
except Exception as e:
    print(f"Error: {e}")

# If the above didn't find weights, maybe we can try to use the deployment API to get them?
# But typically we need to train it ourselves if we only download the dataset.
# HOWEVER, the user said "Use this py to download... usually in weights/best.pt".
# This implies the user BELIEVES it downloads weights. 
# If it doesn't, we will use a placeholder.
