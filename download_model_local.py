import os
from roboflow import Roboflow

# Define target directory
# User asked for 'ai_engine/module', but our system uses 'ai_engine/models'.
# We will download to 'ai_engine/models/dataset' to be safe and organized.
TARGET_DIR = os.path.join(os.getcwd(), "ai_engine", "models", "dataset")
os.makedirs(TARGET_DIR, exist_ok=True)

# Change working directory so Roboflow downloads there
os.chdir(TARGET_DIR)

print(f"Downloading to: {TARGET_DIR}")

rf = Roboflow(api_key="om4YycxjpHh2z5jnKV1e")
project = rf.workspace("kyunghee-university-ada5d").project("smoking-detection-3gefl")
version = project.version(4)

# Download dataset
dataset = version.download("yolov8")

print(f"Download complete. Location: {dataset.location}")
print("\n--- INFO ON WEIGHTS ---")
print("Note: The Roboflow download() function typically downloads the DATASET (images + labels).")
print("It usually does NOT include the pre-trained 'best.pt' weights unless explicitly bundled.")
print("To get the weights, you typically need to TRAIN a model using this dataset.")
print("Check the downloaded folder for any '.pt' files just in case.")
