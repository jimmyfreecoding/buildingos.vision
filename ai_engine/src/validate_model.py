import os
import yaml
from ultralytics import YOLO

def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATASET_DIR = os.path.join(MODELS_DIR, 'dataset', 'Smoking-Detection-4')
    MODEL_PATH = os.path.join(MODELS_DIR, 'smoking_specialist.pt')
    YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
    RUNS_DIR = os.path.join(BASE_DIR, 'runs')

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Check if dataset yaml exists
    if not os.path.exists(YAML_PATH):
        print(f"Error: Dataset config not found at {YAML_PATH}")
        return

    # Create runs directory if it doesn't exist
    os.makedirs(RUNS_DIR, exist_ok=True)

    # Read original yaml
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)

    # Update paths to be absolute to avoid relative path confusion
    # YOLO expects 'path' to be the root, and train/val/test relative to it, or absolute paths
    data['path'] = DATASET_DIR
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'

    # Write temporary yaml with absolute paths
    TEMP_YAML = os.path.join(BASE_DIR, 'config', 'smoking_val_temp.yaml')
    os.makedirs(os.path.dirname(TEMP_YAML), exist_ok=True)
    
    with open(TEMP_YAML, 'w') as f:
        yaml.dump(data, f)

    print(f"Validating model: {MODEL_PATH}")
    print(f"Dataset config: {TEMP_YAML}")
    print(f"Output directory: {RUNS_DIR}")

    try:
        # Load model
        model = YOLO(MODEL_PATH)

        # Validate
        # project: where to save results
        # name: sub-directory name
        metrics = model.val(data=TEMP_YAML, project=RUNS_DIR, name='val_smoking')
        
        print(f"\nValidation complete!")
        print(f"Results (confusion matrix, PR curves, etc.) saved to: {os.path.join(RUNS_DIR, 'val_smoking')}")
        
    except Exception as e:
        print(f"An error occurred during validation: {e}")
    finally:
        # Optional: clean up temp yaml
        # os.remove(TEMP_YAML)
        pass

if __name__ == "__main__":
    main()
