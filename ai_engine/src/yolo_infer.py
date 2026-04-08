import cv2
import numpy as np
from ultralytics import YOLO
import threading

# 全局推理锁，防止多线程同时调用 TensorRT 引擎导致 double free
trt_infer_lock = threading.Lock()

class YoloTensorRTEngine:
    """
    使用 Ultralytics 原生加载器替换底层 PyCUDA/TensorRT API。
    解决原生 API 无法识别 YOLOv8 engine 头部 metadata 导致的 magicTag 报错。
    """
    def __init__(self, engine_path, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        self.engine_path = engine_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 自动推断任务类型
        task = 'pose' if 'pose' in engine_path.lower() else 'detect'
        print(f"✅ 正在使用 Ultralytics 原生引擎加载: {engine_path} (task={task})")
        
        # 原生加载，自带 TRT 推理加速，免除 magicTag 烦恼
        self.model = YOLO(engine_path, task=task)
        
    def predict(self, img, conf_thres=None):
        if img is None:
            return []
            
        actual_conf = conf_thres if conf_thres is not None else self.conf_thres
        
        with trt_infer_lock:
            # verbose=False 减少日志刷屏
            results = self.model(img, conf=actual_conf, iou=self.iou_thres, verbose=False)
        
        parsed_results = []
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()
            
            is_pose = hasattr(r, 'keypoints') and r.keypoints is not None
            if is_pose:
                kpts_data = r.keypoints.data.cpu().numpy()
                
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                res = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(confs[i]),
                    "class_id": int(class_ids[i])
                }
                
                if is_pose:
                    # 展平 17x3 的关键点数组
                    res["keypoints_raw"] = kpts_data[i].flatten().tolist()
                    
                parsed_results.append(res)
                
        return parsed_results
