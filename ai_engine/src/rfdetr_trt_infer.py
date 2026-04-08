import cv2
import numpy as np
import threading


trt_infer_lock = threading.Lock()


# 标准 COCO 80 类别列表
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# COCO 91 类别到 80 类别的映射 (用于解决 91 类模型的索引偏移)
# 注意：91 类中索引 1 是人，但在 80 类中索引 0 是人。
COCO_91_TO_80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15,
    18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43,
    50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

class RFDETRTensorRTEngine:
    def __init__(self, engine_path, conf_thres=0.25, person_class_id=0, max_det=100):
        self.engine_path = engine_path
        self.conf_thres = float(conf_thres)
        self.person_class_id = int(person_class_id)
        self.max_det = int(max_det)
        self.source_name = "rf-detr-trt"
        self.classes = COCO_CLASSES

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"RF-DETR TensorRT runtime dependencies missing: {e}")

        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 使用 autoinit 创建的全局上下文
        self.cuda_context = cuda.Context.get_current()
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()
        self.input_name = None
        self.output_names = []
        self.tensor_meta = {}
        self.bindings = []

        self._init_io()
        print(f"✅ RF-DETR Engine Loaded: {engine_path}")

    def _init_io(self):
        trt = self.trt
        cuda = self.cuda

        num_tensors = self.engine.num_io_tensors
        self.bindings = [0] * num_tensors

        for i in range(num_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                shape = list(self.engine.get_tensor_shape(name))
                if -1 in shape:
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)[1]
                    shape = list(profile_shape)
                    self.context.set_input_shape(name, tuple(shape))
            else:
                self.output_names.append(name)
                shape = list(self.context.get_tensor_shape(name))
                if -1 in shape:
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)[1]
                    shape = list(profile_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(trt.volume(tuple(shape)))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.tensor_meta[name] = {
                "shape": tuple(shape),
                "dtype": dtype,
                "host": host,
                "device": device,
                "index": i
            }
            self.bindings[i] = int(device)
            try:
                self.context.set_tensor_address(name, int(device))
            except Exception:
                pass

        if not self.input_name:
            raise RuntimeError("No input tensor found in engine")

        in_shape = self.tensor_meta[self.input_name]["shape"]
        if len(in_shape) != 4:
            raise RuntimeError(f"Unexpected input shape: {in_shape}")
        self.batch, self.channels, self.input_h, self.input_w = in_shape

    def _preprocess(self, img):
        """
        根据截图表现优化的预处理：
        1. 许多 TensorRT 模型内部已集成归一化，外部只需 0-1 缩放
        2. 保持 Squash 缩放以匹配 app.py
        """
        orig_h, orig_w = img.shape[:2]
        resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 【核心测试项】
        # 很多 TensorRT Engine 导出时自带了 Mean/Std，Python 层再做会导致“双重归一化”。
        # 目前暂时注释掉减均值操作，仅保留 0-1 缩放，排除模型“致盲”可能。
        x = rgb.astype(np.float32) / 255.0
        
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # x = (x - mean) / std
        
        x = np.transpose(x, (2, 0, 1))[None, ...]
        
        scale_x = self.input_w / orig_w
        scale_y = self.input_h / orig_h
        return x, scale_x, scale_y, orig_w, orig_h

    def _infer(self, input_tensor):
        cuda = self.cuda
        np.copyto(self.tensor_meta[self.input_name]["host"], input_tensor.ravel())
        cuda.memcpy_htod_async(self.tensor_meta[self.input_name]["device"], self.tensor_meta[self.input_name]["host"], self.stream)

        try:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        except Exception:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        outputs = {}
        for name in self.output_names:
            meta = self.tensor_meta[name]
            cuda.memcpy_dtoh_async(meta["host"], meta["device"], self.stream)
        self.stream.synchronize()

        for name in self.output_names:
            meta = self.tensor_meta[name]
            outputs[name] = np.array(meta["host"]).reshape(meta["shape"])
        return outputs

    def _parse_outputs(self, outputs, scale_x, scale_y, orig_w, orig_h, conf_thres=None):
        """
        针对 (1, 300, 4) dets 和 (1, 300, 91) labels 的专用解析逻辑：
        1. 分离张量：dets 为坐标，labels 为分类分数 (Logits)
        2. 坐标格式：dets 已经是 0-1 范围，不再做 Sigmoid
        3. 类别偏移：91 类通常包含背景，person 的索引为 1
        4. NMS 抑制：消除重叠冗余框
        """
        # --- 1. 深度诊断日志 ---
        print("\n" + "="*50)
        print("RF-DETR 91-CLASS ENGINE DETECTED")
        for k, v in outputs.items():
            print(f"Tensor: {k:15} | Shape: {str(v.shape):15} | Range: [{np.min(v):.2f}, {np.max(v):.2f}]")
        
        # 2. 锁定张量
        boxes_raw = outputs.get('dets')[0] if 'dets' in outputs else None
        logits_raw = outputs.get('labels')[0] if 'labels' in outputs else None

        if boxes_raw is None or logits_raw is None:
            for v in outputs.values():
                if v.ndim == 3 and v.shape[-1] == 4: boxes_raw = v[0]
                elif v.ndim == 3 and v.shape[-1] == 91: logits_raw = v[0]
        
        if boxes_raw is None or logits_raw is None:
            print("ERROR: Missing 'dets' or 'labels' tensors!")
            return []

        # 3. 执行分类分数计算
        scores_91 = 1 / (1 + np.exp(-np.clip(logits_raw, -15, 15)))
        
        max_scores_91 = np.max(scores_91, axis=1)
        max_indices_91 = np.argmax(scores_91, axis=1)
        
        actual_conf_thres = float(conf_thres) if conf_thres is not None else self.conf_thres
        
        # 4. 初步收集候选框用于 NMS
        candidates = []
        for i in range(len(max_scores_91)):
            if max_scores_91[i] >= actual_conf_thres:
                cx, cy, bw, bh = boxes_raw[i]
                
                # 解码为像素坐标
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h
                
                # 边界剪裁
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
                
                if x1 < x2 and y1 < y2:
                    cls_id_91 = int(max_indices_91[i])
                    cls_id_80 = COCO_91_TO_80.get(cls_id_91, -1)
                    if cls_id_80 != -1:
                        candidates.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(max_scores_91[i]),
                            "class_id": cls_id_80,
                            "class_name": self.classes[cls_id_80]
                        })

        # 5. 执行 NMS (非极大值抑制)
        # 解决图中出现的“2个人被识别成5个候选人”的重叠框问题
        if not candidates: return []
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x['conf'], reverse=True)
        results = []
        
        def calculate_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou

        while len(candidates) > 0:
            best = candidates.pop(0)
            results.append(best)
            
            # 过滤掉与当前最强框重叠度过高的同类框
            # 按照建议将阈值调低到 0.45 以更激进地消除重叠框
            remaining = []
            for item in candidates:
                if item['class_id'] == best['class_id'] and calculate_iou(best['bbox'], item['bbox']) > 0.45:
                    continue
                remaining.append(item)
            candidates = remaining
        
        if results:
            best_det = max(results, key=lambda x: x['conf'])
            print(f"✅ SUCCESS: Detected {best_det['class_name']} ({best_det['conf']:.3f}) after NMS")
            
        return results

    def predict(self, img, conf_thres=None):
        if img is None:
            return []
        
        # 关键：确保在推理线程中使用正确的 CUDA Context
        self.cuda_context.push()
        try:
            with trt_infer_lock:
                x, scale_x, scale_y, w, h = self._preprocess(img)
                outputs = self._infer(x)
                return self._parse_outputs(outputs, scale_x, scale_y, w, h, conf_thres=conf_thres)
        finally:
            self.cuda_context.pop()
