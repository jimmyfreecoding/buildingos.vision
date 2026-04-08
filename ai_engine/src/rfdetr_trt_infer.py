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
            # 兜底：尝试根据形状找
            for v in outputs.values():
                if v.ndim == 3 and v.shape[-1] == 4: boxes_raw = v[0]
                elif v.ndim == 3 and v.shape[-1] == 91: logits_raw = v[0]
        
        if boxes_raw is None or logits_raw is None:
            print("ERROR: Missing 'dets' or 'labels' tensors!")
            return []

        # 3. 执行分类分数计算
        scores_91 = 1 / (1 + np.exp(-np.clip(logits_raw, -15, 15)))
        
        # 关键：获取最强的人员(Person)分数
        # 在 COCO 91 中，1 是人，0 是背景。
        p1_scores = scores_91[:, 1]
        
        max_scores_91 = np.max(scores_91, axis=1)
        max_indices_91 = np.argmax(scores_91, axis=1)
        
        actual_conf_thres = float(conf_thres) if conf_thres is not None else self.conf_thres
        results = []

        for i in range(len(max_scores_91)):
            if max_scores_91[i] >= actual_conf_thres:
                # 4. 坐标解码：dets (0-1) 映射到像素
                # 这种格式通常是 [x1, y1, x2, y2] 或 [cx, cy, w, h]
                # 尝试：[cx, cy, w, h]
                cx, cy, bw, bh = boxes_raw[i]
                
                # 判断格式：如果 x2 < x1 显然不是 x1y1x2y2
                # 大多数 RF-DETR 导出的 dets 是 [x1, y1, x2, y2]
                # 但如果还是挤在左上角，我们继续使用中心点转换
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h
                
                # 5. 类别映射 (91 -> 80)
                # 简单映射：COCO 91 的索引 1 是 Person (对应 80 类的索引 0)
                # 目前先显示 ID 以确定偏移
                cls_id_91 = int(max_indices_91[i])
                
                results.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(max_scores_91[i]),
                    "class_id": cls_id_91,
                    "class_name": f"COCO91_{cls_id_91}" # 临时显示原始 ID 以便校对
                })
        
        if results:
            best = max(results, key=lambda x: x['conf'])
            print(f"✅ SUCCESS: Detected {best['class_name']} ({best['conf']:.3f}) | Person(idx1) Max: {np.max(p1_scores):.3f}")
            
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
