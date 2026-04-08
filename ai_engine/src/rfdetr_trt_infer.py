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
        
        # 【重要测试项】
        # 如果模型分类完全牛头不对马嘴，极有可能是因为输入数据被“双重归一化”成了噪声。
        # 建议测试：注释掉 mean/std 减法，仅保留 / 255.0
        x = rgb.astype(np.float32) / 255.0
        
        # 暂时开启：Mean/Std 归一化。
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        
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
        # 1. 寻找 84 列的主输出 (300个预测框，每个框 4坐标 + 80类别)
        main_tensor = None
        for v in outputs.values():
            if v.ndim == 3 and v.shape[-1] >= 84:
                main_tensor = v[0]
                break
        if main_tensor is None: return []

        # 调试：观察前 3 个原始框的数值
        # 如果 cx, cy 大于 1 或小于 0，说明模型输出的不是归一化的比例值
        print(f"DEBUG RF-DETR Raw Tensor (first 3): \n{main_tensor[:3, :6]}")

        # 1. 拆分坐标和类别概率
        boxes_raw = main_tensor[:, :4]
        logits_raw = main_tensor[:, 4:84]
        
        # 强制执行 Sigmoid 以获取分数
        scores = 1 / (1 + np.exp(-np.clip(logits_raw, -15, 15)))
        
        # 2. 过滤结果
        max_scores = np.max(scores, axis=1)
        max_indices = np.argmax(scores, axis=1)
        
        actual_conf_thres = float(conf_thres) if conf_thres is not None else self.conf_thres
        
        results = []
        for i in range(len(max_scores)):
            if max_scores[i] >= actual_conf_thres:
                # 【关键修正】带保护的坐标解码
                # 即使模型输出的是负数或异常值，我们也强制限制在图像范围内
                cx, cy, bw, bh = boxes_raw[i]
                
                # 如果发现坐标依然不对（如图中万级负数），说明模型输出格式不是 [cx, cy, w, h] 0-1
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h
                
                # 边界剪裁与合法性检查
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
                
                # 如果坐标完全越界（例如 x1 > x2），则跳过该框
                if x1 >= x2 or y1 >= y2:
                    continue

                cls_id = int(max_indices[i])
                # 注意：如果出现“类别张冠李戴”，通常是由于 COCO_CLASSES 顺序不匹配
                # 或者是由于索引位移（例如 0 是背景类，需要 cls_id - 1）
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(max_scores[i]),
                    "class_id": cls_id,
                    "class_name": self.classes[cls_id] if cls_id < len(self.classes) else f"ID_{cls_id}"
                })
        
        # 记录最显著物体的调试信息
        if results:
            best = max(results, key=lambda x: x['conf'])
            print(f"RF-DETR Best Detection: {best['class_name']} conf={best['conf']:.3f} bbox={best['bbox']}")
            
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
