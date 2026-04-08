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
        
        # 核心调整：先只尝试 0-1 归一化。
        # 如果模型内部没做 mean/std，0-1 也会有基本识别率；
        # 如果模型内部做了，再做一次会导致识别率降为 0。
        x = rgb.astype(np.float32) / 255.0
        
        # 暂时注释掉 Mean/Std，观察是否恢复人员识别
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

    def _parse_outputs(self, outputs, scale_x, scale_y, orig_w, orig_h):
        # 1. 获取所有输出张量并打印形状（仅执行一次，用于调试）
        for name, meta in self.tensor_meta.items():
            if name in outputs:
                shape = outputs[name].shape
                # print(f"DEBUG: Output [{name}] shape: {shape}")

        # 2. 寻找 84 列或 80+ 列的主输出
        logits_arr = None
        boxes_arr = None
        main_tensor = None
        
        for v in outputs.values():
            if v.ndim == 3 and v.shape[-1] >= 84:
                main_tensor = v[0]
                break
        
        if main_tensor is None:
            # 备选：如果分开输出 [300, 4] 和 [300, 80]
            for v in outputs.values():
                if v.ndim == 3:
                    if v.shape[-1] == 4: boxes_arr = v[0]
                    elif v.shape[-1] >= 80: logits_arr = v[0]
        else:
            # 核心假设：RF-DETR 可能是 [scores(80), boxes(4)] 格式
            # 我们根据最后一列是否在 0-1 之间且具有坐标特征来探测
            # 尝试 格式 A: [boxes(4), scores(80)]
            # 尝试 格式 B: [scores(80), boxes(4)]
            if np.mean(main_tensor[:, :4]) < 1.0 and np.max(main_tensor[:, :4]) <= 1.01:
                # 看起来前4列更像坐标
                boxes_arr = main_tensor[:, :4]
                logits_arr = main_tensor[:, 4:84]
                # print("DEBUG: Detected Format A [boxes, scores]")
            else:
                # 看起来后4列更像坐标
                logits_arr = main_tensor[:, :80]
                boxes_arr = main_tensor[:, 80:84]
                # print("DEBUG: Detected Format B [scores, boxes]")

        if boxes_arr is None or logits_arr is None: return []

        # 3. 激活函数
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
        
        scores = sigmoid(logits_arr)
        
        # 4. 坐标解码：支持 [cx, cy, w, h] -> [x1, y1, x2, y2]
        # DETR 模型几乎全部输出中心点格式
        boxes = boxes_arr.copy().astype(np.float32)
        
        # 启发式判断：如果 x2 < x1 的情况很多，说明是 [cx, cy, w, h]
        if np.mean(boxes[:, 0] > boxes[:, 2]) > 0.3 or np.mean(boxes[:, 1] > boxes[:, 3]) > 0.3:
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 5. 提取结果
        results = []
        all_max_scores = np.max(scores, axis=1)
        all_max_indices = np.argmax(scores, axis=1)
        
        # 打印 Top 1 调试
        best_idx = np.argmax(all_max_scores)
        best_cls = all_max_indices[best_idx]
        best_name = self.classes[best_cls] if best_cls < len(self.classes) else f"ID_{best_cls}"
        print(f"RF-DETR Best Det: {best_name} ({best_cls}) score={all_max_scores[best_idx]:.3f}")

        indices = np.where(all_max_scores >= self.conf_thres)[0]
        for idx in indices:
            conf = float(all_max_scores[idx])
            cls_id = int(all_max_indices[idx])
            
            x1, y1, x2, y2 = boxes[idx]
            x1, x2 = x1 * orig_w, x2 * orig_w
            y1, y2 = y1 * orig_h, y2 * orig_h
            
            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "class_id": cls_id,
                "class_name": self.classes[cls_id] if cls_id < len(self.classes) else f"cls_{cls_id}"
            })
        return results

    def predict(self, img):
        if img is None:
            return []
        
        # 关键：确保在推理线程中使用正确的 CUDA Context
        self.cuda_context.push()
        try:
            with trt_infer_lock:
                x, scale_x, scale_y, w, h = self._preprocess(img)
                outputs = self._infer(x)
                return self._parse_outputs(outputs, scale_x, scale_y, w, h)
        finally:
            self.cuda_context.pop()
