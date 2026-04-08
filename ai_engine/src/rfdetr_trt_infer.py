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
        针对 aux1 模型优化的解析逻辑：
        1. 多头探测：寻找主头 (output0 或 shape[1]==300, shape[2]==84)
        2. 自动 Sigmoid：如果坐标不在 0-1 之间，尝试对其进行映射
        """
        # --- 1. 深度诊断日志 ---
        print("\n" + "-"*50)
        print("RF-DETR aux1 ENGINE DIAGNOSTICS")
        for k, v in outputs.items():
            print(f"Tensor: {k:20} | Shape: {str(v.shape):15} | Range: [{np.min(v):.2f}, {np.max(v):.2f}]")
        print("-"*50 + "\n")

        boxes_raw = None
        logits_raw = None

        # 2. 锁定主头：优先使用 'output0'，否则找 shape[2] >= 84 的张量
        main_v = None
        if 'output0' in outputs:
            main_v = outputs['output0'][0]
        else:
            # 兜底：寻找最后一个符合形状的（通常最终头在最后或第一个）
            for v in outputs.values():
                if v.ndim == 3 and v.shape[-1] >= 84:
                    main_v = v[0]
                    break
        
        if main_v is not None:
            boxes_raw = main_v[:, :4]
            logits_raw = main_v[:, 4:84]
        else:
            # 兼容分离的 boxes 和 scores
            for k, v in outputs.items():
                if v.ndim == 3 and v.shape[-1] == 4: boxes_raw = v[0]
                elif v.ndim == 3 and v.shape[-1] >= 80: logits_raw = v[0]

        if boxes_raw is None or logits_raw is None:
            print("ERROR: No valid detection heads found!")
            return []

        # 3. 执行分类分数计算
        # 强制执行 Sigmoid 以确保分数在 0-1 之间
        scores = 1 / (1 + np.exp(-np.clip(logits_raw, -15, 15)))
        
        # 4. 坐标自适应解码
        # 如果坐标数值超出了合理的 [0, 1] 比例范围，极有可能是 Engine 导出时未包含 Sigmoid 映射
        box_min, box_max = np.min(boxes_raw), np.max(boxes_raw)
        needs_sigmoid = (box_min < -0.1 or box_max > 1.1)
        
        if needs_sigmoid:
            # 强制映射到 0-1 (DETR 协议)
            cx_cy_w_h = 1 / (1 + np.exp(-np.clip(boxes_raw, -15, 15)))
            # print(f"DEBUG: Auto-applied Sigmoid to box coordinates (Range was [{box_min:.2f}, {box_max:.2f}])")
        else:
            cx_cy_w_h = boxes_raw

        # 5. 过滤结果
        max_scores = np.max(scores, axis=1)
        max_indices = np.argmax(scores, axis=1)
        
        actual_conf_thres = float(conf_thres) if conf_thres is not None else self.conf_thres
        results = []

        for i in range(len(max_scores)):
            if max_scores[i] >= actual_conf_thres:
                cx, cy, bw, bh = cx_cy_w_h[i]
                
                # DETR 标准转换：[cx, cy, w, h] -> [x1, y1, x2, y2]
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h
                
                # 裁剪与边界保护
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
                
                if x1 < x2 and y1 < y2:
                    cls_id = int(max_indices[i])
                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(max_scores[i]),
                        "class_id": cls_id,
                        "class_name": self.classes[cls_id] if cls_id < len(self.classes) else f"ID_{cls_id}"
                    })
        
        if results:
            best = max(results, key=lambda x: x['conf'])
            print(f"✅ SUCCESS: Detected {best['class_name']} with confidence {best['conf']:.3f}")
            
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
