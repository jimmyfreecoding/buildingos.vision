import cv2
import numpy as np
import threading


trt_infer_lock = threading.Lock()


class RFDETRTensorRTEngine:
    def __init__(self, engine_path, conf_thres=0.25, person_class_id=0, max_det=100):
        self.engine_path = engine_path
        self.conf_thres = float(conf_thres)
        self.person_class_id = int(person_class_id)
        self.max_det = int(max_det)
        self.source_name = "rf-detr-trt"

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

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - nw) // 2
        pad_y = (self.input_h - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # 归一化到 0-1
        x = rgb.astype(np.float32) / 255.0
        
        # 针对 RF-DETR / RT-DETR 的标准 ImageNet 归一化 (Mean/Std)
        # 很多 DETR 模型在导出时没有内置这个，需要在预处理做
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x, scale, pad_x, pad_y, w, h

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
            # 调试输出形状，帮助确认模型输出格式
            # print(f"Output [{name}] shape: {meta['shape']}")
        return outputs

    def _parse_outputs(self, outputs, scale, pad_x, pad_y, orig_w, orig_h):
        boxes_arr = None
        logits_arr = None

        # 针对 RT-DETR / RF-DETR 的典型输出结构进行查找
        # 结构 A: [1, 300, 4] boxes 和 [1, 300, 80] logits
        for v in outputs.values():
            if v.ndim == 3:
                if v.shape[-1] == 4:
                    boxes_arr = v[0]
                elif v.shape[-1] > 4:
                    logits_arr = v[0]

        if boxes_arr is None or logits_arr is None:
            # 尝试结构 B: 单一合并输出 [1, 300, 84]
            for v in outputs.values():
                if v.ndim == 3 and v.shape[-1] > 4:
                    boxes_arr = v[0, :, :4]
                    logits_arr = v[0, :, 4:]
                    break

        if boxes_arr is None or logits_arr is None:
            return []

        # RT-DETR 使用 Sigmoid 激活函数处理分类分数
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

        # 针对 RF-DETR 的分数解析优化：
        # 如果模型输出已经是 0-1 之间（即导出的 ONNX 已含 Sigmoid），再次 Sigmoid 会显著降低分数。
        # 我们可以通过检查最大值来启发式判断。
        if np.max(logits_arr) > 1.0 or np.min(logits_arr) < 0.0:
            scores = sigmoid(logits_arr)
        else:
            scores = logits_arr
            
        person_scores = scores[:, self.person_class_id]
        
        # 调试：打印前几个分数的最大值，帮助排查
        print(f"RF-DETR Inference: Raw Max={np.max(logits_arr):.4f}, Final Max={np.max(person_scores):.4f}, Threshold={self.conf_thres}")

        # 复制一份以防修改原数组
        boxes = boxes_arr.copy().astype(np.float32)

        # 归一化坐标转换 (DETR 通常输出 0-1 之间的 cx, cy, w, h)
        if np.max(boxes) <= 1.01:
            # 检查是否是 [cx, cy, w, h] 格式
            # 启发式：如果 x2 < x1 的比例很高，通常是 cx, cy, w, h
            if np.mean(boxes[:, 0] < boxes[:, 2]) > 0.8 and np.max(boxes) <= 1.0:
                # 已经是 x1, y1, x2, y2 归一化格式，直接放大
                boxes[:, [0, 2]] *= self.input_w
                boxes[:, [1, 3]] *= self.input_h
            else:
                # 转换 [cx, cy, w, h] -> [x1, y1, x2, y2] 并放大
                cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = (cx - bw / 2.0) * self.input_w
                y1 = (cy - bh / 2.0) * self.input_h
                x2 = (cx + bw / 2.0) * self.input_w
                y2 = (cy + bh / 2.0) * self.input_h
                boxes = np.stack([x1, y1, x2, y2], axis=1)

        results = []
        # 按置信度排序
        indices = np.where(person_scores >= self.conf_thres)[0]
        if len(indices) == 0:
            return []
            
        filtered_scores = person_scores[indices]
        filtered_boxes = boxes[indices]
        
        order = np.argsort(-filtered_scores)
        for idx in order[: self.max_det]:
            conf = float(filtered_scores[idx])
            x1, y1, x2, y2 = filtered_boxes[idx]
            
            # 还原到原始图像尺寸
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            x1 = int(max(0, min(orig_w - 1, round(x1))))
            y1 = int(max(0, min(orig_h - 1, round(y1))))
            x2 = int(max(0, min(orig_w - 1, round(x2))))
            y2 = int(max(0, min(orig_h - 1, round(y2))))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            results.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": self.person_class_id
            })
        return results

    def predict(self, img):
        if img is None:
            return []
        
        # 关键：确保在推理线程中使用正确的 CUDA Context
        self.cuda_context.push()
        try:
            with trt_infer_lock:
                x, scale, pad_x, pad_y, w, h = self._letterbox(img)
                outputs = self._infer(x)
                return self._parse_outputs(outputs, scale, pad_x, pad_y, w, h)
        finally:
            self.cuda_context.pop()
