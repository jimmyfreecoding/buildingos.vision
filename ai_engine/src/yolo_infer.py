import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

class YoloTensorRTEngine:
    """
    轻量级的 YOLOv8 TensorRT 推理引擎封装。
    完全不依赖 PyTorch 或 Ultralytics 库，专为 Jetson 等边缘设备的生产部署设计。
    """
    def __init__(self, engine_path, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        self.engine_path = engine_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 初始化 TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # 解析输入输出张量信息并分配显存 (PyCUDA)
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine)
        
        # 提取模型真实的输入分辨率 (通常是 [1, 3, 640, 640])
        input_shape = self.engine.get_tensor_shape(self.inputs[0]['name'])
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
             self.imgsz = input_shape[2]

    def _load_engine(self, engine_path):
        """反序列化加载 .engine 文件"""
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"加载 TensorRT 引擎失败: {engine_path}")
        return engine

    def _allocate_buffers(self, engine):
        """分配主机(CPU)和设备(GPU)内存"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        # TensorRT 8.5+ 推荐使用 get_tensor_name 和 get_tensor_shape
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            # 分配 Page-locked (pinned) CPU 内存，加速传输
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配 GPU 显存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': engine.get_tensor_shape(tensor_name)})
            else:
                outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': engine.get_tensor_shape(tensor_name)})
                
        return inputs, outputs, bindings, stream

    def _preprocess(self, img):
        """图像预处理：LetterBox 缩放 -> BGR转RGB -> HWC -> 归一化"""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = (self.imgsz - new_unpad[0]) / 2, (self.imgsz - new_unpad[1]) / 2

        if (w, h) != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img_input = img_padded.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
        img_input = np.ascontiguousarray(img_input).astype(np.float32)
        img_input /= 255.0  # 归一化
        img_input = np.expand_dims(img_input, axis=0) # [1, 3, 640, 640]
        
        return img_input, (r, dw, dh, w, h)

    def _postprocess(self, output_buffer, meta, output_shape):
        """解析 TensorRT 的输出并执行 NMS"""
        # 恢复张量形状并去掉 batch 维度，转置为 (8400, classes_len)
        predictions = output_buffer.reshape(output_shape)
        predictions = np.squeeze(predictions).T
        
        r, dw, dh, orig_w, orig_h = meta
        boxes, scores, class_ids, keypoints = [], [], [], []

        # YOLOv8 Pose 输出第二维度为 56, 普通检测通常为 84 (COCO)
        is_pose = predictions.shape[1] == 56

        for pred in predictions:
            box = pred[0:4]
            if is_pose:
                conf = pred[4]
                class_id = 0
                kpts = pred[5:]
            else:
                class_probs = pred[4:]
                class_id = np.argmax(class_probs)
                conf = class_probs[class_id]
                kpts = None

            if conf > self.conf_thres:
                cx, cy, w, h = box
                cx = (cx - dw) / r
                cy = (cy - dh) / r
                w = w / r
                h = h / r
                
                x1 = max(0, min(int(cx - w / 2), orig_w - 1))
                y1 = max(0, min(int(cy - h / 2), orig_h - 1))
                x2 = max(0, min(int(cx + w / 2), orig_w - 1))
                y2 = max(0, min(int(cy + h / 2), orig_h - 1))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(conf))
                class_ids.append(class_id)
                if is_pose:
                    keypoints.append(kpts)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                res = {
                    "bbox": [x, y, x + w, y + h],
                    "conf": scores[i],
                    "class_id": class_ids[i]
                }
                if is_pose:
                    res["keypoints_raw"] = keypoints[i]
                results.append(res)
                
        return results

    def predict(self, img):
        """执行 TensorRT 推理"""
        if img is None:
            return []
            
        t1 = time.time()
        
        # 1. 预处理并将数据拷贝到 CPU 锁页内存
        input_tensor, meta = self._preprocess(img)
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        
        # 2. 将数据从 CPU 传输到 GPU (H2D)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. 执行推理 (异步)
        # TensorRT 8.5+ 使用 execute_async_v3
        # 我们需要在 context 绑定内存地址
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])
            
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 4. 将结果从 GPU 传回 CPU (D2H)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        # 5. 等待流同步
        self.stream.synchronize()
        
        # 6. 后处理
        # 取第一个输出进行解析 (YOLOv8 通常只有一个输出张量)
        results = self._postprocess(self.outputs[0]['host'], meta, self.outputs[0]['shape'])
        
        t2 = time.time()
        # print(f"[{self.engine_path}] 推理耗时: {(t2 - t1) * 1000:.1f} ms")
        
        return results

# Jetson 上的使用示例：
if __name__ == "__main__":
    from pathlib import Path
    import os
    
    models_dir = Path(__file__).resolve().parent / "models"
    pose_engine = models_dir / "yolov8n-pose.engine"
    smoke_engine = models_dir / "smoking_v8n.engine"
    
    if os.path.exists(pose_engine):
        print("初始化 Jetson TensorRT 引擎...")
        engine = YoloTensorRTEngine(str(pose_engine), conf_thres=0.3)
        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        print(f"测试推理结果: {engine.predict(test_img)}")
    else:
        print(f"找不到模型文件: {pose_engine}")
