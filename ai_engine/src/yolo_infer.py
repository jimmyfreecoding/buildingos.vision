import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import json
import os

# 全局推理锁，防止多线程同时调用 TensorRT 引擎导致 double free
trt_infer_lock = threading.Lock()

class YoloTensorRTEngine:
    """
    纯 TensorRT 实现，彻底脱离 Torch/Ultralytics 依赖。
    支持 YOLOv8/v11 的检测 (Detect) 和姿态 (Pose) 任务。
    """
    def __init__(self, engine_path, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        self.engine_path = engine_path
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 1. 加载引擎并处理 Ultralytics Metadata
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        engine_data = self._load_engine_data(engine_path)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 2. 分配显存/内存缓冲区
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            # 分配页锁定内存 (Host)
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配显存 (Device)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': binding})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': binding, 'shape': self.engine.get_tensor_shape(binding)})

        # 自动推断任务类型
        self.is_pose = 'pose' in engine_path.lower()
        task_str = "POSE" if self.is_pose else "DETECT"
        print(f"✅ 纯 TensorRT 引擎加载成功: {os.path.basename(engine_path)} (Task: {task_str})")

    def _load_engine_data(self, path):
        """读取引擎文件并自动跳过 Ultralytics 的 JSON Metadata 头部"""
        with open(path, 'rb') as f:
            data = f.read()
            
        # 检查是否包含 Ultralytics 的 JSON 头部 (通常以 { 开头)
        if data.startswith(b'{'):
            try:
                # 寻找 JSON 结束标记后的第一个非空字符
                # Ultralytics 的格式通常是: {JSON}\0\0\0...TRT_MAGIC...
                # 寻找 '7b' ({) 到 第一个 'ptr' (TensorRT 的 magic tag)
                # 实际上最简单的方法是寻找 "ms" (magic string)
                # 或者寻找第一个 0x00 后的非 0 区域
                # 这里使用更稳健的方法：定位 TensorRT 的序列化特征
                idx = data.find(b'ptr', 0, 1024) # TensorRT 10.x 之前的特征
                if idx == -1:
                    idx = data.find(b'pt7', 0, 1024) # TensorRT 10.x 之后的特征
                
                if idx != -1:
                    # 向上寻找第一个 0x00 (JSON 后的填充)
                    # 实际上直接从 idx 开始就是正确的 TRT 序列化数据
                    return data[idx:]
            except:
                pass
        return data

    def predict(self, img, conf_thres=None):
        if img is None:
            return []
            
        actual_conf = conf_thres if conf_thres is not None else self.conf_thres
        
        # 1. 预处理
        blob, ratio, (pad_w, pad_h) = self.preprocess(img)
        
        # 2. 推理 (加锁防止 double free)
        with trt_infer_lock:
            np.copyto(self.inputs[0]['host'], blob.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # 执行异步推理 (适配不同版本的 TensorRT API)
            try:
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            except:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
        
        # 3. 后处理 (NMS)
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return self.postprocess(output, img.shape[:2], ratio, (pad_w, pad_h), actual_conf)

    def preprocess(self, img):
        """保持比例的缩放 (Letterbox)"""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # HWC -> CHW, BGR -> RGB, /255.0
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img, r, (dw, dh)

    def postprocess(self, output, orig_shape, ratio, pad, conf_thres):
        """解析 YOLOv8/v11 的输出张量"""
        # output shape: (1, 4 + num_classes [+ 51 if pose], 8400)
        output = output[0] # remove batch dim -> (C, 8400)
        output = output.transpose() # -> (8400, C)
        
        # YOLOv8-Pose: [x, y, w, h, score, kpt1_x, kpt1_y, kpt1_conf, ...]
        # YOLOv8-Detect: [x, y, w, h, cls1_score, cls2_score, ...]
        
        boxes_for_nms = []
        scores = []
        class_ids = []
        keypoints = []
        final_boxes = []
        
        if self.is_pose:
            # Pose 任务通常只有一个类别 (Person)
            mask = output[:, 4] > conf_thres
            valid_output = output[mask]
            
            if len(valid_output) == 0: return []
            
            curr_boxes = valid_output[:, :4]
            curr_scores = valid_output[:, 4]
            curr_kpts = valid_output[:, 5:]
            
            for i in range(len(valid_output)):
                x, y, w, h = curr_boxes[i]
                x1 = (x - w/2 - pad[0]) / ratio
                y1 = (y - h/2 - pad[1]) / ratio
                x2 = (x + w/2 - pad[0]) / ratio
                y2 = (y + h/2 - pad[1]) / ratio
                
                # NMSBoxes 需要 [x, y, w, h]
                boxes_for_nms.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                final_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(curr_scores[i]))
                class_ids.append(0) 
                
                kpts = curr_kpts[i].reshape(-1, 3)
                kpts[:, 0] = (kpts[:, 0] - pad[0]) / ratio
                kpts[:, 1] = (kpts[:, 1] - pad[1]) / ratio
                keypoints.append(kpts.flatten().tolist())
        else:
            # 检测任务
            num_classes = output.shape[1] - 4
            all_scores = output[:, 4:]
            max_scores = np.max(all_scores, axis=1)
            mask = max_scores > conf_thres
            valid_output = output[mask]
            
            if len(valid_output) == 0: return []
            
            curr_boxes = valid_output[:, :4]
            curr_scores = np.max(valid_output[:, 4:], axis=1)
            curr_cls = np.argmax(valid_output[:, 4:], axis=1)
            
            for i in range(len(valid_output)):
                x, y, w, h = curr_boxes[i]
                x1 = (x - w/2 - pad[0]) / ratio
                y1 = (y - h/2 - pad[1]) / ratio
                x2 = (x + w/2 - pad[0]) / ratio
                y2 = (y + h/2 - pad[1]) / ratio
                
                boxes_for_nms.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                final_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(curr_scores[i]))
                class_ids.append(int(curr_cls[i]))
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf_thres, self.iou_thres)
        
        final_results = []
        if len(indices) > 0:
            for i in indices.flatten():
                res = {
                    "bbox": final_boxes[i],
                    "conf": scores[i],
                    "class_id": class_ids[i]
                }
                if self.is_pose:
                    res["keypoints_raw"] = keypoints[i]
                final_results.append(res)
                
        return final_results
