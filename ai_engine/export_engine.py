import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 找不到 ultralytics 库。")
    print("请先在宿主机临时执行: pip3 install ultralytics")
    sys.exit(1)

def export_model_to_engine(pt_path: str, engine_path: str, imgsz: int = 640):
    """
    将 YOLO .pt 模型导出为 TensorRT .engine 模型
    """
    if not os.path.exists(pt_path):
        print(f"错误: 找不到模型文件 {pt_path}")
        return False

    if os.path.exists(engine_path):
        print(f"信息: {engine_path} 已存在，如果需要重新编译，请先手动删除旧的 .engine 文件。")
        return True

    print(f"开始编译 {pt_path} 到 TensorRT engine (FP16, imgsz={imgsz})...")
    print("注意: 编译过程可能需要 5-15 分钟，期间 CPU/GPU 占用会很高，请耐心等待。")
    
    try:
        model = YOLO(pt_path)
        # 导出为 TensorRT engine
        # device=0: 使用宿主机的物理 GPU
        # half=True: 开启 FP16 半精度加速，极大提升 Jetson 推理速度
        # workspace=2: 给 TensorRT 限制最多 2GB 显存用于编译优化，防止 Jetson OOM
        # simplify=True: 简化模型结构，加速 TensorRT 优化过程
        model.export(
            format="engine",
            device=0,
            half=True,
            workspace=2,
            imgsz=imgsz,
            simplify=True
        )
        
        # 验证导出结果
        if os.path.exists(engine_path):
            print(f"成功: {engine_path} 编译完成！")
            return True
        else:
            print(f"警告: 命令执行完成，但未在预期路径找到 {engine_path}。")
            print("如果同目录下生成了其他名字的 .engine，请手动重命名。")
            return False
            
    except Exception as e:
        print(f"编译 {pt_path} 时发生错误: {str(e)}")
        return False

def main():
    # 获取当前脚本所在目录的 models 文件夹
    models_dir = Path(__file__).resolve().parent / "models"
    
    # 定义输入和输出路径
    pose_pt = models_dir / "yolov8n-pose.pt"
    pose_engine = models_dir / "yolov8n-pose.engine"
    
    smoke_pt = models_dir / "smoking_v8n.pt"
    smoke_engine = models_dir / "smoking_v8n.engine"
    
    print("=== 开始自动编译 TensorRT 引擎 (宿主机物理机环境) ===")
    
    pose_success = export_model_to_engine(str(pose_pt), str(pose_engine))
    smoke_success = export_model_to_engine(str(smoke_pt), str(smoke_engine))
    
    if pose_success and smoke_success:
        print("\n=== 所有模型编译成功！ ===")
        print("恭喜！你现在可以执行清理命令还原纯净生产环境了：")
        print("pip3 uninstall -y ultralytics torch torchvision")
    else:
        print("\n=== 部分模型编译失败，请检查上方日志。 ===")

if __name__ == "__main__":
    main()
