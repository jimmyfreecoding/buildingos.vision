import argparse
import os
import sys


def register_custom_onnx_symbolics(opset: int):
    try:
        import torch
        from torch.onnx import register_custom_op_symbolic
        from torch.onnx.symbolic_opset11 import upsample_bicubic2d
    except Exception:
        return

    def _upsample_bicubic2d_aa(g, *args):
        if len(args) >= 5:
            return upsample_bicubic2d(g, args[0], args[1], args[2], args[3], args[4])
        return g.op("Identity", args[0])

    try:
        register_custom_op_symbolic("aten::_upsample_bicubic2d_aa", _upsample_bicubic2d_aa, opset)
    except Exception:
        pass


def parse_size(size_text: str):
    parts = size_text.lower().replace("x", ",").split(",")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid size: {size_text}, expected like 640x640")
    h = int(parts[0])
    w = int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError("height and width must be positive")
    if h % 32 != 0 or w % 32 != 0:
        raise ValueError(f"size must be divisible by 32, got {h}x{w}. try 576x576 or 640x640")
    return h, w


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR ONNX with official rfdetr package")
    parser.add_argument("--variant", default="medium", choices=["nano", "small", "medium", "large"], help="RF-DETR variant")
    parser.add_argument("--size", default="640x640", help="input size, format HxW, must be divisible by 32")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold used in exported graph")
    parser.add_argument("--opset", type=int, default=19, help="onnx opset version, recommend >=18")
    parser.add_argument("--output", default="ai_engine/models/rf-detr.onnx", help="output onnx path")
    args = parser.parse_args()

    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except Exception as e:
        print("failed to import rfdetr, run: pip install \"rfdetr[onnx]\"")
        print(f"details: {e}")
        return 1

    register_custom_onnx_symbolics(args.opset)

    variant_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    model_cls = variant_map[args.variant]

    try:
        h, w = parse_size(args.size)
    except ValueError as e:
        print(f"invalid --size: {e}")
        return 3
    ensure_parent(args.output)

    print(f"loading RF-DETR variant: {args.variant}")
    model = model_cls()

    export_name = os.path.splitext(os.path.basename(args.output))[0]
    export_dir = os.path.dirname(os.path.abspath(args.output)) or "."

    print(f"exporting onnx to: {args.output}")
    try:
        model.export(
            conf_threshold=args.conf,
            shape=(h, w),
            export_name=export_name,
            export_dir=export_dir,
            opset_version=args.opset,
        )
    except TypeError:
        model.export(
            conf_threshold=args.conf,
            shape=(h, w),
            export_name=export_name,
            export_dir=export_dir,
        )
    except Exception as e:
        msg = str(e)
        if "aten::_upsample_bicubic2d_aa" in msg:
            print("export failed on aten::_upsample_bicubic2d_aa")
            print("try upgrading torch/torchvision and export on a newer x86 Python env")
            print("recommended: torch>=2.5, torchvision>=0.20, onnx>=1.16, then retry with --opset 19")
            return 4
        raise

    expected = os.path.join(export_dir, f"{export_name}.onnx")
    if not os.path.exists(expected):
        print(f"export finished but file not found: {expected}")
        return 2

    print(f"onnx ready: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
