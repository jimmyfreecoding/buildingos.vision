import argparse
import os
import sys


def parse_size(size_text: str):
    parts = size_text.lower().replace("x", ",").split(",")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid size: {size_text}, expected like 560x560")
    h = int(parts[0])
    w = int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError("height and width must be positive")
    return h, w


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR ONNX with official rfdetr package")
    parser.add_argument("--variant", default="medium", choices=["nano", "small", "medium", "large"], help="RF-DETR variant")
    parser.add_argument("--size", default="560x560", help="input size, format HxW")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold used in exported graph")
    parser.add_argument("--output", default="ai_engine/models/rf-detr.onnx", help="output onnx path")
    args = parser.parse_args()

    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except Exception as e:
        print("failed to import rfdetr, run: pip install \"rfdetr[onnx]\"")
        print(f"details: {e}")
        return 1

    variant_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    model_cls = variant_map[args.variant]

    h, w = parse_size(args.size)
    ensure_parent(args.output)

    print(f"loading RF-DETR variant: {args.variant}")
    model = model_cls()

    export_name = os.path.splitext(os.path.basename(args.output))[0]
    export_dir = os.path.dirname(os.path.abspath(args.output)) or "."

    print(f"exporting onnx to: {args.output}")
    model.export(
        conf_threshold=args.conf,
        shape=(h, w),
        export_name=export_name,
        export_dir=export_dir,
    )

    expected = os.path.join(export_dir, f"{export_name}.onnx")
    if not os.path.exists(expected):
        print(f"export finished but file not found: {expected}")
        return 2

    print(f"onnx ready: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
