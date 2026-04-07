import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Prepare RF-DETR ONNX for TensorRT on Jetson")
    parser.add_argument("--input", required=True, help="input onnx path")
    parser.add_argument("--output", required=True, help="output onnx path")
    parser.add_argument("--shape", default="1,3,576,576", help="static input shape as N,C,H,W")
    parser.add_argument("--target-opset", type=int, default=16, help="convert opset version for TRT parser compatibility")
    args = parser.parse_args()

    try:
        import onnx
        from onnx import checker, shape_inference, version_converter
    except Exception as e:
        print(f"require onnx package: {e}")
        return 1

    if not os.path.exists(args.input):
        print(f"input not found: {args.input}")
        return 2

    model = onnx.load(args.input)
    checker.check_model(model)

    shape_vals = [int(x.strip()) for x in args.shape.split(",")]
    if len(shape_vals) != 4:
        print("shape must be N,C,H,W")
        return 3

    input_name = model.graph.input[0].name
    dims = model.graph.input[0].type.tensor_type.shape.dim
    for i, v in enumerate(shape_vals):
        dims[i].dim_value = v
        dims[i].dim_param = ""

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    current_opset = model.opset_import[0].version
    if current_opset != args.target_opset:
        try:
            model = version_converter.convert_version(model, args.target_opset)
        except Exception as e:
            print(f"opset convert failed ({current_opset}->{args.target_opset}): {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save(model, args.output)

    try:
        import onnxsim
        sim_model, ok = onnxsim.simplify(
            args.output,
            overwrite_input_shapes={input_name: shape_vals},
            dynamic_input_shape=False,
        )
        if ok:
            onnx.save(sim_model, args.output)
    except Exception as e:
        print(f"onnxsim skipped: {e}")

    print(f"prepared onnx: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
