#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-/app/models/rf-detr.onnx}"
ENGINE_PATH="${2:-/app/models/rf-detr-fp16.engine}"
PRECISION="${3:-fp16}"
INPUT_NAME="${INPUT_NAME:-images}"
MIN_SHAPE="${MIN_SHAPE:-1x3x560x560}"
OPT_SHAPE="${OPT_SHAPE:-1x3x560x560}"
MAX_SHAPE="${MAX_SHAPE:-4x3x560x560}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"

if [[ -x "/usr/src/tensorrt/bin/trtexec" ]]; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
elif command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
else
  echo "trtexec not found"
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ONNX file not found: $ONNX_PATH"
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

COMMON_ARGS=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--workspace=$WORKSPACE_MB"
  "--minShapes=$INPUT_NAME:$MIN_SHAPE"
  "--optShapes=$INPUT_NAME:$OPT_SHAPE"
  "--maxShapes=$INPUT_NAME:$MAX_SHAPE"
  "--builderOptimizationLevel=5"
  "--timingCacheFile=/tmp/rfdetr_timing.cache"
  "--dumpLayerInfo"
  "--dumpProfile"
)

case "${PRECISION,,}" in
  fp16)
    PRECISION_ARGS=("--fp16")
    ;;
  fp32)
    PRECISION_ARGS=()
    ;;
  int8)
    if [[ -z "${CALIB_CACHE:-}" ]]; then
      echo "CALIB_CACHE is required when PRECISION=int8"
      exit 1
    fi
    PRECISION_ARGS=("--int8" "--calib=$CALIB_CACHE")
    ;;
  *)
    echo "Unsupported precision: $PRECISION (fp16|fp32|int8)"
    exit 1
    ;;
esac

"$TRTEXEC" "${COMMON_ARGS[@]}" "${PRECISION_ARGS[@]}"

echo "Engine built: $ENGINE_PATH"
