#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-/app/models/rf-detr.onnx}"
ENGINE_PATH="${2:-/app/models/rf-detr-fp16.engine}"
PRECISION="${3:-fp16}"
INPUT_NAME="${INPUT_NAME:-images}"
MIN_SHAPE="${MIN_SHAPE:-1x3x640x640}"
OPT_SHAPE="${OPT_SHAPE:-1x3x640x640}"
MAX_SHAPE="${MAX_SHAPE:-2x3x640x640}"
WORKSPACE_MB="${WORKSPACE_MB:-512}"
BUILD_MODE="${BUILD_MODE:-static}"
OPT_LEVEL="${OPT_LEVEL:-2}"
SKIP_INFERENCE="${SKIP_INFERENCE:-1}"
USE_TIMING_CACHE="${USE_TIMING_CACHE:-0}"
TIMING_CACHE_FILE="${TIMING_CACHE_FILE:-/tmp/rfdetr_timing.cache}"
MAX_AUX_STREAMS="${MAX_AUX_STREAMS:-0}"
ALLOCATION_STRATEGY="${ALLOCATION_STRATEGY:-runtime}"
ENABLE_PREVIEW_PROFILE_SHARING="${ENABLE_PREVIEW_PROFILE_SHARING:-1}"

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

TRTEXEC_HELP="$("$TRTEXEC" --help 2>&1 || true)"

COMMON_ARGS=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--memPoolSize=workspace:$WORKSPACE_MB"
  "--builderOptimizationLevel=$OPT_LEVEL"
)

if [[ "$USE_TIMING_CACHE" == "1" ]]; then
  COMMON_ARGS+=("--timingCacheFile=$TIMING_CACHE_FILE")
fi

if echo "$TRTEXEC_HELP" | grep -q -- "--maxAuxStreams"; then
  COMMON_ARGS+=("--maxAuxStreams=$MAX_AUX_STREAMS")
fi

if echo "$TRTEXEC_HELP" | grep -q -- "--allocationStrategy"; then
  COMMON_ARGS+=("--allocationStrategy=$ALLOCATION_STRATEGY")
fi

if [[ "$ENABLE_PREVIEW_PROFILE_SHARING" == "1" ]] && echo "$TRTEXEC_HELP" | grep -q -- "--preview"; then
  COMMON_ARGS+=("--preview=+profileSharing0806")
fi

if [[ "${BUILD_MODE,,}" == "dynamic" ]]; then
  COMMON_ARGS+=(
    "--minShapes=$INPUT_NAME:$MIN_SHAPE"
    "--optShapes=$INPUT_NAME:$OPT_SHAPE"
    "--maxShapes=$INPUT_NAME:$MAX_SHAPE"
  )
else
  COMMON_ARGS+=("--shapes=$INPUT_NAME:$OPT_SHAPE")
fi

if [[ "$SKIP_INFERENCE" == "1" ]]; then
  COMMON_ARGS+=("--skipInference")
fi

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
