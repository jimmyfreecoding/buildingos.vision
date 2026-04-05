#!/usr/bin/env bash
set -euo pipefail

exec /home/buildingos/ai/llama.cpp/build/bin/llama-server \
  -m /home/buildingos/ai/llama.cpp/models/gemma4/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /home/buildingos/ai/llama.cpp/models/gemma4/mmproj-F16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 40 \
  --threads 4 \
  --parallel 1 \
  --alias buildingos_review_engine
