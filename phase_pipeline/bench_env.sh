#!/usr/bin/env bash
set -euo pipefail

export ROOT="/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production"
export DB="/Volumes/VAULT/_human_/Jorge_Perez_Discovery/perez_db"

export WORKERS=1
export FRAME_MODE="all_source_frames"
export FPS_FILTER="none"
export MAX_ASSETS=2
export MAX_FRAMES=0

export MODEL="/Volumes/VAULT/_machine_/yolo11_coreml/yoloe-11l-seg-forensic.onnx"
export PROVIDERS="CoreMLExecutionProvider,CPUExecutionProvider"
export ORT_THREADS=1

print_bench_env() {
  echo "===== BENCH SETTINGS ====="
  echo "PHASE=${PHASE:-unset}"
  echo "ROOT=$ROOT"
  echo "DB=$DB"
  echo "WORKERS=$WORKERS"
  echo "FRAME_MODE=$FRAME_MODE"
  echo "FPS_FILTER=$FPS_FILTER"
  echo "MAX_ASSETS=$MAX_ASSETS"
  echo "MAX_FRAMES=$MAX_FRAMES"
  echo "MODEL=$MODEL"
  echo "PROVIDERS=$PROVIDERS"
  echo "ORT_THREADS=$ORT_THREADS"
  echo "START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=========================="
}
