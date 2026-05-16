#!/usr/bin/env bash
set -euo pipefail

CASE_ROOT="/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production/25AGJ201B/C-25-395076-2_credit_card"
DECODE_ROOT="/Volumes/VAULT/_machine_/decode_stage"
DETECT_ROOT="/Volumes/VAULT/_machine_/mlx_detections_stage"
MODEL="models/yolo26n-seg.npz"

mkdir -p "$DECODE_ROOT" "$DETECT_ROOT" /tmp/yolo_mlx_logs

echo "STEP 0/1: hash unique videos and decode frames"
find "$CASE_ROOT" -type f \( \
  -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" \
\) -print0 | sort -z | while IFS= read -r -d '' v; do
  sha="$(shasum -a 256 "$v" | awk '{print $1}')"
  out="$DECODE_ROOT/$sha"
  donefile="$out/.decode_done"

  if [ -f "$donefile" ]; then
    echo "DECODE SKIP $sha $v"
    continue
  fi

  mkdir -p "$out"
  echo "$v" > "$out/source_path.txt"
  echo "DECODE START $sha $v"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$v" \
    -vsync 0 \
    -q:v 3 \
    "$out/frame_%08d.jpg" || echo "DECODE ERROR $sha $v" > "$out/.decode_error"

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$donefile"
  echo "DECODE DONE $sha"
done

echo "STEP 2: detect decoded frame folders with 2 workers"

find "$DECODE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | \
xargs -I{} -P 2 bash -lc '
  d="$1"
  sha="$(basename "$d")"
  if [ ! -f "$d/.decode_done" ]; then
    echo "DETECT SKIP no decode_done $sha"
    exit 0
  fi
  if [ -f "'"$DETECT_ROOT"'/$sha/manifest.json" ]; then
    echo "DETECT SKIP done $sha"
    exit 0
  fi
  echo "DETECT START $sha"
  python3 scripts/mlx_frames_to_jsonl.py \
    --frames-dir "$d" \
    --out-root "'"$DETECT_ROOT"'" \
    --model "'"$MODEL"'" \
    --conf 0.55 \
    --chunk 250 \
    > "/tmp/yolo_mlx_logs/$sha.log" 2>&1
  echo "DETECT DONE $sha"
' _ {}

echo "DONE"
echo "decode root: $DECODE_ROOT"
echo "detect root: $DETECT_ROOT"
echo "logs: /tmp/yolo_mlx_logs"
