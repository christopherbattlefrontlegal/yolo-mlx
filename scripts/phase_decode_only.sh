#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production"
OUT="/Volumes/VAULT/_machine_/decode_stage"

WORKERS=1

mkdir -p "$OUT"

find "$ROOT" -type f \( \
  -iname "*.mp4" -o \
  -iname "*.mov" -o \
  -iname "*.mkv" -o \
  -iname "*.avi" \
\) -print0 |
xargs -0 -P "$WORKERS" -I{} bash -lc '

v="$1"

id=$(printf "%s" "$v" | shasum -a 256 | awk "{print \$1}")

d="'"$OUT"'/$id"

mkdir -p "$d"

echo "START $v"

ffmpeg \
  -hide_banner \
  -loglevel error \
  -threads 0 \
  -fflags +genpts \
  -i "$v" \
  -vf "setpts=N/FRAME_RATE/TB" \
  -fps_mode passthrough \
  "$d/frame_%08d.jpg"

echo "DONE $v"

' _ {}
