#!/usr/bin/env bash
set -euo pipefail

source phase_pipeline/bench_env.sh
export PHASE="phase_02_extract_frames"
print_bench_env

ASSETS="$DB/assets.jsonl"
FRAMES_ROOT="$DB/frames"
mkdir -p "$FRAMES_ROOT"

python3 - "$ASSETS" "$MAX_ASSETS" <<'PY' | xargs -0 -P "$WORKERS" -I{} bash -lc '
set -euo pipefail

line="$1"
asset_id=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])[\"asset_id\"])" "$line")
path=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])[\"path\"])" "$line")

out="$FRAMES_ROOT/$asset_id"
donefile="$out/.done"

if [ -f "$donefile" ]; then
  echo "SKIP frames $asset_id"
  exit 0
fi

mkdir -p "$out"
printf "%s\n" "$path" > "$out/source_path.txt"

echo "START frames asset=$asset_id path=$path"

t0=$(date +%s)

ffmpeg -hide_banner -loglevel error -y \
  -i "$path" \
  -vsync 0 \
  -q:v 3 \
  "$out/frame_%08d.jpg"

find "$out" -name "frame_*.jpg" | sort > "$out/frames.list"
count=$(wc -l < "$out/frames.list" | tr -d " ")
t1=$(date +%s)
elapsed=$((t1-t0))
[ "$elapsed" -le 0 ] && elapsed=1

{
  echo "{"
  echo "  \"asset_id\": \"$asset_id\","
  echo "  \"source_path\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$path"),"
  echo "  \"frames\": $count,"
  echo "  \"elapsed_sec\": $elapsed,"
  echo "  \"frames_per_sec\": $(python3 -c "print(round($count/$elapsed, 3))"),"
  echo "  \"frame_mode\": \"$FRAME_MODE\","
  echo "  \"fps_filter\": \"$FPS_FILTER\","
  echo "  \"workers\": \"$WORKERS\""
  echo "}"
} > "$out/phase_02_manifest.json"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$donefile"

echo "DONE frames asset=$asset_id frames=$count elapsed=${elapsed}s fps=$(python3 -c "print(round($count/$elapsed, 2))")"
' _ {}
PY
import json,sys
assets=sys.argv[1]
max_assets=int(sys.argv[2])
n=0
for line in open(assets,encoding="utf-8"):
    r=json.loads(line)
    if r.get("kind")=="video" and not r.get("duplicate"):
        print(json.dumps(r,separators=(",",":")), end="\0")
        n+=1
        if max_assets > 0 and n >= max_assets:
            break
PY
