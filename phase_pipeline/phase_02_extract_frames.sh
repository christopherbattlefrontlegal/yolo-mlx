#!/usr/bin/env bash
set -euo pipefail

DB="${DB:?set DB}"
WORKERS="${WORKERS:-8}"
FPS="${FPS:-1}"

ASSETS="$DB/assets.jsonl"
FRAMES_ROOT="$DB/frames"

mkdir -p "$FRAMES_ROOT"

python3 - "$ASSETS" <<'PY' | xargs -0 -P "${WORKERS}" -I{} bash -lc '
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

echo "START frames $asset_id"

ffmpeg -hide_banner -loglevel error -y \
  -i "$path" \
  -vf "fps=$FPS" \
  -q:v 3 \
  "$out/frame_%08d.jpg"

find "$out" -name "frame_*.jpg" | sort > "$out/frames.list"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$donefile"

echo "DONE frames $asset_id"
' _ {}
PY
import json,sys
for line in open(sys.argv[1],encoding="utf-8"):
    r=json.loads(line)
    if r.get("kind")=="video" and not r.get("duplicate"):
        print(json.dumps(r,separators=(",",":")), end="\0")
PY
