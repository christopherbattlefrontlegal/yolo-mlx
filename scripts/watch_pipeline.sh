#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/Volumes/VAULT/_machine_/mlx_detections_stage}"

while true; do
  clear
  echo "YOLO MLX PIPELINE WATCH"
  echo "root: $ROOT"
  echo "time: $(date)"
  echo

  printf "%-10s %-12s %-12s %-10s %-10s %-10s %s\n" "STATE" "DONE" "TOTAL" "FPS" "ETA(min)" "DETS" "ASSET"
  echo "--------------------------------------------------------------------------------"

  find "$ROOT" -name status.json -maxdepth 2 2>/dev/null | sort | while read -r f; do
    python3 - "$f" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    s = json.loads(p.read_text())
except Exception:
    raise SystemExit
eta = s.get("eta_sec")
eta_min = "" if eta is None else f"{eta/60:.1f}"
print(f"{s.get('state',''):<10} {s.get('frames_done',0):<12} {s.get('frames_total',0):<12} {s.get('fps',0):<10.2f} {eta_min:<10} {s.get('detections_written',0):<10} {s.get('asset_id','')[:12]}")
PY
  done

  echo
  echo "recent logs:"
  ls -t /tmp/yolo_mlx_logs/*.log 2>/dev/null | head -4 | while read -r log; do
    echo "--- $(basename "$log") ---"
    tail -3 "$log"
  done

  sleep 2
done
