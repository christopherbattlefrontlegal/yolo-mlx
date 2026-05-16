#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

from yolo26mlx import YOLO

CHUNK = 1024 * 1024


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def boxes_to_json(result):
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    names = getattr(result, "names", {}) or {}

    xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else list(boxes.xyxy)
    conf = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
    cls = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)

    for i, box in enumerate(xyxy):
        class_id = int(cls[i]) if i < len(cls) else None
        out.append({
            "det_id": str(i),
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id),
            "confidence": float(conf[i]) if i < len(conf) else None,
            "bbox_xyxy": [float(v) for v in box],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", default="models/yolo26n-seg.npz")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if args.limit > 0:
        frames = frames[:args.limit]

    asset_id = frames_dir.name
    asset_dir = out_root / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    detections_path = asset_dir / "detections.jsonl"
    manifest_path = asset_dir / "manifest.json"
    status_path = asset_dir / "status.json"
    source_path = asset_dir / "source_frames_dir.txt"
    source_path.write_text(str(frames_dir) + "\n", encoding="utf-8")

    model = YOLO(args.model, task="segment")

    t0 = time.perf_counter()
    written = 0
    det_total = 0

    with open(detections_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
        for start in range(0, len(frames), args.chunk):
            batch = frames[start:start + args.chunk]
            results = model.predict(batch, conf=args.conf, imgsz=args.imgsz, save=False, stream=False, rect=True)

            for local_i, result in enumerate(results):
                frame_index = start + local_i
                detections = boxes_to_json(result)
                rec = {
                    "asset_id": asset_id,
                    "frame_index": frame_index,
                    "frame_path": str(batch[local_i]),
                    "detections": detections,
                }
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                det_total += len(detections)

            dt = max(time.perf_counter() - t0, 1e-9)
            status = {
                "state": "running",
                "asset_id": asset_id,
                "frames_total": len(frames),
                "frames_done": written,
                "frames_remaining": max(len(frames) - written, 0),
                "detections_written": det_total,
                "fps": written / dt,
                "elapsed_sec": dt,
                "eta_sec": (len(frames) - written) / (written / dt) if written else None,
                "chunk": args.chunk,
                "model": args.model,
            }
            tmp_status = status_path.with_suffix(".json.tmp")
            tmp_status.write_text(json.dumps(status, indent=2), encoding="utf-8")
            tmp_status.replace(status_path)
            print(f"\rframes={written}/{len(frames)} fps={written/dt:.1f} dets={det_total}", end="", flush=True)

    elapsed = time.perf_counter() - t0
    manifest = {
        "asset_id": asset_id,
        "source_frames_dir": str(frames_dir),
        "model": args.model,
        "output": str(detections_path),
        "frames_written": written,
        "detections_written": det_total,
        "elapsed_sec": elapsed,
        "fps": written / elapsed if elapsed else None,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "chunk": args.chunk,
    }

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)

    print("\nDONE")
    print("frames:", written)
    print("fps:", round(written / elapsed, 2) if elapsed else None)
    print("out:", detections_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
