#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from yolo26mlx import YOLO

CHUNK = 1024 * 1024
W = 640
H = 640
FRAME_BYTES = W * H * 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def ffprobe_fps(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        n, d = out.split("/")
        return float(n) / float(d) if float(d) else 30.0
    except Exception:
        return 30.0


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


def read_exact(pipe, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", default="models/yolo26n-seg.npz")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-fps", type=float, default=5.0)
    args = ap.parse_args()

    video = Path(args.video)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    asset_id = sha256_file(video)
    asset_dir = out_root / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    detections_path = asset_dir / "detections.jsonl"
    manifest_path = asset_dir / "manifest.json"
    source_path = asset_dir / "source_path.txt"
    source_path.write_text(str(video) + "\n", encoding="utf-8")

    fps = ffprobe_fps(video)
    model = YOLO(args.model, task="segment")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video),
        "-vf", (f"fps={args.sample_fps},scale={W}:{H}" if args.sample_fps > 0 else f"scale={W}:{H}"),
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    t0 = time.perf_counter()
    frame_index = 0
    written = 0
    det_total = 0

    try:
        with open(detections_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
            while True:
                imgs = []
                idxs = []

                for _ in range(args.batch):
                    if args.limit > 0 and frame_index >= args.limit:
                        break

                    raw = read_exact(proc.stdout, FRAME_BYTES)
                    if len(raw) != FRAME_BYTES:
                        break

                    arr = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                    imgs.append(Image.fromarray(arr, "RGB"))
                    idxs.append(frame_index)
                    frame_index += 1

                if not imgs:
                    break

                results = model.predict(imgs, conf=args.conf, imgsz=W, save=False, stream=False, rect=True)

                for idx, result in zip(idxs, results):
                    detections = boxes_to_json(result)
                    rec = {
                        "asset_id": asset_id,
                        "frame_index": idx,
                        "timestamp_sec": idx / fps,
                        "detections": detections,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    det_total += len(detections)

                dt = max(time.perf_counter() - t0, 1e-9)
                print(f"\rframes={written} fps={written/dt:.1f} dets={det_total}", end="", flush=True)

    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    elapsed = time.perf_counter() - t0

    manifest = {
        "asset_id": asset_id,
        "source_video": str(video),
        "model": args.model,
        "output": str(detections_path),
        "frames_written": written,
        "detections_written": det_total,
        "elapsed_sec": elapsed,
        "fps": written / elapsed if elapsed else None,
        "video_fps": fps,
        "conf": args.conf,
        "batch": args.batch,
        "sample_fps": args.sample_fps,
        "ffmpeg_stderr_tail": stderr[-4000:],
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
