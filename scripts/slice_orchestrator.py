#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass

import cv2

from yolo26mlx import YOLO

CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Config:
    video: str
    model: str
    tracker: str
    out_root: str
    conf: float
    imgsz: int
    max_frames: int


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps


def safe_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def build_record(asset_id: str, frame_index: int, fps: float, result) -> dict:
    rec = {
        "asset_id": asset_id,
        "frame_index": frame_index,
        "timestamp_sec": frame_index / fps,
        "detections": [],
    }

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return rec

    xyxy = safe_list(getattr(boxes, "xyxy", []))
    conf = safe_list(getattr(boxes, "conf", []))
    cls = safe_list(getattr(boxes, "cls", []))
    ids_raw = getattr(boxes, "id", None)
    ids = safe_list(ids_raw) if ids_raw is not None else [None] * len(xyxy)

    names = getattr(result, "names", {}) or {}

    for i, box in enumerate(xyxy):
        class_id = int(cls[i]) if i < len(cls) else None
        track_id = ids[i] if i < len(ids) else None

        rec["detections"].append({
            "det_id": f"{frame_index}:{i}",
            "class_id": class_id,
            "class_name": names.get(class_id) if isinstance(names, dict) else None,
            "confidence": float(conf[i]) if i < len(conf) else None,
            "bbox_xyxy": [float(v) for v in box],
            "track_id": int(track_id) if track_id is not None else None,
        })

    return rec


def move_preview(video: str, out_root: str, out_dir: str) -> str | None:
    base = os.path.splitext(os.path.basename(video))[0]
    candidates = [
        os.path.join(out_root, f"{base}_tracked.mp4"),
        os.path.join("results", f"{base}_tracked.mp4"),
        f"{base}_tracked.mp4",
    ]

    dst = os.path.join(out_dir, "preview_tracked.mp4")

    for src in candidates:
        if os.path.isfile(src):
            shutil.move(src, dst)
            return dst

    return None


def run(cfg: Config) -> int:
    if not os.path.isfile(cfg.video):
        print(f"missing video: {cfg.video}", file=sys.stderr)
        return 1

    if not os.path.isfile(cfg.model):
        print(f"missing model: {cfg.model}", file=sys.stderr)
        return 1

    os.makedirs(cfg.out_root, exist_ok=True)

    print(f"hashing: {cfg.video}")
    asset_id = sha256_file(cfg.video)

    out_dir = os.path.join(cfg.out_root, asset_id)
    os.makedirs(out_dir, exist_ok=True)

    frames_path = os.path.join(out_dir, "frames.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    checkpoint_path = os.path.join(out_dir, ".checkpoint")

    fps = video_fps(cfg.video)
    model = YOLO(cfg.model)

    written = 0
    last_frame_index = -1
    error = None
    t0 = time.time()

    try:
        with open(frames_path, "w", encoding="utf-8", buffering=1) as f:
            stream = model.track(
                cfg.video,
                tracker=cfg.tracker,
                conf=cfg.conf,
                imgsz=cfg.imgsz,
                stream=True,
                save=True,
                vid_stride=1,
            )

            for frame_index, result in enumerate(stream):
                rec = build_record(asset_id, frame_index, fps, result)
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

                written += 1
                last_frame_index = frame_index

                with open(checkpoint_path, "w", encoding="utf-8") as cp:
                    cp.write(str(frame_index))

                if written % 30 == 0:
                    elapsed = max(time.time() - t0, 1e-9)
                    print(
                        f"frame {written}/{cfg.max_frames} | "
                        f"{written / elapsed:.2f} fps | "
                        f"dets={len(rec['detections'])}"
                    )

                if written >= cfg.max_frames:
                    break

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    preview_path = move_preview(cfg.video, cfg.out_root, out_dir)

    elapsed = time.time() - t0
    manifest = {
        "asset_id": asset_id,
        "video_path": cfg.video,
        "model_path": cfg.model,
        "tracker": cfg.tracker,
        "conf": cfg.conf,
        "imgsz": cfg.imgsz,
        "max_frames": cfg.max_frames,
        "frames_written": written,
        "last_frame_index": last_frame_index,
        "video_fps": fps,
        "elapsed_sec": elapsed,
        "throughput_fps": written / elapsed if elapsed > 0 else None,
        "error": error,
        "outputs": {
            "frames_jsonl": frames_path,
            "preview_mp4": preview_path,
        },
    }

    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)

    print(f"\ndone: {written} frames | {written / max(elapsed, 1e-9):.2f} fps")
    print(f"jsonl:    {frames_path}")
    print(f"preview:  {preview_path or '(missing)'}")
    print(f"manifest: {manifest_path}")

    return 0 if error is None else 2


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="video.mp4")
    p.add_argument("--model", default="models/yolo26n.npz")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--out-root", default="results")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=300)
    a = p.parse_args()

    return Config(
        video=a.video,
        model=a.model,
        tracker=a.tracker,
        out_root=a.out_root,
        conf=a.conf,
        imgsz=a.imgsz,
        max_frames=a.max_frames,
    )


if __name__ == "__main__":
    sys.exit(run(parse_args()))