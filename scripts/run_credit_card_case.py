#!/usr/bin/env python3
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CASE_ROOT = Path("/Volumes/Samsung_1tb/Perez_Cases_combined/COMBINED_DA_Production/25AGJ201B/C-25-395076-2_credit_card")
DECODE_ROOT = Path("/Volumes/VAULT/_machine_/decode_stage")
DETECT_ROOT = Path("/Volumes/VAULT/_machine_/mlx_detections_stage")
MODEL = "models/yolo26n-seg.npz"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def videos():
    out = []
    for p in CASE_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def decode_video(path: Path):
    sha = sha256_file(path)
    out = DECODE_ROOT / sha
    done = out / ".decode_done"
    err = out / ".decode_error"

    if done.exists():
        print(f"DECODE SKIP {sha} {path}", flush=True)
        return sha, out

    out.mkdir(parents=True, exist_ok=True)
    (out / "source_path.txt").write_text(str(path) + "\n", encoding="utf-8")

    print(f"DECODE START {sha} {path}", flush=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path),
        "-q:v", "3",
        str(out / "frame_%08d.jpg"),
    ]

    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        err.write_text((r.stderr or "") + "\n", encoding="utf-8")
        print(f"DECODE ERROR {sha} {path}", flush=True)
    else:
        done.write_text("done\n", encoding="utf-8")
        print(f"DECODE DONE {sha}", flush=True)

    return sha, out


def detect_folder(frame_dir: Path):
    sha = frame_dir.name
    manifest = DETECT_ROOT / sha / "manifest.json"

    if manifest.exists():
        print(f"DETECT SKIP {sha}", flush=True)
        return sha

    if not (frame_dir / ".decode_done").exists():
        print(f"DETECT SKIP no decode_done {sha}", flush=True)
        return sha

    print(f"DETECT START {sha}", flush=True)

    log_dir = Path("/tmp/yolo_mlx_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sha}.log"

    cmd = [
        "python3", "scripts/mlx_frames_to_jsonl.py",
        "--frames-dir", str(frame_dir),
        "--out-root", str(DETECT_ROOT),
        "--model", MODEL,
        "--conf", "0.55",
        "--chunk", "250",
    ]

    with log_path.open("w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

    if r.returncode == 0:
        print(f"DETECT DONE {sha}", flush=True)
    else:
        print(f"DETECT ERROR {sha} see {log_path}", flush=True)

    return sha


def main():
    DECODE_ROOT.mkdir(parents=True, exist_ok=True)
    DETECT_ROOT.mkdir(parents=True, exist_ok=True)

    seen = set()
    decoded = []

    print("STEP 0/1: hash unique videos and decode frames", flush=True)
    for v in videos():
        sha = sha256_file(v)
        if sha in seen:
            print(f"DUP SKIP {sha} {v}", flush=True)
            continue
        seen.add(sha)
        _, frame_dir = decode_video(v)
        decoded.append(frame_dir)

    print("STEP 2: detect decoded frame folders with 2 workers", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(detect_folder, d) for d in decoded]
        for fut in as_completed(futs):
            fut.result()

    print("DONE", flush=True)
    print(f"decode root: {DECODE_ROOT}")
    print(f"detect root: {DETECT_ROOT}")
    print("logs: /tmp/yolo_mlx_logs")


if __name__ == "__main__":
    main()
