#!/usr/bin/env python3

import os
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent.parent

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".m4v"
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def hash_worker(path):
    try:
        return sha256_file(path), path
    except Exception:
        return None

def find_videos(case_root):
    out = []
    for root, _, files in os.walk(case_root):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                out.append(str(Path(root) / f))
    return sorted(out)

MODEL = None
ARGS = None

def init_worker(model_path, out_root, batch, sample_fps):
    global MODEL, ARGS
    from yolo26mlx import YOLO

    MODEL = YOLO(model_path, task="segment")

    ARGS = {
        "out_root": out_root,
        "batch": batch,
        "sample_fps": sample_fps,
    }

def process_video(job):
    sha, video = job

    out_dir = Path(ARGS["out_root"]) / sha
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        return f"SKIP {sha}"

    cmd = [
        "python3",
        str(ROOT / "scripts" / "mlx_video_pipe_to_jsonl.py"),
        "--video", video,
        "--out-root", ARGS["out_root"],
        "--model", "models/yolo26n-seg.npz",
        "--batch", str(ARGS["batch"]),
        "--sample-fps", str(ARGS["sample_fps"]),
    ]

    t0 = time.time()

    p = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    dt = time.time() - t0

    return f"DONE {sha} sec={round(dt,1)}"

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--case-root", required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--model", default="models/yolo26n-seg.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hash-workers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sample-fps", type=float, default=5.0)

    args = ap.parse_args()

    print("DISCOVER")
    vids = find_videos(args.case_root)

    print(f"VIDEOS_FOUND {len(vids)}")

    print("HASHING")

    hashed = []

    with Pool(processes=args.hash_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(hash_worker, vids), 1):
            if result:
                hashed.append(result)

            if i % 100 == 0:
                print(f"HASHED {i}/{len(vids)}", flush=True)

    seen = set()
    unique = []

    for sha, path in hashed:
        if sha in seen:
            continue
        seen.add(sha)
        unique.append((sha, path))

    unique.sort()

    print(f"UNIQUE {len(unique)}")
    print(f"WORKERS {args.workers}")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            args.model,
            args.out_root,
            args.batch,
            args.sample_fps,
        ),
    ) as ex:

        futures = [ex.submit(process_video, x) for x in unique]

        done = 0

        for f in futures:
            try:
                r = f.result()
                done += 1
                print(f"[{done}/{len(unique)}] {r}", flush=True)
            except Exception as e:
                print("ERROR", e, flush=True)

if __name__ == "__main__":
    main()
