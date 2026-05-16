#!/usr/bin/env python3
import argparse
import hashlib
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
CHUNK = 1024 * 1024
LOG_PATH = Path("/tmp/perez_pipeline_events.jsonl")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()

def log_event(event: str, **kw):
    rec = {"ts": time.time(), "event": event, **kw}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def run_one(video: Path, out_root: Path, model: str, batch: int, sample_fps: float):
    sha = sha256_file(video)
    manifest = out_root / sha / "manifest.json"
    if manifest.exists():
        print(f"SKIP {sha} {video}", flush=True)
        log_event("skip", sha=sha, video=str(video))
        return

    print(f"START {sha} {video}", flush=True)
    t0 = time.time()
    log_event("start", sha=sha, video=str(video))
    cmd = [
        "python3", "scripts/mlx_video_pipe_to_jsonl.py",
        "--video", str(video),
        "--out-root", str(out_root),
        "--model", model,
        "--conf", "0.55",
        "--batch", str(batch),
        "--sample-fps", str(sample_fps),
    ]
    r = subprocess.run(cmd)
    dt = time.time() - t0
    manifest = out_root / sha / "manifest.json"
    frames = None
    dets = None
    fps = None
    if manifest.exists():
        try:
            j = json.loads(manifest.read_text())
            frames = j.get("frames_written")
            dets = j.get("detections_written")
            fps = j.get("fps")
        except Exception:
            pass
    status = "done" if r.returncode == 0 else "error"
    log_event(status, sha=sha, video=str(video), returncode=r.returncode, elapsed_sec=dt, frames=frames, detections=dets, fps=fps)
    print(("DONE " if r.returncode == 0 else "ERROR ") + f"{sha} frames={frames} fps={fps}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--out-root", default="/Volumes/VAULT/_machine_/mlx_pipe_detections_stage")
    ap.add_argument("--model", default="models/yolo26n-seg.npz")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sample-fps", type=float, default=15.0)
    ap.add_argument("--hash-workers", type=int, default=24)
    args = ap.parse_args()

    LOG_PATH.write_text("", encoding="utf-8")
    case_root = Path(args.case_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vids = sorted(
        [p for p in case_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: str(p),
    )

    print(f"HASHING total={len(vids)} hash_workers={args.hash_workers}", flush=True)

    with ThreadPoolExecutor(max_workers=args.hash_workers) as hex:
        hashed = list(hex.map(lambda v: (sha256_file(v), v), vids))

    seen = set()
    unique = []
    for sha, v in sorted(hashed, key=lambda x: (x[0], str(x[1]))):
        if sha in seen:
            print(f"DUP {sha} {v}", flush=True)
            continue
        seen.add(sha)
        unique.append(v)

    print(f"VIDEOS total={len(vids)} unique={len(unique)} workers={args.workers} batch={args.batch} sample_fps={args.sample_fps}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, v, out_root, args.model, args.batch, args.sample_fps) for v in unique]
        for fut in as_completed(futs):
            fut.result()

if __name__ == "__main__":
    main()
