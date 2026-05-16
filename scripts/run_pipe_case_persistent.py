#!/usr/bin/env python3
"""Orchestrator that spawns N persistent MLX workers. Each worker
imports MLX and loads the model once, then processes videos from a
shared queue via stdin/stdout. Eliminates per-video Python+MLX
cold-start.
"""
import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
CHUNK = 1024 * 1024
LOG_PATH = Path("/tmp/perez_pipeline_events.jsonl")
WORKER_SCRIPT = "scripts/mlx_worker_persistent.py"


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


def worker_loop(worker_id: int, proc: subprocess.Popen, work_q: queue.Queue):
    ready_line = proc.stdout.readline()
    if not ready_line:
        print(f"WORKER {worker_id} failed to start", flush=True)
        return
    try:
        ready = json.loads(ready_line)
    except Exception:
        print(f"WORKER {worker_id} bad ready line: {ready_line!r}", flush=True)
        return
    if ready.get("status") != "ready":
        print(f"WORKER {worker_id} bad ready: {ready}", flush=True)
        return

    print(f"WORKER {worker_id} READY", flush=True)
    log_event("worker_ready", worker=worker_id)

    while True:
        item = work_q.get()
        try:
            if item is None:
                try:
                    proc.stdin.write("__SHUTDOWN__\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                break

            sha, video = item
            t0 = time.time()
            log_event("start", worker=worker_id, sha=sha, video=str(video))

            try:
                proc.stdin.write(str(video) + "\n")
                proc.stdin.flush()
            except Exception as e:
                log_event("error", worker=worker_id, sha=sha,
                          video=str(video), error=f"stdin write failed: {e!r}")
                break

            result_line = proc.stdout.readline()
            if not result_line:
                log_event("error", worker=worker_id, sha=sha,
                          video=str(video), error="worker died (empty stdout)")
                break

            try:
                result = json.loads(result_line)
            except Exception as e:
                log_event("error", worker=worker_id, sha=sha,
                          video=str(video), error=f"bad json: {e!r}",
                          raw=result_line[:500])
                continue

            dt = time.time() - t0
            log_event(
                result.get("status", "done"),
                worker=worker_id,
                sha=sha,
                video=str(video),
                elapsed_sec=dt,
                frames=result.get("frames"),
                detections=result.get("detections"),
                fps=result.get("fps"),
            )
            print(
                f"{result.get('status', 'done').upper():5s} "
                f"w={worker_id} frames={result.get('frames')} "
                f"fps={result.get('fps')} {video}",
                flush=True,
            )
        finally:
            work_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", default="models/yolo26n-seg.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sample-fps", type=float, default=15.0)
    ap.add_argument("--hash-workers", type=int, default=24)
    ap.add_argument("--conf", type=float, default=0.55)
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

    with ThreadPoolExecutor(max_workers=args.hash_workers) as hex_ex:
        hashed = list(hex_ex.map(lambda v: (sha256_file(v), v), vids))

    seen = set()
    unique = []
    for sha, v in sorted(hashed, key=lambda x: (x[0], str(x[1]))):
        if sha in seen:
            print(f"DUP {sha} {v}", flush=True)
            continue
        seen.add(sha)
        unique.append((sha, v))

    print(
        f"VIDEOS total={len(vids)} unique={len(unique)} workers={args.workers} "
        f"batch={args.batch} sample_fps={args.sample_fps}",
        flush=True,
    )

    work_q: queue.Queue = queue.Queue()
    for item in unique:
        work_q.put(item)
    for _ in range(args.workers):
        work_q.put(None)

    procs = []
    threads = []
    for i in range(args.workers):
        cmd = [
            "python3",
            WORKER_SCRIPT,
            "--out-root", str(out_root),
            "--model", args.model,
            "--conf", str(args.conf),
            "--batch", str(args.batch),
            "--sample-fps", str(args.sample_fps),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(target=worker_loop, args=(i, proc, work_q), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for proc in procs:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()

    log_event("orchestrator_done", workers=args.workers, videos=len(unique))


if __name__ == "__main__":
    main()
