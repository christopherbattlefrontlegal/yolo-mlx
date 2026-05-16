#!/usr/bin/env python3
"""
dedup_media.command

Photographs + videos only. Nothing else is hashed, nothing else is touched.

Pipeline:
  1. Recursively walk <input_dir> for image/video files (case-insensitive).
     Skip symlinks and hidden files.
  2. SHA-256 every file in parallel.
  3. Group by hash. For each hash pick ONE path (oldest mtime, then shortest path).
  4. Write <output_dir>/_unique_media_<ts>.txt and <output_dir>/_media_hashes_<ts>.txt.

Usage:
  ./dedup_then_ocr\\ copy\\ 2.command <input_dir> <output_dir>
  ./dedup_then_ocr\\ copy\\ 2.command         (prompts interactively)
"""

import hashlib
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".webp", ".gif", ".dng", ".cr2",
    ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm",
    ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".3gp",
}

CHUNK = 1 << 20  # 1 MiB read buffer
WORKERS = int(os.environ.get("JOBS", cpu_count()))


def clean(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return os.path.expanduser(s)


def prompt(label: str) -> str:
    return clean(input(f"{label}: "))


def media_kind(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def iter_media(root: str):
    if os.path.isfile(root) and not os.path.islink(root):
        if media_kind(root):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            if media_kind(name) is None:
                continue
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and not os.path.islink(p):
                yield p


def sha256_with_size(path: str):
    try:
        h = hashlib.sha256()
        size = 0
        with open(path, "rb", buffering=0) as f:
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                h.update(buf)
                size += len(buf)
        return (path, h.hexdigest(), size, None)
    except OSError as e:
        return (path, None, 0, f"{type(e).__name__}: {e}")


def pick_keep(paths: list[str]) -> str:
    ranked = []
    for p in paths:
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = float("inf")
        ranked.append((mt, len(p), p))
    ranked.sort()
    return ranked[0][2]


def main() -> int:
    if len(sys.argv) >= 3:
        src = clean(sys.argv[1])
        out_dir = clean(sys.argv[2])
    else:
        src = prompt("Source folder to scan")
        out_dir = prompt("Output folder")

    if not os.path.exists(src):
        print(f"Error: input path does not exist: {src}", file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")

    # ---- Step 1: enumerate image + video files ----
    print(f"\n[1/3] Walking for image + video under: {src}")
    files = list(iter_media(src))
    total = len(files)
    print(f"      Found {total:,} media files")
    if total == 0:
        print("Nothing to do.")
        return 0

    # ---- Step 2: SHA-256 ----
    print(f"\n[2/3] SHA-256 with {WORKERS} workers")
    hashes_file = os.path.join(out_dir, f"_media_hashes_{stamp}.txt")
    by_hash: dict[str, list[str]] = defaultdict(list)
    errors: list[tuple[str, str]] = []
    bytes_done = 0
    count = 0
    start = time.time()
    last_report = start

    with open(hashes_file, "w", encoding="utf-8", errors="replace",
              buffering=1 << 16) as fout, Pool(processes=WORKERS) as pool:
        for path, hexd, size, err in pool.imap_unordered(
            sha256_with_size, files, chunksize=1
        ):
            if err:
                errors.append((path, err))
                fout.write(f"# ERROR {err}\t{path}\n")
            else:
                by_hash[hexd].append(path)
                fout.write(f"{hexd}  {size}  {path}\n")
                bytes_done += size
            count += 1
            now = time.time()
            if now - last_report >= 0.5 or count == total:
                elapsed = max(now - start, 1e-9)
                rate = count / elapsed
                mbps = (bytes_done / (1024 * 1024)) / elapsed
                pct = 100 * count / total
                eta = (total - count) / rate if rate > 0 else 0
                sys.stderr.write(
                    f"\r{count:>6,}/{total:,} ({pct:5.1f}%) | "
                    f"{rate:>5,.0f} file/s | {mbps:>6,.1f} MiB/s | "
                    f"err {len(errors)} | ETA {eta:>5.0f}s   "
                )
                sys.stderr.flush()
                last_report = now

    sys.stderr.write("\n")
    elapsed = time.time() - start
    print(f"      Hashed {total:,} files in {elapsed:.1f}s "
          f"({(bytes_done/(1024**3))/max(elapsed,1e-9):.2f} GiB/s)")
    print(f"      Hash report: {hashes_file}")

    # ---- Step 3: pick one path per unique SHA-256 ----
    unique: list[tuple[str, str, str]] = []
    for h, paths in by_hash.items():
        keep = pick_keep(paths)
        unique.append((h, keep, media_kind(keep)))
    unique.sort(key=lambda x: x[1])
    dupes = total - len(unique)
    n_image = sum(1 for _, _, k in unique if k == "image")
    n_video = sum(1 for _, _, k in unique if k == "video")
    list_file = os.path.join(out_dir, f"_unique_media_{stamp}.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("# unique media files by SHA-256 (one KEEP per content hash)\n")
        f.write(f"# source     : {src}\n")
        f.write(f"# total      : {total}\n")
        f.write(f"# unique     : {len(unique)}\n")
        f.write(f"# images     : {n_image}\n")
        f.write(f"# videos     : {n_video}\n")
        f.write(f"# duplicates : {dupes}\n")
        f.write(f"# errors     : {len(errors)}\n\n")
        for h, p, kind in unique:
            f.write(f"{h}\t{kind}\t{p}\n")
    print(f"\n[3/3] Unique files: {len(unique):,}  "
          f"(images {n_image:,}, videos {n_video:,}, "
          f"removed {dupes:,} content-duplicates)")
    print(f"      List file  : {list_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
