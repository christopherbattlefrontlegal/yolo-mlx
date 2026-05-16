#!/usr/bin/env python3
import argparse
import json
import time

import cv2
import numpy as np
import onnxruntime as ort


def prep(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img)


def load_frames(video, n):
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(prep(frame))
    cap.release()
    return frames


def bench(name, providers, model, frames):
    sess = ort.InferenceSession(model, providers=providers)

    for x in frames[:10]:
        sess.run(None, {"images": x})

    t0 = time.perf_counter()
    for x in frames:
        sess.run(None, {"images": x})
    dt = time.perf_counter() - t0

    return {
        "name": name,
        "providers": sess.get_providers(),
        "frames": len(frames),
        "infer_sec": dt,
        "infer_fps": len(frames) / dt if dt else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames)

    variants = [
        ("coreml_default", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        ("coreml_all", [("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}), "CPUExecutionProvider"]),
        ("coreml_cpu_gpu", [("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndGPU"}), "CPUExecutionProvider"]),
        ("coreml_cpu_ane", [("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndNeuralEngine"}), "CPUExecutionProvider"]),
    ]

    results = []
    for name, providers in variants:
        print("\n===", name, "===")
        try:
            r = bench(name, providers, args.model, frames)
        except Exception as e:
            r = {"name": name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(json.dumps(r, indent=2))

    good = [r for r in results if r.get("infer_fps")]
    if good:
        best = max(good, key=lambda r: r["infer_fps"])
        print("\nBEST", best["name"], round(best["infer_fps"], 2), "fps")


if __name__ == "__main__":
    main()
