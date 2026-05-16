#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CHUNK = 1024 * 1024


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def load_names(session):
    raw = session.get_modelmeta().custom_metadata_map.get("names", "{}")
    return ast.literal_eval(raw)


def make_session(model_path):
    providers = [
        ("CoreMLExecutionProvider", {
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
            "ModelCacheDirectory": "/Volumes/VAULT/_machine_/ort_coreml_cache",
        }),
        "CPUExecutionProvider",
    ]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=providers,
    )

    print("active providers:", sess.get_providers())
    return sess


def preprocess(frame, imgsz):
    img = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img)


def xywh_to_xyxy_np(boxes):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    out = np.empty_like(boxes[:, :4], dtype=np.float32)
    out[:, 0] = x - w / 2
    out[:, 1] = y - h / 2
    out[:, 2] = x + w / 2
    out[:, 3] = y + h / 2
    return out


def postprocess(output0, names, conf_floor, topk):
    preds = output0[0].T

    scores = preds[:, 4:146]
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]

    valid_classes = np.isin(cls_ids, list(names.keys()))
    keep = (confs >= conf_floor) & valid_classes

    if not np.any(keep):
        return []

    boxes = preds[keep, :4]
    cls_ids = cls_ids[keep]
    confs = confs[keep]

    order = np.argsort(-confs)
    if topk > 0:
        order = order[:topk]

    boxes_xyxy = xywh_to_xyxy_np(boxes[order])
    cls_ids = cls_ids[order]
    confs = confs[order]

    detections = []
    for box, cls_id, conf in zip(boxes_xyxy, cls_ids, confs):
        ci = int(cls_id)
        detections.append({
            "label": names.get(ci, str(ci)),
            "confidence": round(float(conf), 4),
            "bbox": [float(v) for v in box],
        })

    return detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    video = Path(args.video)
    model = Path(args.model)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    asset_id = sha256_file(video)
    asset_dir = out_root / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    detections_path = asset_dir / "detections.jsonl"
    manifest_path = asset_dir / "manifest.json"
    source_path = asset_dir / "source_path.txt"
    source_path.write_text(str(video) + "\n", encoding="utf-8")

    sess = make_session(str(model))
    names = load_names(sess)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    t0 = time.perf_counter()
    decode_t = prep_t = infer_t = post_t = write_t = 0.0
    frames = 0
    det_total = 0

    with open(detections_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
        while True:
            if args.max_frames > 0 and frames >= args.max_frames:
                break

            t = time.perf_counter()
            ok, frame = cap.read()
            decode_t += time.perf_counter() - t

            if not ok:
                break

            if args.stride > 1 and frames % args.stride != 0:
                frames += 1
                continue

            t = time.perf_counter()
            inp = preprocess(frame, args.imgsz)
            prep_t += time.perf_counter() - t

            t = time.perf_counter()
            output0, output1 = sess.run(None, {"images": inp})
            infer_t += time.perf_counter() - t

            t = time.perf_counter()
            detections = postprocess(output0, names, args.conf, args.topk)
            post_t += time.perf_counter() - t

            rec = {
                "asset_id": asset_id,
                "frame_index": frames,
                "timestamp_sec": frames / fps,
                "detections": detections,
            }

            t = time.perf_counter()
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            write_t += time.perf_counter() - t

            det_total += len(detections)
            frames += 1

            if args.progress_every > 0 and frames % args.progress_every == 0:
                elapsed = max(time.perf_counter() - t0, 1e-9)
                print(
                    f"frame={frames} fps={frames/elapsed:.2f} "
                    f"dets={det_total} infer_fps={frames/max(infer_t,1e-9):.2f}"
                )

    cap.release()

    elapsed = time.perf_counter() - t0
    manifest = {
        "asset_id": asset_id,
        "source_video": str(video),
        "model": str(model),
        "output": str(detections_path),
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        },
        "run": {
            "frames_written": frames,
            "detections_written": det_total,
            "elapsed_sec": elapsed,
            "total_fps": frames / elapsed if elapsed else None,
            "decode_sec": decode_t,
            "preprocess_sec": prep_t,
            "infer_sec": infer_t,
            "postprocess_sec": post_t,
            "write_sec": write_t,
            "infer_fps": frames / infer_t if infer_t else None,
            "providers": sess.get_providers(),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "topk": args.topk,
        },
    }

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)

    print("\nDONE")
    print("frames:", frames)
    print("total_fps:", round(frames / elapsed, 2) if elapsed else None)
    print("infer_fps:", round(frames / infer_t, 2) if infer_t else None)
    print("out:", detections_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
