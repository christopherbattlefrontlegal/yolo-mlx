#!/usr/bin/env python3
import argparse
import ast
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_names(sess):
    raw = sess.get_modelmeta().custom_metadata_map.get("names", "{}")
    return ast.literal_eval(raw)


def make_session(model):
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model, providers=providers)


def preprocess(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [
        float(x - w / 2),
        float(y - h / 2),
        float(x + w / 2),
        float(y + h / 2),
    ]


def detect(sess, names, inp, conf_floor, topk):
    output0, _ = sess.run(None, {"images": inp})
    preds = output0[0].T

    scores = preds[:, 4:146]
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]

    keep = confs >= conf_floor
    keep &= np.isin(cls_ids, list(names.keys()))

    if not np.any(keep):
        return []

    rows = preds[keep]
    cls_ids = cls_ids[keep]
    confs = confs[keep]

    order = np.argsort(-confs)
    if topk > 0:
        order = order[:topk]

    out = []
    for idx in order:
        cls_id = int(cls_ids[idx])
        out.append({
            "label": names.get(cls_id, str(cls_id)),
            "confidence": round(float(confs[idx]), 4),
            "bbox": xywh_to_xyxy(rows[idx, :4]),
        })
    return out


def process_asset(sess, names, frame_dir, out_dir, conf, topk, flush_every):
    asset_id = frame_dir.name
    out_path = out_dir / f"{asset_id}.detections.jsonl"
    done_path = out_dir / f"{asset_id}.done"

    if done_path.exists():
        print(f"SKIP {asset_id}")
        return

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"EMPTY {asset_id}")
        return

    print(f"START {asset_id} frames={len(frames)}")

    t0 = time.time()
    written = 0
    buffer = []

    with open(out_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
        for i, frame in enumerate(frames):
            inp = preprocess(frame)
            if inp is None:
                continue

            detections = detect(sess, names, inp, conf, topk)

            rec = {
                "asset_id": asset_id,
                "frame_index": i,
                "frame_path": str(frame),
                "detections": detections,
            }

            buffer.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))

            if len(buffer) >= flush_every:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

            written += 1

        if buffer:
            f.write("\n".join(buffer) + "\n")

    done_path.write_text("done\n", encoding="utf-8")

    elapsed = max(time.time() - t0, 1e-9)
    print(f"DONE {asset_id} frames={written} fps={written/elapsed:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default="/Volumes/VAULT/_machine_/decode_stage")
    ap.add_argument("--out-root", default="/Volumes/VAULT/_machine_/detections_stage")
    ap.add_argument("--model", default="/Volumes/VAULT/_machine_/yolo11_coreml/yoloe-11l-seg-forensic.onnx")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--flush-every", type=int, default=256)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = make_session(args.model)
    names = load_names(sess)

    print("providers:", sess.get_providers())

    for frame_dir in sorted(frames_root.iterdir()):
        if frame_dir.is_dir():
            process_asset(
                sess=sess,
                names=names,
                frame_dir=frame_dir,
                out_dir=out_dir,
                conf=args.conf,
                topk=args.topk,
                flush_every=args.flush_every,
            )


if __name__ == "__main__":
    main()
