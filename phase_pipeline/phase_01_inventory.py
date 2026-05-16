#!/usr/bin/env python3
import argparse, hashlib, json, os, time
from pathlib import Path

VIDEO_EXT={".mp4",".mov",".m4v",".avi",".mts",".m2ts",".wmv"}
AUDIO_EXT={".wav",".mp3",".m4a",".aac",".flac",".ogg"}
PDF_EXT={".pdf"}
CHUNK=1024*1024

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(CHUNK),b""):
            h.update(b)
    return h.hexdigest()

def kind(p):
    e=p.suffix.lower()
    if e in VIDEO_EXT: return "video"
    if e in AUDIO_EXT: return "audio"
    if e in PDF_EXT: return "pdf"
    return "other"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--db", required=True)
    args=ap.parse_args()

    root=Path(args.root)
    db=Path(args.db)
    db.mkdir(parents=True, exist_ok=True)

    assets=db/"assets.jsonl"
    seen={}
    n=0
    unique=0

    with open(assets,"w",encoding="utf-8",buffering=1024*1024) as out:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if "/.DS_Store" in str(p):
                continue

            try:
                st=p.stat()
                sha=sha256_file(p)
            except Exception as e:
                print("SKIP",p,e)
                continue

            duplicate_of=seen.get(sha)
            if duplicate_of is None:
                seen[sha]=str(p)
                unique+=1

            rec={
                "asset_id":sha,
                "path":str(p),
                "kind":kind(p),
                "size":st.st_size,
                "mtime":st.st_mtime,
                "duplicate":duplicate_of is not None,
                "duplicate_of":duplicate_of,
            }
            out.write(json.dumps(rec,separators=(",",":"))+"\n")
            n+=1
            if n%1000==0:
                print("inventory",n,"unique",unique)

    print("DONE inventory")
    print("files",n)
    print("unique",unique)
    print("assets",assets)

if __name__=="__main__":
    main()
