#!/usr/bin/env python3
"""Re-encode all mp4 files in a directory tree from mpeg4 (mp4v) to h264.

Skips files that are already h264. Encodes in-place (overwrites original).

Usage:
    python scripts/reencode_videos.py --src CONVERTED
    python scripts/reencode_videos.py --src CONVERTED --workers 8
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_codec(path: str) -> str:
    """Get video codec name via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def reencode_one(path: str) -> str:
    """Re-encode a single file to h264 if needed. Returns status message."""
    codec = get_codec(path)
    if codec == "h264":
        return f"SKIP (already h264): {path}"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(path))
    os.close(tmp_fd)

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            os.unlink(tmp_path)
            return f"FAIL: {path} ({result.stderr[-200:]})"

        os.replace(tmp_path, path)
        return f"OK: {path}"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return f"ERROR: {path} ({e})"


def main():
    parser = argparse.ArgumentParser(description="Re-encode mp4 files to h264")
    parser.add_argument("--src", default="CONVERTED", help="Root directory to scan")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    mp4s = sorted(glob.glob(os.path.join(args.src, "**", "*.mp4"), recursive=True))
    print(f"Found {len(mp4s)} mp4 files in {args.src}")

    done = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reencode_one, p): p for p in mp4s}
        for future in as_completed(futures):
            msg = future.result()
            done += 1
            if msg.startswith("OK"):
                pass
            elif msg.startswith("SKIP"):
                skipped += 1
            else:
                failed += 1
                print(msg)

            if done % 100 == 0 or done == len(mp4s):
                print(f"  Progress: {done}/{len(mp4s)} "
                      f"(skipped={skipped}, failed={failed})")

    print(f"\nDone. {done - skipped - failed} re-encoded, "
          f"{skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
