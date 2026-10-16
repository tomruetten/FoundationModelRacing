#!/usr/bin/env python3
"""Show the newest PNG in a directory in a window, refreshing as new frames
arrive. Used next to the Isaac Sim window to display the 3DGS camera render
the car "sees" (the Isaac viewport only shows the track mesh + car marker).

    conda activate roboracer
    python scripts/live_frame_viewer.py --watch-dir outputs/isaac_demo_<ts>/frames
Press q or Esc in the window to quit.
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--title", default="Camera 3DGS render (what the car sees)")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--idle-exit", type=float, default=0.0,
                    help="quit after this many seconds without a new frame (0 = never)")
    args = ap.parse_args()

    watch = Path(args.watch_dir)
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    last = None
    last_change = time.time()
    while True:
        files = sorted(watch.glob(args.pattern)) if watch.exists() else []
        if files and files[-1] != last:
            img = cv2.imread(str(files[-1]))
            if img is not None:
                if args.scale != 1.0:
                    img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
                cv2.putText(img, files[-1].name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow(args.title, img)
                last = files[-1]
                last_change = time.time()
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if args.idle_exit and files and time.time() - last_change > args.idle_exit:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
