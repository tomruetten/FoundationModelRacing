"""
Splat render server (Process B of the "Kamera Splat in Isaac Sim" bridge).

WHY a separate process: Isaac Sim ships its own bundled Python/torch/CUDA
stack tied to the Omniverse Kit runtime. Loading nerfstudio+gsplat (which
need their own torch/CUDA build) inside the SAME Python process as
`from isaacsim import SimulationApp` risks a dependency conflict that may
silently break either Isaac Sim or the renderer. Running them as two
separate processes that talk via the filesystem sidesteps this entirely --
simple, robust, no environment debugging required.

Protocol (simple, file-based):
  - Isaac Sim writes its current camera pose to <queue_dir>/request_<N>.json
    as {"c2w": [[...4x4 nested list...]]}. Optional "objects": [{"name" | "path",
    "T_wo": [[4x4]]}] inserts object splats (opponents, cones; see object_splats.py)
    posed by T_wo (object -> the c2w world frame) into the scene for this frame.
  - This server polls <queue_dir>, finds the lowest unprocessed request_<N>.json,
    renders the frame, writes <queue_dir>/response_<N>.<bmp|png|jpg> (--image-format,
    default bmp: lossless and 13x faster than png), then deletes the request file
    (signals "done" without needing a second channel).
  - Isaac Sim polls for response_<N>.* and reads it once present.

Run this FIRST, in the nerfstudio/gsplat-capable conda env (nerfstudio-py310
locally, or whatever the lab's "probe" env is called -- per TECHNICAL_NOTES.md the lab
env should have nerfstudio+gsplat installed the same way as this machine):

    conda run -n nerfstudio-py310 python3 splat_render_server.py \
        --queue-dir isaac_sim_output/splat_queue

Then run isaac_sim_camera_splat_demo.py separately (Isaac Sim's own Python).
"""
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)  # logs stay readable when redirected to a file
import os
import time
from pathlib import Path

import numpy as np


def _save_image(rgb, path, fmt):
    from PIL import Image
    if fmt == "jpg":
        Image.fromarray(rgb).save(path, format="JPEG", quality=95)
    elif fmt == "bmp":
        Image.fromarray(rgb).save(path, format="BMP")
    else:
        Image.fromarray(rgb).save(path, format="PNG", compress_level=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", default="isaac_sim_output/splat_queue")
    parser.add_argument("--checkpoint", choices=["cropped", "neartriage", "pure", "lidarinit"], default="cropped",
                         help="'cropped'=26.20dB BEST (1280x600, no car front, pure COLMAP 997 frames), "
                              "'neartriage'=25.05dB (1280x800), "
                              "'pure'=22.29dB, 'lidarinit'=19.71dB best LPIPS")
    parser.add_argument("--config", default=None,
                         help="override SplatCameraRenderer config.yml path directly "
                              "(takes precedence over --checkpoint)")
    parser.add_argument("--data-dir", default=None,
                        help="dataset dir holding transforms.json (only needed if the path stored in config.yml "
                             "does not exist on this machine and auto-detection fails)")
    parser.add_argument("--object", action="append", default=[], metavar="NAME=PATH.npz",
                        help="object splat available to requests by name (scripts/extract_object_splat.py); "
                             "requests may also give a 'path' directly")
    parser.add_argument("--image-format", choices=["bmp", "png", "jpg"], default="bmp",
                        help="response image format: bmp is lossless and 13x faster than png (12 vs 163 ms per 1280x800 "
                             "frame incl. the client's read); jpg q95 is as fast and 15x smaller but lossy")
    parser.add_argument("--object-soften", type=float, default=1.3,
                        help="footprint factor for inserted object gaussians (mesh-derived objects are sharper than the scene)")
    parser.add_argument("--poll-interval", type=float, default=0.005)
    args = parser.parse_args()

    from splat_camera_renderer import SplatCameraRenderer, CHECKPOINTS
    from object_splats import ObjectRegistry
    from PIL import Image
    objects = ObjectRegistry(args.object)

    config_path = Path(args.config) if args.config else CHECKPOINTS[args.checkpoint]
    renderer = SplatCameraRenderer(config_path=config_path,
                                   data_dir=Path(args.data_dir) if args.data_dir else None)
    renderer.object_soften = args.object_soften

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    print(f"[splat_render_server] watching {queue_dir} ... (Ctrl+C to stop)")

    n_served = 0
    stop_flag = queue_dir / "STOP"
    while True:
        if stop_flag.exists():
            print("[splat_render_server] STOP file seen, exiting")
            stop_flag.unlink()
            break

        requests = sorted(queue_dir.glob("request_*.json"))
        if not requests:
            time.sleep(args.poll_interval)
            continue

        req_path = requests[0]
        idx_str = req_path.stem.split("_", 1)[1]
        resp_path = queue_dir / f"response_{idx_str}.{args.image_format}"

        try:
            import json
            req = json.load(open(req_path))
            c2w = np.array(req["c2w"])
            already_in_model_frame = req.get("already_in_model_frame", False)
            rgb = renderer.render(c2w, already_in_model_frame=already_in_model_frame,
                                  objects=objects.resolve(req.get("objects")))
            # write to a temp name and rename: the consumer polls for
            # response_<N>.png and must never see a half-written file
            tmp_path = resp_path.with_name(resp_path.stem + ".tmp" + resp_path.suffix)
            _save_image(rgb, tmp_path, args.image_format)
            os.replace(tmp_path, resp_path)
            n_served += 1
            if n_served % 20 == 0:
                print(f"[splat_render_server] served {n_served} frames "
                      f"(last: {req_path.name})")
        except Exception as e:
            print(f"[splat_render_server] ERROR on {req_path.name}: {e}")
            # write a tiny black placeholder so the consumer doesn't hang forever
            tmp_path = resp_path.with_name(resp_path.stem + ".tmp" + resp_path.suffix)
            _save_image(np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8), tmp_path, args.image_format)
            os.replace(tmp_path, resp_path)
        finally:
            req_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
