"""
LiDAR render server (Process A of the Phase-2 "LiDAR Splat in Isaac Sim" bridge).

Analogous to splat_render_server.py, but renders synthetic LiDAR scans instead
of RGB images.  Uses the trained LiDAR-3DGS model (best.pt) with a 6-face
cubemap to cover the Livox Mid-360's 360° FOV.

Protocol (file-based IPC, same pattern as Camera bridge):
  Request  <queue_dir>/request_<N>.json:
      {
          "R_wl": [[3x3 row-major]],   # world → LiDAR sensor rotation
          "t_wl": [x, y, z],           # LiDAR sensor origin in world (metres)
          "use_raydrop": true,          # optional, default true
          "objects": [{"name": "opponent", "T_wo": [[4x4]]}]   # optional object splats
      }                                    # (T_wo object -> world; see object_splats.py)

  Response <queue_dir>/response_<N>.npz:
      {"xyz": Nx3 float32,             # point cloud, LiDAR sensor frame
       "intensity": N float32}         # 0-1 intensity from the model

  When the response file appears the request file is deleted (done signal).

Run this FIRST, in the nerfstudio/gsplat-capable conda env:
    conda run -n nerfstudio-py310 python3 scripts/lidar_render_server.py \\
        --queue-dir isaac_sim_output/lidar_queue

Then run isaac_sim_lidar_splat_demo.py in Isaac Sim's own Python (separate process).

Why separate process: same reason as Camera bridge -- Isaac Sim's bundled
torch/CUDA version conflicts with nerfstudio/gsplat's requirements.
"""
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)  # logs stay readable when redirected to a file
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", default="isaac_sim_output/lidar_queue")
    parser.add_argument("--checkpoint", default=None,
                        help="override path to best.pt directly")
    parser.add_argument("--no-raydrop", action="store_true",
                        help="disable raydrop masking (keep all rendered points)")
    parser.add_argument("--hybrid", action="store_true",
                        help="MoCap+LiDAR hybrid mode (2026-07-03): return the real "
                             "captured scan for known replay frames (exact "
                             "camera_frame_idx match) or a close-enough real neighbor, "
                             "falling back to the synthetic 3DGS model only for "
                             "genuinely novel poses. See lidar_renderer_hybrid.py.")
    parser.add_argument("--alpha-thresh", type=float, default=None,
                        help="drop rays with accumulated opacity below this (e.g. 0.99); "
                             "use with --no-raydrop for checkpoints trained without a raydrop head")
    parser.add_argument("--object", action="append", default=[], metavar="NAME=PATH.npz",
                        help="object splat available to requests by name (scripts/extract_object_splat.py)")
    parser.add_argument("--livox-bag", default=None,
                        help="ROS2 bag with /livox/lidar scans: sample each scan along recorded Livox return directions "
                             "(cycling through the recorded patterns) instead of the dense 6x256x256 cubemap")
    parser.add_argument("--object-lidar-stride", type=int, default=4,
                        help="LiDAR layer uses every Nth object gaussian with a sqrt(N) larger footprint (1 = full asset)")
    parser.add_argument("--poll-interval", type=float, default=0.005)
    args = parser.parse_args()
    from object_splats import ObjectRegistry
    objects = ObjectRegistry(args.object)

    if args.hybrid:
        from lidar_renderer_hybrid import HybridLidarRenderer
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        renderer = HybridLidarRenderer(checkpoint_path=ckpt)
    else:
        from lidar_renderer import LidarRenderer, CHECKPOINT_PATH
        ckpt = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
        renderer = LidarRenderer(checkpoint_path=ckpt)
        renderer.object_lidar_stride = args.object_lidar_stride
    use_raydrop = not args.no_raydrop
    patterns = None
    if args.livox_bag and not args.hybrid:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from render_learned_demo import load_livox_patterns
        patterns = [d for _, d in load_livox_patterns(args.livox_bag, seconds=6.0)]
        print(f"[lidar_render_server] Livox sampling: {len(patterns)} recorded scan patterns, "
              f"{int(np.mean([len(d) for d in patterns]))} rays each")

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lidar_render_server] watching {queue_dir} ... (Ctrl+C to stop)")
    print(f"[lidar_render_server] raydrop={'on' if use_raydrop else 'off'}")

    n_served = 0
    stop_flag = queue_dir / "STOP"

    while True:
        if stop_flag.exists():
            print("[lidar_render_server] STOP file seen, exiting")
            stop_flag.unlink()
            break

        requests = sorted(queue_dir.glob("request_*.json"))
        if not requests:
            time.sleep(args.poll_interval)
            continue

        req_path = requests[0]
        idx_str = req_path.stem.split("_", 1)[1]
        resp_path = queue_dir / f"response_{idx_str}.npz"
        # np.savez_compressed appends ".npz" if the name doesn't already end
        # with it, so the temp name must itself end in ".npz" or the rename
        # below will look for the wrong file (e.g. "....npz.tmp.npz").
        tmp_path = queue_dir / f"response_{idx_str}.tmp.npz"

        try:
            req = json.load(open(req_path))
            R_wl = np.array(req["R_wl"], dtype=np.float64)
            t_wl = np.array(req["t_wl"], dtype=np.float64)
            raydrop = req.get("use_raydrop", use_raydrop)

            objs = objects.resolve(req.get("objects"))
            if args.hybrid:
                if objs:
                    print(f"[lidar_render_server] WARNING objects ignored in --hybrid mode ({req_path.name})")
                xyz, intensity = renderer.render(
                    R_wl, t_wl, use_raydrop=raydrop,
                    camera_frame_idx=req.get("camera_frame_idx"))
            elif patterns is not None:
                dirs = patterns[n_served % len(patterns)]
                xyz, intensity = renderer.render_along_rays(R_wl, t_wl, dirs, use_raydrop=raydrop,
                                                            alpha_thresh=args.alpha_thresh, objects=objs)
            else:
                xyz, intensity = renderer.render(R_wl, t_wl, use_raydrop=raydrop,
                                                 alpha_thresh=args.alpha_thresh, objects=objs)
            # uncompressed: compressing 130k points cost ~100 ms per scan, the file is 2 MB either way for the client
            np.savez(tmp_path, xyz=xyz.astype(np.float32), intensity=intensity.astype(np.float32))
            tmp_path.rename(resp_path)
            n_served += 1
            if n_served % 20 == 0 or n_served <= 3:
                print(f"[lidar_render_server] served {n_served} scans "
                      f"(last: {req_path.name}, {len(xyz):,} pts)")
        except Exception as e:
            print(f"[lidar_render_server] ERROR on {req_path.name}: {e}")
            # write empty scan so the consumer doesn't hang
            np.savez(tmp_path, xyz=np.zeros((0, 3), dtype=np.float32),
                     intensity=np.zeros(0, dtype=np.float32))
            tmp_path.rename(resp_path)
        finally:
            req_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
