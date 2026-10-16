#!/usr/bin/env python3
"""Time one simulation step: camera render, LiDAR render, dynamics step, and the file-queue
round trip the Isaac clients use.

    conda activate roboracer && source scripts/env.sh
    python scripts/benchmark_sim_step.py                 # in-process components + queue round trips
    python scripts/benchmark_sim_step.py --no-queue      # components only

Scene: the New1 maps from scripts/env.sh, ego at frame 600, the demo's six objects (two
F1TENTH cars 2.5 / 4 m ahead, four cones). Prints medians in ms and writes
outputs/benchmark_sim_step.json.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation" / "isaac_sim"))
sys.path.insert(0, str(ROOT / "scripts"))
import object_splats as osp  # noqa: E402


def median_ms(fn, n=15, warm=3, sync=True):
    import torch
    for _ in range(warm):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return round(1000 * float(np.median(ts)), 1)


def scene():
    N1, O = os.environ["NEW1_ROOT"], os.environ["OBJECTS_DIR"]
    plane = osp.FloorPlane.load(N1 + "/dataset/floor_plane.json")
    frames = json.load(open(N1 + "/dataset/transforms.json"))["frames"]
    cum = osp.path_arclength(frames)
    k = 600
    c2w = np.array(frames[k]["transform_matrix"])
    car, cone = osp.ObjectSplat.load(O + "/f1tenth.npz"), osp.ObjectSplat.load(O + "/cone_photoscan.npz")

    def ahead(g, lat=0.0):
        return osp.pose_from_camera_frame(np.array(frames[osp.frame_ahead(k, cum, g, None)]["transform_matrix"]), floor=plane, lateral=lat)
    art = car.articulated(wheel_angle=1.0, steer_angle=0.1, pitch=0.01, roll=0.01)
    objects = [(car, ahead(2.5), art), (car, ahead(4.0, -0.2), art)] + [(cone, ahead(g, l)) for g, l in ((1.4, -.5), (2.5, .55), (3.8, -.5), (5.0, .55))]
    T_ego = osp.pose_from_camera_frame(c2w, floor=plane)
    R_wl, t_wl = T_ego[:3, :3], T_ego[:3, 3] + T_ego[:3, :3] @ np.array([0.27, 0.0, 0.11])
    return c2w, objects, R_wl, t_wl, car


def queue_round_trip(server_cmd, queue, make_request, response_glob, n=15, ready_marker="watching"):
    """Start a render server, send n requests one after another, return the median wall time
    per request (write request -> response file readable), like the Isaac clients see it."""
    queue = Path(queue)
    if queue.exists():
        for f in queue.glob("*"):
            f.unlink()
    queue.mkdir(parents=True, exist_ok=True)
    log = open(queue.parent / (queue.name + ".log"), "w")
    proc = subprocess.Popen(server_cmd + ["--queue-dir", str(queue)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT / "simulation" / "isaac_sim")
    t0 = time.time()
    while ready_marker not in open(log.name).read():
        if proc.poll() is not None or time.time() - t0 > 600:
            raise RuntimeError(f"server failed: {open(log.name).read()[-800:]}")
        time.sleep(0.2)
    ts = []
    try:
        for i in range(n + 3):
            req = queue / f"request_{i:06d}.json"
            t1 = time.perf_counter()
            json.dump(make_request(), open(req, "w"))
            while True:
                hits = [p for p in queue.glob(response_glob.format(i=i)) if ".tmp" not in p.name]
                if hits:
                    break
                time.sleep(0.001)
            if hits[0].suffix == ".npz":
                np.load(hits[0])["xyz"]
            else:
                from PIL import Image
                np.asarray(Image.open(hits[0]).convert("RGB"))
            if i >= 3:
                ts.append(time.perf_counter() - t1)
            hits[0].unlink()
    finally:
        (queue / "STOP").touch()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return round(1000 * float(np.median(ts)), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-queue", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark_sim_step.json"))
    args = ap.parse_args()
    from splat_camera_renderer import SplatCameraRenderer
    from lidar_renderer import LidarRenderer
    from render_learned_demo import load_livox_patterns
    c2w, objects, R_wl, t_wl, car = scene()
    res = {}
    cam = SplatCameraRenderer(config_path=Path(os.environ["CAMERA_3DGS_CONFIG"]))
    res["camera scene only"] = median_ms(lambda: cam.render(c2w))
    res["camera + 6 objects"] = median_ms(lambda: cam.render(c2w, objects=objects))
    lidar = LidarRenderer(checkpoint_path=Path(os.environ["LIDAR_3DGS_CKPT"]))
    bag = Path(os.environ["RAW_SESSION_142014"]) / "all_data/all_data_0.db3"
    dirs = load_livox_patterns(bag, seconds=1.0)[3][1]
    res["lidar livox scene only"] = median_ms(lambda: lidar.render_along_rays(R_wl, t_wl, dirs, use_raydrop=False, alpha_thresh=.99))
    for stride in (1, 4):
        lidar.object_lidar_stride = stride
        res[f"lidar livox + 6 objects, object stride {stride}"] = median_ms(
            lambda: lidar.render_along_rays(R_wl, t_wl, dirs, use_raydrop=False, alpha_thresh=.99, objects=objects), n=8)
    lidar.object_lidar_stride = 4

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gru = load("gru", ROOT / "dynamics/deployed/15_gru_inference.py").GRUInference(str(ROOT / "dynamics/deployed/trained_models/gru_run30_best.pth"))
        res["dynamics GRU step"] = median_ms(lambda: gru.step(vx=1.0, vy=0.0, yr=0.1, ax=0.2, ay=0.1, speed_cmd=1.2, steering_cmd=0.05), n=200, warm=10, sync=False)
        try:
            ms = load("msnn", ROOT / "dynamics/deployed/13_inference_msnn.py").MSNNInference(str(ROOT / "dynamics/deployed/trained_models/msnn_exp9_phase2_backup"))
            res["dynamics MS-NN step"] = median_ms(lambda: ms.step(0.05, 1.0, 0.2, 0.1, 0.0), n=100, warm=30, sync=False)
        except Exception as e:  # noqa: BLE001
            res["dynamics MS-NN step"] = f"n/a ({str(e)[:60]})"
    total_in = res["camera + 6 objects"] + res["lidar livox + 6 objects, object stride 4"] + res["dynamics GRU step"]
    res["IN-PROCESS step: camera + livox lidar + GRU, 6 objects"] = round(total_in, 1)

    if not args.no_queue:
        del cam, lidar   # free the GPU for the servers
        import torch; torch.cuda.empty_cache()
        O = os.environ["OBJECTS_DIR"]
        obj_req = [{"path": O + "/f1tenth.npz", "T_wo": T.tolist(), "articulation": {"wheel_angle": 1.0}} for _, T, *_ in objects[:2]] + \
                  [{"path": O + "/cone_photoscan.npz", "T_wo": T.tolist()} for _, T in objects[2:]]
        py = sys.executable
        for fmt in ("png", "bmp"):
            cmd = [py, "-u", "01_camera_render_server.py", "--config", os.environ["CAMERA_3DGS_CONFIG"], "--image-format", fmt]
            res[f"QUEUE camera round trip, {fmt}, 6 objects"] = queue_round_trip(cmd, f"/tmp/isaac_queue/bench_cam_{fmt}",
                                                                                lambda: {"c2w": c2w.tolist(), "objects": obj_req}, "response_{i:06d}.*")
        for stride in (1, 4):
            cmd = [py, "-u", "03_lidar_render_server.py", "--checkpoint", os.environ["LIDAR_3DGS_CKPT"], "--no-raydrop", "--alpha-thresh", "0.99",
                   "--object-lidar-stride", str(stride)]
            res[f"QUEUE lidar round trip (dense cubemap), object stride {stride}, 6 objects"] = queue_round_trip(
                cmd, f"/tmp/isaac_queue/bench_lidar_{stride}", lambda: {"R_wl": R_wl.tolist(), "t_wl": t_wl.tolist(), "use_raydrop": False, "objects": obj_req},
                "response_{i:06d}.npz", n=10)
        cmd = [py, "-u", "03_lidar_render_server.py", "--checkpoint", os.environ["LIDAR_3DGS_CKPT"], "--no-raydrop", "--alpha-thresh", "0.99",
               "--object-lidar-stride", "4", "--livox-bag", str(bag)]
        res["QUEUE lidar round trip (livox pattern), object stride 4, 6 objects"] = queue_round_trip(
            cmd, "/tmp/isaac_queue/bench_lidar_livox", lambda: {"R_wl": R_wl.tolist(), "t_wl": t_wl.tolist(), "use_raydrop": False, "objects": obj_req},
            "response_{i:06d}.npz", n=10)
        res["QUEUE step: camera bmp + lidar stride 4 (sequential, as the clients do it)"] = round(
            res["QUEUE camera round trip, bmp, 6 objects"] + res["QUEUE lidar round trip (dense cubemap), object stride 4, 6 objects"] + res["dynamics GRU step"], 1)
        res["QUEUE step before: camera png + lidar stride 1"] = round(
            res["QUEUE camera round trip, png, 6 objects"] + res["QUEUE lidar round trip (dense cubemap), object stride 1, 6 objects"] + res["dynamics GRU step"], 1)
    print()
    for k, v in res.items():
        print(f"{k:78s} {v if isinstance(v, str) else f'{v:8.1f} ms'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
