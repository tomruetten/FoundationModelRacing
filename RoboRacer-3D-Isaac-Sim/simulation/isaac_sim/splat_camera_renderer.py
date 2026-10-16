"""
Pose -> RGB rendering bridge for the trained Camera-3DGS (gsplat) model.

This is the core building block for "Kamera Splat in Isaac Sim": given an
arbitrary 6-DoF camera pose (any pose, not just training poses), render the
RGB frame the Camera-3DGS model predicts for it. Isaac Sim will supply the
live pose each simulation step (vehicle dynamics); this module renders the
photorealistic frame instead of Isaac's native PBR renderer drawing the mesh.

Loads experiments/layout02_colmap_metric (pure-COLMAP-pose-trained, 22.29 dB
PSNR reference per thesis Ch.4.4) by default -- this is the checkpoint Felix's
"use COLMAP not MoCap" directive applies to.

Usage (standalone test, renders one frame at the dataset's own frame-0 pose
to confirm the bridge reproduces the same output as ns-eval/ns-viewer):
    conda run -n nerfstudio-py310 python3 splat_camera_renderer.py --selftest

Usage (as a module, for the Isaac Sim glue script):
    from splat_camera_renderer import SplatCameraRenderer
    r = SplatCameraRenderer()
    rgb = r.render(c2w_4x4_nerfstudio_convention)  # -> HxWx3 uint8 numpy array
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import os
import numpy as np
import torch

# Available checkpoints (all COLMAP-pose-trained per Felix's directive):
#   "neartriage" - layout02_near_triage_lidar_init: scene-area PSNR 25.05 dB
#                  (full-image 22.35 dB) -- RECOMMENDED for Isaac Sim deployment.
#                  near-triage frame selection + LiDAR point-cloud init.
#   "pure"       - layout02_colmap_metric: 22.29 dB full-image PSNR, 30k steps.
#   "lidarinit"  - layout02_colmap_lidar_init: 19.71 dB PSNR but LPIPS 0.277
#                  (best perceptual), looks good in Viser free-viewpoint nav.
# Paths relative to this script's location so they work on any machine.
_SCRIPTS_DIR = Path(__file__).parent
_EXPERIMENTS_DIR = _SCRIPTS_DIR.parent / "experiments"

CHECKPOINTS = {
    "neartriage": _EXPERIMENTS_DIR / "layout02_near_triage_lidar_init/outputs/dataset/splatfacto/2026-07-01_143505/config.yml",
    "pure": _EXPERIMENTS_DIR / "layout02_colmap_metric/outputs/dataset/splatfacto/2026-06-17_193124/config.yml",
    "lidarinit": _EXPERIMENTS_DIR / "layout02_colmap_lidar_init/outputs/dataset/splatfacto/2026-06-30_121444/config.yml",
    # "cropped" - layout02_near_triage_pure_colmap_cropped: 26.20 dB, best PSNR so
    # far. near-triage + pure COLMAP poses, 997 frames, physically cropped to
    # 1280x600 (drops bottom 200 rows -- no car-hood floaters). Checkpoint file
    # received from home machine 2026-07-04 (checkpoints_2026-07-04.tar.gz).
    "cropped": _EXPERIMENTS_DIR / "layout02_near_triage_pure_colmap_cropped/outputs/dataset/splatfacto/2026-07-02_114249/config.yml",
}
DEFAULT_CONFIG = CHECKPOINTS["neartriage"]


class SplatCameraRenderer:
    def __init__(self, config_path: Path = DEFAULT_CONFIG, device: str = "cuda",
                 data_dir: Path = None):
        from nerfstudio.utils.eval_utils import eval_setup
        import json

        self.device = device
        config_path = Path(config_path)
        resolved = {"data": None}

        def _relocate(config):
            """Make a checkpoint trained on another machine loadable here.

            nerfstudio's config.yml stores absolute paths from the training
            machine (e.g. /home/<user>/<project>/...). At inference we only
            need the dataset's transforms.json (intrinsics + applied_transform)
            and the nerfstudio_models/ directory next to config.yml.
            Resolution order for the data dir: explicit data_dir argument ->
            path from config.yml if it exists -> same path with the
            first three path components replaced by $DATA_DRIVE_ROOT ->
            <config dir>/../../dataset (New1 checkpoint layout) ->
            <config dir>/../../../../dataset/colmap_dataset_full (session layout).
            """
            candidates = []
            if data_dir is not None:
                candidates.append(Path(data_dir))
            orig = Path(config.data) if config.data is not None else None
            if orig is not None:
                candidates.append(orig)
                root = os.environ.get("DATA_DRIVE_ROOT")
                # /home/<user>/<project>/rest -> $DATA_DRIVE_ROOT/rest
                if root and len(orig.parts) > 4 and orig.parts[1] == "home":
                    candidates.append(Path(root, *orig.parts[4:]))
            candidates.append(config_path.parent.parent.parent / "dataset")
            candidates.append(config_path.parents[3] / "dataset" / "colmap_dataset_full")
            for c in candidates:
                if (c / "transforms.json").exists():
                    resolved["data"] = c
                    break
            if resolved["data"] is None:
                raise FileNotFoundError(
                    "No dataset dir with transforms.json found for this checkpoint. "
                    f"Tried: {[str(c) for c in candidates]}. Pass data_dir= explicitly.")
            config.data = resolved["data"]
            config.pipeline.datamanager.data = resolved["data"]
            # Gaussians are restored from the checkpoint; no need to read the
            # (possibly missing) sparse_pc.ply used for training initialisation.
            config.pipeline.datamanager.dataparser.load_3D_points = False
            models_dir = config_path.parent / "nerfstudio_models"
            if models_dir.exists():
                config.get_checkpoint_dir = lambda: models_dir
            return config

        config, pipeline, _, step = eval_setup(config_path, test_mode="inference",
                                               update_config_callback=_relocate)
        self.data_dir = resolved["data"]
        print(f"[SplatCameraRenderer] dataset dir: {self.data_dir}")
        self.pipeline = pipeline
        self.model = pipeline.model
        self.step = step

        # The model's Gaussians live in a re-centered/rescaled coordinate frame
        # (nerfstudio's auto_scale_poses/center_method), NOT the raw poses in
        # the dataset's transforms.json. dataparser_transforms.json records the
        # exact transform+scale applied at training time; we must apply the
        # SAME transform to any new (e.g. Isaac-Sim-supplied) pose before
        # rendering, or the camera will be looking at empty space.
        dpt_path = config_path.parent / "dataparser_transforms.json"
        dpt = json.load(open(dpt_path))
        composed = np.eye(4)
        composed[:3, :4] = np.array(dpt["transform"])  # 3x4
        self.dataparser_scale = float(dpt["scale"])

        # dataparser_transforms.json's "transform" is transform_matrix @
        # applied_transform when transforms.json carries an "applied_transform"
        # (nerfstudio_dataparser.py). Poses taken from transforms.json are
        # already past applied_transform, so it must be divided out once or
        # renders come out ~90 deg rotated (bug found on the New1 checkpoint,
        # 2026-07-15). Checkpoints whose transforms.json has no
        # applied_transform are unaffected by this branch.
        meta = json.load(open(self.data_dir / "transforms.json"))
        if "applied_transform" in meta:
            applied = np.eye(4)
            applied[:3, :4] = np.array(meta["applied_transform"])
            composed = composed @ np.linalg.inv(applied)
            note = "applied_transform divided out"
        else:
            note = "no applied_transform in transforms.json"
        self.dataparser_transform = composed[:3, :4]
        print(f"[SplatCameraRenderer] dataparser scale={self.dataparser_scale:.6f} "
              f"(loaded from {dpt_path.name}, {note})")

        # intrinsics: pull from the dataset's own cameras (constant for this dataset)
        train_cams = pipeline.datamanager.train_dataset.cameras
        self.fx = float(train_cams.fx[0])
        self.fy = float(train_cams.fy[0])
        self.cx = float(train_cams.cx[0])
        self.cy = float(train_cams.cy[0])
        self.width = int(train_cams.width[0])
        self.height = int(train_cams.height[0])
        print(
            f"[SplatCameraRenderer] loaded step={step}, "
            f"intrinsics fx={self.fx:.2f} fy={self.fy:.2f} "
            f"cx={self.cx:.2f} cy={self.cy:.2f} {self.width}x{self.height}"
        )

    def to_model_frame(self, c2w_raw: np.ndarray) -> np.ndarray:
        """Apply the dataparser's center/scale transform to a RAW pose (the
        same convention as the dataset's transforms.json) so it lands in the
        coordinate frame the trained Gaussians actually live in."""
        c2w_h = np.eye(4)
        c2w_h[:3, :4] = c2w_raw[:3, :4]
        applied = self.dataparser_transform @ c2w_h  # (3,4) @ (4,4) -> (3,4)
        applied[:, 3] *= self.dataparser_scale
        return applied

    def world_to_model(self) -> np.ndarray:
        """4x4 similarity raw metric world (transforms.json frame) -> model (training) frame."""
        M = np.eye(4)
        M[:3, :4] = self.dataparser_transform
        return np.diag([self.dataparser_scale] * 3 + [1.0]) @ M

    object_soften = 1.3   # footprint factor for inserted objects (see ObjectSplat.posed); an object's meta["soften"] overrides

    @contextmanager
    def objects_inserted(self, objects):
        """Temporarily append posed object gaussians (object_splats.ObjectSplat) to the model.
        objects: list of (ObjectSplat, T_wo[, part_transforms]) with T_wo object -> raw world frame (4x4).
        SH coefficients are rotated with the object, so colours stay view-correct."""
        if not objects:
            yield
            return
        import torch
        gp = self.model.gauss_params
        keys = ("means", "scales", "quats", "opacities", "features_dc", "features_rest")
        base = {k: gp[k] for k in keys}
        parts = {k: [base[k].data] for k in keys}
        n_sh = base["features_rest"].shape[1] + 1
        M = self.world_to_model()
        dev = base["means"].device
        soften = getattr(self, "object_soften", 1.3)
        for entry in objects:
            obj, T_wo, part_tf = (entry + (None,))[:3] if len(entry) == 2 else entry
            p = obj.posed(M @ np.asarray(T_wo, dtype=np.float64), device=dev, n_sh=n_sh, part_transforms=part_tf,
                          soften=float(obj.meta.get("soften", soften)))
            parts["means"].append(p["means"])
            parts["scales"].append(p["scales_log"])
            parts["quats"].append(p["quats"])
            parts["opacities"].append(p["opacities"].reshape(-1, *base["opacities"].shape[1:]))
            parts["features_dc"].append(p["sh"][:, 0])
            parts["features_rest"].append(p["sh"][:, 1:])
        try:
            for k in keys:
                gp[k] = torch.nn.Parameter(torch.cat(parts[k]).to(base[k].dtype), requires_grad=False)
            yield
        finally:
            for k in keys:
                gp[k] = base[k]

    def render(self, c2w: np.ndarray, already_in_model_frame: bool = False, objects=None) -> np.ndarray:
        """
        objects: optional list of (ObjectSplat, T_wo) inserted into the scene for this frame
             (T_wo object -> the same raw world frame as c2w).
        c2w: 3x4 or 4x4 camera-to-world matrix, nerfstudio/OpenGL convention
             (X-right, Y-up, Z-back), in the SAME raw coordinate frame as the
             dataset's transforms.json (i.e. real-world COLMAP-metric frame)
             -- the dataparser transform is applied automatically unless
             already_in_model_frame=True.
        Returns: HxWx3 uint8 RGB numpy array.
        """
        from nerfstudio.cameras.cameras import Cameras, CameraType

        c2w_model = c2w[:3, :4] if already_in_model_frame else self.to_model_frame(c2w)
        c2w_t = torch.tensor(c2w_model, dtype=torch.float32).unsqueeze(0)
        camera = Cameras(
            camera_to_worlds=c2w_t,
            fx=torch.tensor([self.fx]),
            fy=torch.tensor([self.fy]),
            cx=torch.tensor([self.cx]),
            cy=torch.tensor([self.cy]),
            width=torch.tensor([self.width]),
            height=torch.tensor([self.height]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(self.device)

        with torch.no_grad(), self.objects_inserted(objects):
            outputs = self.model.get_outputs_for_camera(camera[0:1])
        rgb = outputs["rgb"].clamp(0, 1).cpu().numpy()
        return (rgb * 255).astype(np.uint8)


def _selftest():
    import json

    config_path = CHECKPOINTS[_args.checkpoint]
    renderer = SplatCameraRenderer(config_path=config_path)
    dataset_dir = config_path.parents[4] / "dataset"
    transforms = json.load(open(dataset_dir / "transforms.json"))
    # NOTE: frame index 0 in this dataset is a known overexposed outlier frame
    # (mean brightness ~246/255 vs ~120-130 for normal frames) -- use frame 50
    # instead so the selftest sanity-checks against a representative frame.
    test_frame = transforms["frames"][50]
    c2w = np.array(test_frame["transform_matrix"])

    rgb = renderer.render(c2w)
    gt_path = dataset_dir / test_frame["file_path"]
    from PIL import Image
    gt = np.array(Image.open(gt_path).convert("RGB"))
    diff = np.abs(rgb.astype(float) - gt.astype(float)).mean()
    print(f"[selftest] rendered shape={rgb.shape} mean={rgb.mean():.1f} | "
          f"GT mean={gt.mean():.1f} | mean abs diff={diff:.1f} "
          f"(should be ~8-12, matching the model's ~22dB PSNR; large diff = bug)")

    out_path = _SCRIPTS_DIR.parent / "isaac_sim_output" / "splat_camera_renderer_selftest.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)
    print(f"[selftest] saved to {out_path} -- inspect visually to confirm it matches "
          f"the dataset's frame at index 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS.keys()), default="pure",
                         help="'pure'=22.29dB highest-PSNR COLMAP checkpoint, "
                              "'lidarinit'=19.71dB but better LPIPS/visual quality")
    _args = parser.parse_args()
    if _args.selftest:
        _selftest()
