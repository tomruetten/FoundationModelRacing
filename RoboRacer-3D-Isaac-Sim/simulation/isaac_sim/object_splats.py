"""Object Gaussian splats: a standalone set of gaussians (an opponent car, a cone) in its
own metric frame that is inserted into BOTH scene maps at render time -- the camera 3DGS
(splatfacto) and the LiDAR 3DGS (raw gsplat) -- and exported as a second
ParticleField3DGaussianSplat prim for Isaac Sim.

Object frame: metres, z up, +x forward, origin on the ground under the object's centre.
T_wo (4x4): object -> map world. Rotation plus translation, optionally a uniform scale
(the camera map lives in nerfstudio units, the 3DGRUT Isaac scene in metres, ...).

Why not just concatenate the gaussians? Two reasons the naive way goes wrong:
  * the camera map stores view-dependent colour as spherical harmonics (degree 3) in the
    frame the gaussians live in. Rotating a gaussian without rotating its SH coefficients
    gives colour casts (seen 2026-09-10 with the whole scene). sh_rotation() builds the
    exact real-SH rotation matrix for gsplat's basis, so posed() rotates the coefficients.
  * the LiDAR map has no view dependence but different colour channels (intensity,
    raydrop logit). lidar_colors() gives object gaussians a never-dropped return.

File format (npz): means (N,3) m, scales_log (N,3), quats (N,4) w,x,y,z, opacities (N,)
logit, sh (N,16,3) with sh[:,0] = features_dc, plus meta (json string).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# gsplat / 3DGS real spherical-harmonics constants (same basis and ordering as
# gsplat.spherical_harmonics and nerfstudio's splatfacto)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
N_SH = 16
LIDAR_OBJECT_MIN_OPACITY = 0.95   # objects are solid surfaces for the LiDAR map
LIDAR_OBJECT_RAYDROP_LOGIT = -10.0   # sigmoid -> 0: never dropped


# ----------------------------------------------------------------------------- SH maths
def sh_basis(dirs: np.ndarray) -> np.ndarray:
    """Real SH basis (degree 0..3, gsplat ordering) at unit directions. (K,3) -> (K,16)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    cols = [
        np.full_like(x, SH_C0),
        -SH_C1 * y, SH_C1 * z, -SH_C1 * x,
        SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * xz, SH_C2[4] * (xx - yy),
        SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z, SH_C3[2] * y * (4 * zz - xx - yy),
        SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
        SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy),
    ]
    return np.stack(cols, axis=1)


def sh_eval(sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Colour of each gaussian for its own view direction. sh (N,K,3), dirs (N,3) unit -> (N,3)."""
    B = sh_basis(dirs)[:, : sh.shape[1]]
    return np.einsum("nk,nkc->nc", B, sh)


def sh_rotation(R: np.ndarray, n_coeffs: int = N_SH) -> np.ndarray:
    """Matrix D with  sh_world = D @ sh_object  (per colour channel) for an object rotated by
    R (object -> world). Exact: the rotated degree-l basis lies in the span of the degree-l
    basis, so projecting Y(R^T d) onto Y(d) on sample directions recovers it (block diagonal,
    orthogonal). Works for any truncation n_coeffs in {1, 4, 9, 16}."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(512, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    B = sh_basis(d)[:, :n_coeffs]
    B_rot = sh_basis(d @ R)[:, :n_coeffs]          # rows Y(R^T d_k)
    D, *_ = np.linalg.lstsq(B, B_rot, rcond=None)  # B @ D = B_rot  ->  D = M^T, c' = D @ c
    return D


def rotate_sh(sh: np.ndarray, R: np.ndarray) -> np.ndarray:
    """sh (N,K,3) in the object frame -> coefficients valid after rotating the object by R."""
    D = sh_rotation(R, sh.shape[1])
    return np.einsum("ij,njc->nic", D, sh)


# ---------------------------------------------------------------------- rotation helpers
def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z). Shepperd's method."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, (w,x,y,z), broadcasting over leading dims."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def decompose_similarity(T: np.ndarray):
    """4x4 similarity -> (R rotation, s uniform scale, t translation)."""
    A = np.asarray(T, dtype=np.float64)[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    if s <= 0:
        raise ValueError("transform is not a proper similarity (det <= 0)")
    R = A / s
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        raise ValueError("rotation block is not orthogonal (non-uniform scale or shear)")
    return R, s, np.asarray(T, dtype=np.float64)[:3, 3].copy()


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_axis(axis, angle: float) -> np.ndarray:
    """Rotation about an arbitrary axis (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _quat_mul_t(q1, q2):
    """Hamilton product on torch tensors (w,x,y,z), broadcasting."""
    import torch
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------- container
class ObjectSplat:
    """Gaussians of one object in the object frame (metres, z up, +x forward)."""

    KEYS = ("means", "scales_log", "quats", "opacities", "sh")

    def __init__(self, means, scales_log, quats, opacities, sh, meta: dict | None = None, parts=None):
        self.means = np.asarray(means, dtype=np.float32)
        self.scales_log = np.asarray(scales_log, dtype=np.float32)
        self.quats = np.asarray(quats, dtype=np.float32)
        self.quats /= np.linalg.norm(self.quats, axis=1, keepdims=True)
        self.opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)
        sh = np.asarray(sh, dtype=np.float32)
        if sh.shape[1] < N_SH:   # pad missing higher orders with zeros
            sh = np.concatenate([sh, np.zeros((len(sh), N_SH - sh.shape[1], 3), np.float32)], axis=1)
        self.sh = sh
        self.meta = dict(meta or {})
        # parts: per-gaussian index into meta["parts"] (list of dicts with name, kind
        # ("body" | "wheel" | "shadow"), and for wheels centre (m, object frame), axis, radius,
        # steer (bool)); None = one rigid body. Parts with "lidar": false are skipped in the
        # LiDAR map (the contact shadow is a purely visual thing).
        self.parts = None if parts is None else np.asarray(parts, dtype=np.int16).reshape(-1)
        self._dev = {}
        self._proxies = {}

    # -- io
    @classmethod
    def load(cls, path) -> "ObjectSplat":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        obj = cls(z["means"], z["scales_log"], z["quats"], z["opacities"], z["sh"], meta,
                  parts=z["parts"] if "parts" in z else None)
        obj.meta.setdefault("path", str(path))
        return obj

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {"parts": self.parts} if self.parts is not None else {}
        np.savez_compressed(path, means=self.means, scales_log=self.scales_log, quats=self.quats,
                            opacities=self.opacities, sh=self.sh, meta=json.dumps(self.meta), **extra)
        return path

    def proxy(self, stride: int) -> "ObjectSplat":
        """Every stride-th gaussian with a sqrt(stride) larger footprint: same surface coverage,
        stride times fewer gaussians. Used for the LiDAR layer, where a 2 mm CAD sampling buys
        nothing (returns on the car: 57 vs 59 at stride 4, LiDAR time 97 -> 31 ms). Cached."""
        stride = int(max(1, stride))
        if stride == 1:
            return self
        if stride not in self._proxies:
            idx = np.arange(0, self.n, stride)
            self._proxies[stride] = ObjectSplat(self.means[idx], self.scales_log[idx] + 0.5 * np.log(stride), self.quats[idx],
                                                self.opacities[idx], self.sh[idx], self.meta,
                                                parts=None if self.parts is None else self.parts[idx])
        return self._proxies[stride]

    # -- parts
    @property
    def part_meta(self) -> list:
        return self.meta.get("parts", [])

    def part_index(self, name: str) -> int:
        for i, p in enumerate(self.part_meta):
            if p["name"] == name:
                return i
        raise KeyError(name)

    def subset(self, mask) -> "ObjectSplat":
        """New object with the selected gaussians (same parts table)."""
        mask = np.asarray(mask, bool)
        return ObjectSplat(self.means[mask], self.scales_log[mask], self.quats[mask], self.opacities[mask], self.sh[mask],
                           self.meta, parts=None if self.parts is None else self.parts[mask])

    def with_part(self, other: "ObjectSplat", part: dict) -> "ObjectSplat":
        """Append another gaussian set as a new part (e.g. a contact shadow)."""
        parts_meta = list(self.part_meta) or [{"name": "body", "kind": "body"}]
        own = self.parts if self.parts is not None else np.zeros(self.n, np.int16)
        parts_meta.append(dict(part))
        new = np.full(other.n, len(parts_meta) - 1, np.int16)
        meta = dict(self.meta); meta["parts"] = parts_meta
        return ObjectSplat(np.vstack([self.means, other.means]), np.vstack([self.scales_log, other.scales_log]),
                           np.vstack([self.quats, other.quats]), np.concatenate([self.opacities, other.opacities]),
                           np.concatenate([self.sh, other.sh]), meta, parts=np.concatenate([own, new]))

    def articulated(self, wheel_angle: float = 0.0, steer_angle: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> dict:
        """Per-part local rigid transforms (4x4, object frame) for a driving state: wheels turn
        by wheel_angle [rad] about their axle, steerable wheels yaw by steer_angle [rad] about the
        vertical through their centre, and the whole object pitches (nose down positive) and
        rolls (right side down positive) about the ground contact centre. Returns
        {part_index: T} for use with posed(part_transforms=...)."""
        out = {}
        body = np.eye(4)
        body[:3, :3] = rot_y(pitch) @ rot_x(roll)
        body[:3, 3] = [0.0, 0.0, 0.0]
        for i, p in enumerate(self.part_meta):
            T = body.copy()
            if p.get("kind") == "wheel":
                c = np.asarray(p["centre"], dtype=np.float64)
                a = np.asarray(p.get("axis", [0.0, 1.0, 0.0]), dtype=np.float64)
                Rw = rot_axis(a, wheel_angle)
                if p.get("steer") and steer_angle:
                    Rw = rot_z(steer_angle) @ Rw
                Tw = np.eye(4); Tw[:3, :3] = Rw; Tw[:3, 3] = c - Rw @ c
                T = body @ Tw
            elif p.get("kind") == "shadow":
                T = np.eye(4)            # the shadow stays on the floor
            out[i] = T
        return out

    @property
    def n(self) -> int:
        return len(self.means)

    def bbox(self):
        return self.means.min(axis=0), self.means.max(axis=0)

    def __repr__(self):
        lo, hi = self.bbox()
        return f"ObjectSplat({self.meta.get('name', '?')}, {self.n} gaussians, extent {np.round(hi - lo, 2).tolist()} m)"

    # -- geometry
    def _base(self, device):
        """Base tensors on the device (cached)."""
        import torch
        if device not in self._dev:
            self._dev[device] = {k: torch.from_numpy(getattr(self, k)).to(device) for k in self.KEYS}
        return self._dev[device]

    def posed(self, T_wo: np.ndarray, device="cuda", n_sh: int = N_SH, opaque_min: float | None = None,
              part_transforms: dict | None = None, for_lidar: bool = False, soften: float = 1.0) -> dict:
        """Gaussians transformed by the similarity T_wo (object -> target frame).
        part_transforms: {part_index: 4x4 rigid transform in the object frame} applied before
        T_wo (see articulated()); for_lidar drops parts flagged "lidar": false; soften scales
        every gaussian's footprint (mesh-derived objects are razor sharp next to a splat scene;
        1.3 to 1.4 matches New1's softness, 1.8 and more bloats the geometry).
        Returns torch tensors: means (N,3), scales_log (N,3), quats (N,4), opacities (N,),
        sh (N,n_sh,3) with SH coefficients rotated into the target frame."""
        import torch
        R, s, t = decompose_similarity(T_wo)
        b = self._base(device)
        keep = None
        if for_lidar and self.parts is not None:
            skip = [i for i, p in enumerate(self.part_meta) if p.get("lidar", True) is False]
            if skip:
                keep = torch.from_numpy(~np.isin(self.parts, skip)).to(device)
        means_o, quats_o, sh_o = b["means"], b["quats"], b["sh"][:, :n_sh]
        if part_transforms and self.parts is not None:
            means_o, quats_o, sh_o = means_o.clone(), quats_o.clone(), sh_o.clone()
            parts_t = torch.from_numpy(self.parts.astype(np.int64)).to(device)
            for pi, Tp in part_transforms.items():
                Rp, _, tp = decompose_similarity(Tp)
                if np.allclose(Rp, np.eye(3), atol=1e-12) and np.allclose(tp, 0, atol=1e-12):
                    continue
                m = parts_t == pi
                if not bool(m.any()):
                    continue
                Rp_t = torch.tensor(Rp, dtype=torch.float32, device=device)
                means_o[m] = means_o[m] @ Rp_t.T + torch.tensor(tp, dtype=torch.float32, device=device)
                qp = torch.tensor(quat_from_matrix(Rp), dtype=torch.float32, device=device)
                quats_o[m] = _quat_mul_t(qp[None], quats_o[m])
                Dp = torch.tensor(sh_rotation(Rp, n_sh), dtype=torch.float32, device=device)
                sh_o[m] = torch.einsum("ij,njc->nic", Dp, sh_o[m])
        A = torch.tensor(s * R, dtype=torch.float32, device=device)
        means = means_o @ A.T + torch.tensor(t, dtype=torch.float32, device=device)
        scales_log = b["scales_log"] + float(np.log(s * soften))
        qR = torch.tensor(quat_from_matrix(R), dtype=torch.float32, device=device)
        quats = _quat_mul_t(qR[None], quats_o)
        D = torch.tensor(sh_rotation(R, n_sh), dtype=torch.float32, device=device)
        sh = torch.einsum("ij,njc->nic", D, sh_o)
        opac = b["opacities"]
        if opaque_min is not None:
            opac = torch.clamp(opac, min=_logit(opaque_min))
        out = {"means": means, "scales_log": scales_log, "quats": quats, "opacities": opac, "sh": sh}
        if keep is not None:
            out = {k: v[keep] for k, v in out.items()}
        return out

    def lidar_mask(self) -> np.ndarray:
        """Gaussians that take part in the LiDAR map (parts with "lidar": false excluded)."""
        if self.parts is None:
            return np.ones(self.n, bool)
        skip = [i for i, p in enumerate(self.part_meta) if p.get("lidar", True) is False]
        return ~np.isin(self.parts, skip)

    def lidar_colors(self, n_channels: int, device="cuda"):
        """Colour channels for the LiDAR map (for the gaussians of lidar_mask()): ch0 intensity,
        ch1 raydrop logit (never dropped)."""
        import torch
        m = self.lidar_mask()
        c = torch.zeros((int(m.sum()), n_channels), dtype=torch.float32, device=device)
        # intensity: luminance of the base colour, mapped to the LiDAR model's 0..1 range
        rgb = np.clip(0.5 + SH_C0 * self.sh[m, 0], 0.0, 1.0)
        c[:, 0] = torch.from_numpy((0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).astype(np.float32)).to(device)
        if n_channels > 1:
            c[:, 1] = LIDAR_OBJECT_RAYDROP_LOGIT
        return c


# ------------------------------------------------------------------------------ registry
class ObjectRegistry:
    """Objects known to a render server: registered at start (--object name=path.npz) or
    referenced by path inside a request. Request format:
        "objects": [{"name": "opponent", "T_wo": [[4x4]],
                     "articulation": {"wheel_angle": 1.2, "steer_angle": 0.1, "pitch": 0.0, "roll": 0.0}},
                    {"path": "cone.npz", "T_wo": [[4x4]]}]
    articulation is optional and needs an object with parts (import_object_splat.py tags wheels).
    """

    def __init__(self, specs=()):
        self.objects: dict[str, ObjectSplat] = {}
        for spec in specs or ():
            name, _, path = spec.partition("=")
            if not path:
                path, name = name, Path(name).stem
            self.objects[name] = ObjectSplat.load(path)
            self.objects[name].meta.setdefault("name", name)
            print(f"[objects] {name}: {self.objects[name]}")

    def resolve(self, req_objects) -> list[tuple[ObjectSplat, np.ndarray]]:
        out = []
        for o in req_objects or []:
            key = o.get("name") or o.get("path")
            if key not in self.objects:
                if o.get("path") and Path(o["path"]).exists():
                    self.objects[key] = ObjectSplat.load(o["path"])
                    self.objects[key].meta.setdefault("name", key)
                    print(f"[objects] loaded from request: {self.objects[key]}")
                else:
                    print(f"[objects] WARNING unknown object {key!r}, skipped")
                    continue
            obj = self.objects[key]
            art = o.get("articulation")
            part_tf = obj.articulated(**{k: float(v) for k, v in art.items()}) if (art and obj.parts is not None) else None
            out.append((obj, np.asarray(o["T_wo"], dtype=np.float64), part_tf))
        return out


# ------------------------------------------------------------------------- floor plane
class FloorPlane:
    """The reconstructed floor of a scene as a plane: unit normal n (pointing up) and a point
    p0 on it, in the scene's world frame. Objects placed with it stand on the floor the
    cameras actually see, instead of hanging a fixed offset below the (noisy) camera path
    (New1: the camera height above the reconstructed floor jitters by 3.5 cm from frame to
    frame and drifts by 3 cm along the lap, which made a 5 cm wheel sink into the floor)."""

    def __init__(self, normal, point, meta: dict | None = None, clearance: float = 0.0):
        n = np.asarray(normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)
        self.point = np.asarray(point, dtype=np.float64)
        self.meta = dict(meta or {})
        # Visual clearance: objects are placed this far above the plane. The floor of a splat is
        # not a surface but a band of opaque gaussians (New1: centres spread -6..+7.5 cm around
        # the plane); an object exactly on the plane loses its lowest centimetres in that band
        # (wheels look sunk), and culling the band costs 25 dB of floor appearance.
        self.clearance = float(clearance)

    def height(self, p) -> np.ndarray:
        """Signed distance of point(s) above the plane."""
        return (np.asarray(p, dtype=np.float64) - self.point) @ self.normal

    def project(self, p) -> np.ndarray:
        """Foot point(s) of p on the plane."""
        p = np.asarray(p, dtype=np.float64)
        return p - np.multiply.outer(self.height(p), self.normal)

    def transformed(self, S: np.ndarray) -> "FloorPlane":
        """Plane expressed in another frame, X' = S X (similarity, uniform scale allowed)."""
        R, sc, _ = decompose_similarity(S)
        return FloorPlane(R @ self.normal, (S @ np.append(self.point, 1.0))[:3], self.meta, clearance=self.clearance * sc)

    @classmethod
    def fit(cls, points, up_hint=None, trim: float = 0.03, iters: int = 4, meta: dict | None = None) -> "FloorPlane":
        """Robust plane through floor samples (N,3): least squares, then repeatedly drop
        samples farther than `trim` metres from the plane and refit (cones, floaters and
        under-floor noise are the outliers)."""
        P = np.asarray(points, dtype=np.float64)
        keep = np.ones(len(P), bool)
        for _ in range(iters):
            Q = P[keep]
            c = Q.mean(axis=0)
            _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
            n = vt[2]
            if up_hint is not None and np.dot(n, up_hint) < 0:
                n = -n
            res = (P - c) @ n
            new = np.abs(res) <= trim
            if new.sum() < 10 or np.array_equal(new, keep):
                keep = new if new.sum() >= 10 else keep
                break
            keep = new
        res = (P[keep] - c) @ n
        m = dict(meta or {})
        m.update({"n_samples": int(len(P)), "n_inliers": int(keep.sum()), "trim_m": trim,
                  "inlier_rms_m": float(np.sqrt(np.mean(res ** 2)))})
        return cls(n, c, m)

    def save(self, path) -> Path:
        path = Path(path)
        json.dump({"normal": self.normal.tolist(), "point": self.point.tolist(), "clearance_m": self.clearance,
                   "meta": self.meta}, open(path, "w"), indent=1)
        return path

    @classmethod
    def load(cls, path) -> "FloorPlane":
        d = json.load(open(path))
        return cls(d["normal"], d["point"], d.get("meta"), clearance=d.get("clearance_m", 0.0))

    def __repr__(self):
        return (f"FloorPlane(normal={np.round(self.normal, 4).tolist()}, rms={self.meta.get('inlier_rms_m', float('nan')):.3f} m, "
                f"clearance={self.clearance * 100:.1f} cm)")


def floor_samples_from_scene(means_w: np.ndarray, opacities_logit: np.ndarray, scales_log_w: np.ndarray,
                             trajectory_xyz: np.ndarray, up: np.ndarray, corridor: float = 1.2,
                             cell: float = 0.10, min_per_cell: int = 15) -> np.ndarray:
    """Floor samples from a scene's gaussians: opaque, small gaussians within `corridor` metres
    (in the floor plane) of the camera path, at most 0.6 m below / 0.05 m above the median
    camera height along `up`; per `cell` x `cell` bin the modal height (2 cm histogram) is one
    sample. Returns (M,3) world points."""
    from scipy.spatial import cKDTree
    ex = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(up, ex)
    Rl = np.stack([ex, ey, up])                       # world -> levelled (z along up)
    Pl = means_w @ Rl.T
    Cl = trajectory_xyz @ Rl.T
    opac = 1.0 / (1.0 + np.exp(-opacities_logit))
    sel = (opac > 0.5) & (np.exp(scales_log_w).max(axis=1) < 0.15)
    d, _ = cKDTree(Cl[:, :2]).query(Pl[:, :2])
    zc = np.median(Cl[:, 2])
    sel &= (d < corridor) & (Pl[:, 2] > zc - 0.6) & (Pl[:, 2] < zc + 0.05)
    Q = Pl[sel]
    ij = np.floor(Q[:, :2] / cell).astype(np.int64)
    keys = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.argsort(keys)
    keys, Q = keys[order], Q[order]
    uniq, start = np.unique(keys, return_index=True)
    ends = np.r_[start[1:], len(keys)]
    out = []
    for a, b in zip(start, ends):
        if b - a < min_per_cell:
            continue
        z = Q[a:b, 2]
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
        out.append([Q[a:b, 0].mean(), Q[a:b, 1].mean(), edges[np.argmax(hist)] + 0.01])
    return np.array(out) @ Rl                          # back to world


# ------------------------------------------------------------------------- pose helpers
def trajectory_up(frames, min_frames: int = 20) -> np.ndarray:
    """Floor normal of a recorded lap: the camera drives at constant height above the
    floor, so the best-fit plane through its positions is parallel to the floor. Sign
    follows the cameras' mean +y (OpenGL up). Falls back to +z for short trajectories.
    Needed because COLMAP frames are not gravity aligned (New1: 9.5 deg tilt)."""
    P = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)
    if len(P) < min_frames:
        return np.array([0.0, 0.0, 1.0])
    Q = P - P.mean(axis=0)
    _, sv, vt = np.linalg.svd(Q, full_matrices=False)
    if sv[1] < 1e-3:                       # a straight line does not define a plane
        return np.array([0.0, 0.0, 1.0])
    n = vt[2]
    cam_up = np.mean([np.array(f["transform_matrix"])[:3, 1] for f in frames], axis=0)
    return n if np.dot(n, cam_up) >= 0 else -n


def pose_from_camera_frame(c2w: np.ndarray, ground_offset: float = 0.20, lateral: float = 0.0,
                           ahead: float = 0.0, up=None, floor: "FloorPlane | None" = None) -> np.ndarray:
    """Object pose (4x4, object -> world) derived from a camera-to-world matrix (OpenGL
    convention): +x = camera heading projected on the floor, origin on the floor, shifted
    `lateral` m to the left and `ahead` m forward. Used to drive an opponent along a
    recorded trajectory.

    floor: a FloorPlane of the scene -> the origin is the camera position projected onto the
    reconstructed floor (plus the plane's visual clearance) and +z its normal (preferred: no
    dependence on the camera height above the floor).
    Without it: +z = `up` (default world z; trajectory_up() for tilted COLMAP frames) and the
    origin sits `ground_offset` below the camera."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if floor is not None:
        up = floor.normal
        base = floor.project(c2w[:3, 3]) + floor.clearance * up
    else:
        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        base = c2w[:3, 3] - ground_offset * up
    fwd = -c2w[:3, 2].copy()
    fwd -= np.dot(fwd, up) * up
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.cross(up, np.array([0.0, 1.0, 0.0]))
    left = np.cross(up, fwd)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([fwd, left, up])
    T[:3, 3] = base + ahead * fwd + lateral * left
    return T


def path_arclength(frames) -> np.ndarray:
    """Cumulative distance along a trajectory's camera positions [m per frame]."""
    P = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def frame_ahead(i: int, cum: np.ndarray, gap_m: float | None, lead_frames: int | None) -> int:
    """Index of the trajectory frame the opponent occupies for ego frame i: the first frame
    at least gap_m metres further along the path (speed independent), or i + lead_frames."""
    n = len(cum)
    if gap_m is not None:
        return int(min(np.searchsorted(cum, cum[i] + gap_m), n - 1))
    return int(min(i + (lead_frames or 0), n - 1))


def similarity_from_frames(frames) -> np.ndarray | None:
    """A trajectory converted by scripts/wrap_3dgrut_usd.py stores both transform_matrix
    (Isaac frame) and transform_matrix_source (render-server frame). Recover the similarity
    S with  X_isaac = S X_source  (4x4, uniform scale folded into the rotation block), or
    None when the frames carry no source poses."""
    pairs = [(np.array(f["transform_matrix"]), np.array(f["transform_matrix_source"]))
             for f in frames if "transform_matrix_source" in f]
    if not pairs:
        return None
    Ri, Rs = pairs[0][0][:3, :3], pairs[0][1][:3, :3]
    R = Ri @ Rs.T
    P_i = np.array([p[0][:3, 3] for p in pairs])
    P_s = np.array([p[1][:3, 3] for p in pairs])
    d_i = np.linalg.norm(P_i - P_i[0], axis=1)
    j = int(np.argmax(d_i))
    s = d_i[j] / max(np.linalg.norm(P_s[j] - P_s[0]), 1e-12) if len(pairs) > 1 else 1.0
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = P_i[0] - s * R @ P_s[0]
    return S


# ----------------------------------------------------------------------------- extraction
def nerfstudio_world_from_train(dataparser_json, transforms_json=None) -> np.ndarray:
    """4x4 mapping nerfstudio training units -> the raw metric frame of transforms.json."""
    dp = json.load(open(dataparser_json))
    T = np.eye(4)
    T[:3, :4] = np.array(dp["transform"])
    s = float(dp["scale"])
    if transforms_json:
        meta = json.load(open(transforms_json))
        if "applied_transform" in meta:
            A = np.eye(4)
            A[:3, :4] = np.array(meta["applied_transform"])
            T = T @ np.linalg.inv(A)
    return np.linalg.inv(T) @ np.diag([1 / s, 1 / s, 1 / s, 1.0])


def load_nerfstudio_gaussians(ckpt_path) -> dict:
    """Raw splatfacto gaussians from a nerfstudio checkpoint (training frame, numpy)."""
    import torch
    gp = torch.load(ckpt_path, map_location="cpu", weights_only=False)["pipeline"]
    g = lambda k: gp[f"_model.gauss_params.{k}"].float().numpy()  # noqa: E731
    sh = np.concatenate([g("features_dc")[:, None, :], g("features_rest")], axis=1)
    return {"means": g("means"), "scales_log": g("scales"), "quats": g("quats"),
            "opacities": g("opacities").reshape(-1), "sh": sh}


def estimate_floor_z(means_w: np.ndarray, opac: np.ndarray, center, radius: float = 0.8,
                     bin_m: float = 0.02) -> float:
    """Height of the ground around `center`: mode of a z histogram of opaque gaussians in
    an xy ring around the object (the ground has by far the most gaussians)."""
    center = np.asarray(center)
    d = np.linalg.norm(means_w[:, :2] - center[:2], axis=1)
    sel = (d < radius) & (d > 0.25) & (opac > 0.5) & (means_w[:, 2] < center[2])
    z = means_w[sel, 2]
    if len(z) < 50:
        raise ValueError("not enough ground gaussians around the object to estimate the floor")
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    hist, edges = np.histogram(z, bins=edges)
    return float(edges[np.argmax(hist)] + bin_m / 2)


def extract_object(gauss: dict, world_from_train: np.ndarray, center, size, *, yaw_deg: float = 0.0,
                   floor_z: float | None = None, above_floor: float = 0.015,
                   opacity_thresh: float = 0.3, max_scale: float | None = None,
                   meta: dict | None = None) -> ObjectSplat:
    """Cut the gaussians inside an axis-aligned box (world frame, metres) out of a scene and
    re-express them in the object frame: origin at (center xy, floor_z), +x rotated by yaw.
    SH coefficients and covariances are rotated along.

    yaw_deg: heading of the object's forward axis in the world frame (so that +x_object
    points forward). floor_z: ground height; default = bottom of the box."""
    R_wt, s_wt, t_wt = decompose_similarity(world_from_train)
    means_w = gauss["means"] @ (s_wt * R_wt).T + t_wt
    scales_w = gauss["scales_log"] + np.log(s_wt)
    center, size = np.asarray(center, dtype=np.float64), np.asarray(size, dtype=np.float64)
    lo, hi = center - size / 2, center + size / 2
    if floor_z is None:
        floor_z = float(lo[2])
    lo[2] = max(lo[2], floor_z + above_floor)
    keep = np.all((means_w >= lo) & (means_w <= hi), axis=1)
    keep &= 1 / (1 + np.exp(-gauss["opacities"])) >= opacity_thresh
    if max_scale:
        keep &= np.exp(scales_w).max(axis=1) <= max_scale
    idx = np.where(keep)[0]
    if len(idx) == 0:
        raise ValueError("no gaussians inside the box")

    # world -> object: translate to the origin, then undo the heading
    R_ow = rot_z(-np.radians(yaw_deg))
    origin = np.array([center[0], center[1], floor_z])
    means_o = (means_w[idx] - origin) @ R_ow.T
    R_ot = R_ow @ R_wt                     # training frame -> object frame (rotation only)
    quats_o = quat_mul(quat_from_matrix(R_ot)[None], gauss["quats"][idx] / np.linalg.norm(gauss["quats"][idx], axis=1, keepdims=True))
    sh_o = rotate_sh(gauss["sh"][idx], R_ot)
    m = dict(meta or {})
    m.update({"n": int(len(idx)), "box_center_world": center.tolist(), "box_size_world": size.tolist(),
              "floor_z_world": floor_z, "yaw_deg_world": yaw_deg, "opacity_thresh": opacity_thresh})
    return ObjectSplat(means_o, scales_w[idx], quats_o, gauss["opacities"][idx], sh_o, m)


# -------------------------------------------------------------------------------- preview
def render_preview(obj: ObjectSplat, out_png, n_views: int = 4, size_px: int = 384, device="cuda",
                   elevation_deg: float = 20.0, T_wo: np.ndarray | None = None):
    """Turntable renders of the object alone (white background) -> one PNG strip.
    With T_wo the object is first posed by it and the cameras orbit the posed object, which
    exercises the same SH rotation path used by the render servers."""
    import torch
    from gsplat import rasterization
    from PIL import Image

    T_wo = np.eye(4) if T_wo is None else np.asarray(T_wo, dtype=np.float64)
    p = obj.posed(T_wo, device=device)
    lo, hi = obj.bbox()
    c_obj = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2, 1.0])
    centre = (T_wo @ c_obj)[:3]
    R_wo, s, _ = decompose_similarity(T_wo)
    radius = 2.2 * s * float(np.linalg.norm(hi - lo))
    f = size_px / (2 * np.tan(np.radians(50 / 2)))
    K = torch.tensor([[f, 0, size_px / 2], [0, f, size_px / 2], [0, 0, 1]], dtype=torch.float32, device=device)[None]
    tiles = []
    for k in range(n_views):
        az, el = 2 * np.pi * k / n_views, np.radians(elevation_deg)
        eye = centre + radius * (R_wo @ np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]))
        z = eye - centre
        z /= np.linalg.norm(z)                              # OpenGL camera: looks along -z
        x = np.cross(R_wo @ np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c2w = np.eye(4); c2w[:3, :3] = np.column_stack([x, y, z]); c2w[:3, 3] = eye
        c2w[:3, 1:3] *= -1                                  # -> OpenCV for gsplat's viewmat
        viewmat = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means=p["means"], quats=p["quats"], scales=torch.exp(p["scales_log"]),
                opacities=torch.sigmoid(p["opacities"]), colors=p["sh"], viewmats=viewmat, Ks=K,
                width=size_px, height=size_px, sh_degree=3, render_mode="RGB")
            img = img[0] + (1.0 - alpha[0])          # white background
        tiles.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    strip = np.concatenate(tiles, axis=1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_png)
    return out_png

# ------------------------------------------------------------------ import: splat PLY / mesh
def axis_vector(name: str) -> np.ndarray:
    """'X', '-Y', 'Z', ... -> unit vector."""
    sign = -1.0 if name.startswith("-") else 1.0
    idx = "XYZ".index(name.strip("+-").upper())
    v = np.zeros(3); v[idx] = sign
    return v


def frame_from_axes(forward: str, up: str) -> np.ndarray:
    """Rotation R_obj_from_src that maps a source asset's `forward` and `up` axes onto the
    object frame (+x forward, +y left, +z up)."""
    f, u = axis_vector(forward), axis_vector(up)
    if abs(np.dot(f, u)) > 1e-6:
        raise ValueError("forward and up axes must differ")
    left = np.cross(u, f)
    return np.stack([f, left, u])            # rows: object axes expressed in the source frame


def load_splat_ply(path) -> dict:
    """Standard 3DGS PLY (x y z, f_dc_0..2, f_rest_0..44 channel-major, opacity, scale_0..2,
    rot_0..3 as w x y z) -> dict in the same layout as load_nerfstudio_gaussians()."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = set(v.data.dtype.names)
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    n = len(means)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    n_rest = len([k for k in names if k.startswith("f_rest_")])
    if n_rest:
        rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], 1).astype(np.float32)
        rest = rest.reshape(n, 3, n_rest // 3).transpose(0, 2, 1)       # channel-major -> (N,K,3)
    else:
        rest = np.zeros((n, 0, 3), np.float32)
    sh = np.concatenate([dc[:, None], rest], axis=1)
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    opac = np.asarray(v["opacity"], np.float32).reshape(-1)
    return {"means": means, "scales_log": scales, "quats": quats, "opacities": opac, "sh": sh}


def gaussians_to_object(gauss: dict, R_obj_from_src: np.ndarray, scale: float = 1.0, floor: str | float = "min",
                        centre: str = "footprint", meta: dict | None = None, part_names: list | None = None) -> ObjectSplat:
    """Re-express gaussians (any frame) in the object frame: rotate by R_obj_from_src, scale
    units -> metres, put the origin on the floor ('min' = lowest gaussian, or a z value in the
    rotated metric frame) under the footprint centre. SH and covariances rotate along."""
    means = (gauss["means"] @ R_obj_from_src.T) * scale
    lo, hi = means.min(0), means.max(0)
    z0 = float(lo[2]) if floor == "min" else float(floor)
    origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z0]) if centre == "footprint" else np.array([0, 0, z0])
    means = means - origin
    q = gauss["quats"] / np.linalg.norm(gauss["quats"], axis=1, keepdims=True)
    quats = quat_mul(quat_from_matrix(R_obj_from_src)[None], q)
    sh = rotate_sh(gauss["sh"], R_obj_from_src)
    m = dict(meta or {})
    m.update({"n": int(len(means)), "scale_to_m": scale, "origin_src": (origin / scale).tolist()})
    parts = gauss.get("parts")
    if parts is not None and part_names:
        m["parts"] = wheel_parts_meta(means, parts, part_names)
    return ObjectSplat(means, gauss["scales_log"] + np.log(scale), quats, gauss["opacities"], sh, m, parts=parts)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear RGB (glTF baseColorFactor, MDL constants) -> display sRGB, the space 3DGS colours live in."""
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def gaussians_from_mesh(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray, spacing: float = 0.004,
                        overlap: float = 2.0, opacity: float = 0.99, footprint: float = 1.0, seed: int = 0,
                        face_part: np.ndarray | None = None) -> dict:
    """Surface-sampled gaussians for a triangle mesh (metres): the mesh2splat / mesh2gaussian
    recipe, vectorised. Each face gets ceil(overlap * area / spacing^2) flat gaussians lying in
    the face plane, oriented along the face's longest edge, with footprint area spacing^2 /
    overlap (times footprint^2), thickness spacing / 4, constant colour (SH degree 0).
    face_colors (M,3) in 0..1."""
    rng = np.random.default_rng(seed)
    V, F = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    ok = area > 1e-12
    normal = np.zeros_like(cross); normal[ok] = cross[ok] / (2 * area[ok, None])
    # expected count per face; stochastic rounding so a million tiny CAD faces do not each
    # force a gaussian (total stays overlap * total_area / spacing^2)
    expect = np.where(ok, overlap * area / spacing ** 2, 0.0)
    n_per = np.floor(expect).astype(int) + (rng.random(len(F)) < (expect - np.floor(expect))).astype(int)
    fidx = np.repeat(np.arange(len(F)), n_per)
    N = len(fidx)
    r1, r2 = rng.random(N), rng.random(N)
    sq = np.sqrt(r1)
    bary = np.stack([1 - sq, r2 * sq, 1 - (1 - sq) - r2 * sq], 1)
    pos = bary[:, :1] * a[fidx] + bary[:, 1:2] * b[fidx] + bary[:, 2:] * c[fidx]
    # in-plane frame: longest edge projected into the plane
    edges = np.stack([b - a, c - b, a - c], 1)                       # (M,3,3)
    L = np.linalg.norm(edges, axis=2)
    longest = edges[np.arange(len(F)), L.argmax(1)]
    t = longest - np.sum(longest * normal, 1, keepdims=True) * normal
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    bt = np.cross(normal, t)
    Rf = np.stack([t, bt, normal], 2)                                 # columns: tangent, bitangent, normal
    # anisotropy from the face shape (as in mesh2gaussian): rf = L_max^2 / (4 A)
    rf = np.clip(L.max(1) ** 2 / (4 * np.maximum(area, 1e-12)), 1e-3, 1e3)
    s1 = np.sqrt(rf / (np.pi * overlap)) * spacing
    s2 = spacing / np.sqrt(np.pi * overlap * rf)
    s1, s2 = np.clip(s1, 0.3 * spacing, 6 * spacing) * footprint, np.clip(s2, 0.3 * spacing, 6 * spacing) * footprint
    scales = np.stack([s1, s2, np.full_like(s1, spacing / 4)], 1)[fidx]
    quats = np.stack([quat_from_matrix(Rf[i]) for i in range(len(F))])[fidx]
    rgb = np.clip(np.asarray(face_colors, np.float64), 0, 1)[fidx]
    sh = np.zeros((N, N_SH, 3), np.float32)
    sh[:, 0] = (rgb - 0.5) / SH_C0
    out = {"means": pos.astype(np.float32), "scales_log": np.log(scales).astype(np.float32),
           "quats": quats.astype(np.float32), "opacities": np.full(N, _logit(opacity), np.float32), "sh": sh}
    if face_part is not None:
        out["parts"] = np.asarray(face_part, np.int16)[fidx]
    return out


# ------------------------------------------------------------------ parts, shadow, motion
F1TENTH_PART_RULES = (
    ("wheel_fr", ("wheel_1_", "fr_wheel_hub", "wheel_shaft_1_solid1_fr")),
    ("wheel_rr", ("wheel_2_", "rear_wheel_hub_2", "wheel_shaft_2")),
    ("wheel_fl", ("wheel_3_", "fl_wheel_hub")),
    ("wheel_rl", ("wheel_4_", "rear_wheel_hub_1", "wheel_shaft_1_solid1")),
)


def classify_part(mesh_name: str) -> str:
    """Mesh (prim) name of the AutoDRIVE F1TENTH CAD -> part name ("body" or wheel_xx).
    Wheel_1 front right, Wheel_2 rear right, Wheel_3 front left, Wheel_4 rear left (asset frame:
    +X forward, +Z right)."""
    n = mesh_name.lower()
    for part, keys in F1TENTH_PART_RULES:
        if any(k in n for k in keys):
            return part
    return "body"


def wheel_parts_meta(means: np.ndarray, parts: np.ndarray, names: list, axis=(0.0, 1.0, 0.0)) -> list:
    """Part table for a car: wheel centre = centre of its gaussians, radius = half of the
    vertical extent, front wheels steerable. `names` indexes the part ids."""
    table = []
    for i, name in enumerate(names):
        entry = {"name": name, "kind": "wheel" if name.startswith("wheel") else "body"}
        if entry["kind"] == "wheel":
            m = parts == i
            if m.any():
                P = means[m]
                lo, hi = P.min(0), P.max(0)
                entry.update({"centre": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
                              "radius": float((hi[2] - lo[2]) / 2), "axis": list(axis), "steer": name.endswith(("fl", "fr"))})
        table.append(entry)
    return table


def contact_shadow(footprint_x: float, footprint_y: float, darkness: float = 0.55, spacing: float = 0.012,
                   height: float = 0.004, seed: int = 0) -> ObjectSplat:
    """A soft dark patch on the floor under a car (ambient occlusion of a diffusely lit hall):
    flat gaussians on a rounded rectangle footprint, opacity fading from `darkness` at the
    centre to zero at the edge, black colour. Visual only (parts flag lidar=false)."""
    rng = np.random.default_rng(seed)
    hx, hy = footprint_x / 2, footprint_y / 2
    nx, ny = int(np.ceil(2 * hx / spacing)), int(np.ceil(2 * hy / spacing))
    gx, gy = np.meshgrid(np.linspace(-hx, hx, nx), np.linspace(-hy, hy, ny))
    pts = np.c_[gx.ravel(), gy.ravel()] + rng.normal(scale=spacing * 0.15, size=(nx * ny, 2))
    # normalised radial distance with rounded corners (p-norm 4)
    r = (np.abs(pts[:, 0] / hx) ** 4 + np.abs(pts[:, 1] / hy) ** 4) ** 0.25
    keep = r < 1.25
    pts, r = pts[keep], r[keep]
    alpha = darkness * np.clip(1.0 - (r / 1.25) ** 2, 0, 1) ** 1.5
    alpha = np.clip(alpha, 0.01, 0.95)
    n = len(pts)
    means = np.c_[pts, np.full(n, height)]
    scales = np.log(np.tile([spacing * 0.9, spacing * 0.9, 0.001], (n, 1)))
    quats = np.tile([1.0, 0, 0, 0], (n, 1))
    sh = np.zeros((n, N_SH, 3), np.float32)
    sh[:, 0] = (0.02 - 0.5) / SH_C0
    return ObjectSplat(means, scales, quats, np.log(alpha / (1 - alpha)), sh, {"name": "shadow"})


def driving_state_from_path(P: np.ndarray, times: np.ndarray, wheelbase: float = 0.33, smooth: int = 5) -> dict:
    """Speed, longitudinal/lateral acceleration and Ackermann steering angle along a sampled
    path (positions P (N,3) in a floor-aligned frame, times (N,)). Used to animate wheels and
    body attitude for an object driven along a recorded path."""
    from scipy.ndimage import uniform_filter1d
    dt = np.gradient(times)
    v_vec = np.gradient(P, axis=0) / dt[:, None]
    v = np.linalg.norm(v_vec[:, :2], axis=1)
    v = uniform_filter1d(v, smooth)
    heading = np.unwrap(np.arctan2(v_vec[:, 1], v_vec[:, 0]))
    heading = uniform_filter1d(heading, smooth)
    yaw_rate = np.gradient(heading) / dt
    a_x = uniform_filter1d(np.gradient(v) / dt, smooth)
    a_y = v * yaw_rate
    curvature = np.where(v > 0.05, yaw_rate / np.maximum(v, 0.05), 0.0)
    steer = np.arctan(wheelbase * curvature)
    return {"v": v, "a_x": a_x, "a_y": a_y, "yaw_rate": yaw_rate, "steer": steer, "heading": heading}


def attitude_from_accel(a_x: float, a_y: float, pitch_gain: float = np.radians(1.2), roll_gain: float = np.radians(1.5),
                        max_deg: float = 4.0) -> tuple:
    """Body pitch (nose down positive) and roll (right side down positive) of a soft-sprung
    RC car under acceleration: braking lifts the rear, cornering leans outwards. Gains in
    rad per m/s^2, clipped."""
    lim = np.radians(max_deg)
    pitch = float(np.clip(-pitch_gain * a_x, -lim, lim))
    roll = float(np.clip(roll_gain * a_y, -lim, lim))
    return pitch, roll
