"""Build a synthetic hall + hose-bounded track USD scene and a scenario JSON.

    python -m track_synth.build --map /path/IROS2026.yaml --out outputs/iros2026

Runs with plain `usd-core` (no Isaac). The hall is a labelled placeholder, not a scan.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .extract import extract
from .meshes import (stacked_hose_layers, corrugation_normal_map, hose_albedo, concrete_albedo)

HOSE_RADIUS = .15      # 30 cm diameter
HOSE_LAYERS = 2
HALL_MARGIN = (7., 7.)  # hall extends this far past the map bounds (x, y)
HALL_HEIGHT = 9.
# Camera and mount copied from configs/racing_142014.json (OAK-D Pro W on Car02).
CAMERA = dict(width=640, height=400, fx=283.4349068198737, fy=291.87822087105116,
              cx=320.97935, cy=194.30535)
CAMERA_MOUNT = dict(translation=[0.29267, 0.06987, 0.06559],
                    quaternion=[-0.4743109522221859, 0.4989799497372533, -0.5079499488336963, 0.5177209478494539])


def _material(stage, path, color, roughness=.6, metallic=0., texture=None, normal=None, uv_scale=None, emissive=None):
    from pxr import UsdShade, Sdf, Gf
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    if emissive:
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))

    def texture_node(name, file, colorspace, output):
        reader = UsdShade.Shader.Define(stage, f'{path}/{name}_uv')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex = UsdShade.Shader.Define(stage, f'{path}/{name}')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(colorspace)
        tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        if uv_scale:
            xform = UsdShade.Shader.Define(stage, f'{path}/{name}_xf')
            xform.CreateIdAttr('UsdTransform2d')
            xform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            xform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(xform.ConnectableAPI(), 'result')
        else:
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        return tex.CreateOutput(output, Sdf.ValueTypeNames.Float3 if output == 'rgb' else Sdf.ValueTypeNames.Float)

    if texture:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture_node('albedo', texture, 'sRGB', 'rgb'))
    else:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if normal:
        shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            texture_node('normal', normal, 'raw', 'rgb'))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _mesh(stage, path, data, material=None):
    from pxr import UsdGeom, UsdShade, Vt, Gf, Sdf
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(data['points'], dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(data['counts'], dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(data['indices'], dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr('none')
    if 'normals' in data:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(data['normals'], dtype=np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if 'uvs' in data:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        primvar.Set(Vt.Vec2fArray.FromNumpy(np.asarray(data['uvs'], dtype=np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
    return mesh


def _quad(x0, y0, x1, y1, z, uv_repeat):
    return dict(points=np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]), counts=np.array([4]),
                indices=np.array([0, 1, 2, 3]), normals=np.tile([0, 0, 1.], (4, 1)),
                uvs=np.array([[0, 0], [uv_repeat[0], 0], [uv_repeat[0], uv_repeat[1]], [0, uv_repeat[1]]]))


def _wall(p0, p1, height, inward):
    (x0, y0), (x1, y1) = p0, p1
    pts = np.array([[x0, y0, 0], [x1, y1, 0], [x1, y1, height], [x0, y0, height]])
    return dict(points=pts, counts=np.array([4]), indices=np.array([0, 1, 2, 3]),
                normals=np.tile([*inward, 0.], (4, 1)), uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


def _box(stage, path, center, size, yaw=0., material=None):
    from pxr import UsdGeom, UsdShade, Gf
    cube = UsdGeom.Cube.Define(stage, path); cube.CreateSizeAttr(1.)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center)); xf.AddRotateZOp().Set(float(np.degrees(yaw)))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if material is not None:
        UsdShade.MaterialBindingAPI(cube).Bind(material)


def _cylinder(stage, path, center, radius, height, material=None):
    from pxr import UsdGeom, UsdShade, Gf
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius); cyl.CreateHeightAttr(height); cyl.CreateAxisAttr('Z')
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    if material is not None:
        UsdShade.MaterialBindingAPI(cyl).Bind(material)


def build_hall(stage, geometry, materials, rng, light_scale=1.):
    """Placeholder industrial hall: floor, walls, roof, beams, columns and clutter outside the track."""
    from pxr import UsdGeom, UsdLux, Gf
    x0, y0, x1, y1 = geometry.bounds
    hx, hy = HALL_MARGIN
    X0, Y0, X1, Y1 = x0 - hx, y0 - hy, x1 + hx, y1 + hy
    H = HALL_HEIGHT
    UsdGeom.Xform.Define(stage, '/World/Hall')
    _mesh(stage, '/World/Hall/Floor', _quad(X0, Y0, X1, Y1, 0., ((X1 - X0) / 4., (Y1 - Y0) / 4.)), materials['floor'])
    _mesh(stage, '/World/Hall/Roof', _quad(X0, Y0, X1, Y1, H, ((X1 - X0) / 3., (Y1 - Y0) / 3.)), materials['roof'])
    # roof faces down
    roof = stage.GetPrimAtPath('/World/Hall/Roof')
    UsdGeom.Mesh(roof).CreateNormalsAttr([(0, 0, -1)] * 4); UsdGeom.Mesh(roof).CreateDoubleSidedAttr(True)
    walls = [((X0, Y0), (X1, Y0), (0, 1)), ((X1, Y0), (X1, Y1), (-1, 0)),
             ((X1, Y1), (X0, Y1), (0, -1)), ((X0, Y1), (X0, Y0), (1, 0))]
    for i, (p0, p1, inward) in enumerate(walls):
        _mesh(stage, f'/World/Hall/Wall{i}', _wall(p0, p1, H, inward), materials['wall'])
    # steel: roof beams across x every 6 m, columns along the walls, a yellow crane girder
    for i, y in enumerate(np.arange(Y0 + 3., Y1, 6.)):
        _box(stage, f'/World/Hall/Beam{i}', ((X0 + X1) / 2, y, H - .5), (X1 - X0, .3, .5), 0., materials['steel'])
    for i, x in enumerate(np.arange(X0 + 4., X1, 10.)):
        for j, y in enumerate((Y0 + .4, Y1 - .4)):
            _box(stage, f'/World/Hall/Column{i}_{j}', (x, y, H / 2), (.5, .5, H), 0., materials['steel'])
    _box(stage, '/World/Hall/Crane', ((X0 + X1) / 2, Y1 - 4., H - 1.6), (X1 - X0, .5, .5), 0., materials['yellow'])
    # clutter only where it cannot intersect the track or its hoses
    tree = cKDTree(np.vstack(geometry.walls)); centre_tree = cKDTree(geometry.centerline)
    objects = [('cabinet', materials['blue'], (1.0, .6, 2.0)), ('crate', materials['wood'], (1.2, .8, .9)),
               ('machine', materials['dark'], (1.6, 1.0, 1.4)), ('cabinet', materials['grey'], (.9, .7, 1.8))]
    placed = 0
    for attempt in range(600):
        if placed >= 28:
            break
        x = rng.uniform(X0 + 2, X1 - 2); y = rng.uniform(Y0 + 2, Y1 - 2)
        name, material, size = objects[rng.integers(len(objects))]
        radius = np.hypot(size[0], size[1]) / 2 + .4
        if tree.query([x, y])[0] < radius + HOSE_RADIUS + 1.6 or centre_tree.query([x, y])[0] < radius + 2.:
            continue
        _box(stage, f'/World/Hall/Clutter{placed}', (x, y, size[2] / 2), size, rng.uniform(0, np.pi), material)
        placed += 1
    # mocap tripod stand-ins (thin poles) around the perimeter of the track
    for k, angle in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False)):
        cx, cy = (x0 + x1) / 2 + (x1 - x0) / 2 * 1.08 * np.cos(angle), (y0 + y1) / 2 + (y1 - y0) / 2 * 1.2 * np.sin(angle)
        if tree.query([cx, cy])[0] < 1.6:
            continue
        _cylinder(stage, f'/World/Hall/Tripod{k}', (cx, cy, 1.6), .02, 3.2, materials['dark'])
    # lighting: dim ambient dome + rows of ceiling panels
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/Ambient')
    dome.CreateIntensityAttr(100. * light_scale); dome.CreateColorAttr((.92, .95, 1.))
    for i, y in enumerate(np.linspace(Y0 + 6., Y1 - 6., 5)):
        for j, x in enumerate(np.linspace(X0 + 8., X1 - 8., 7)):
            light = UsdLux.RectLight.Define(stage, f'/World/Lighting/Panel{i}_{j}')
            light.CreateWidthAttr(2.4); light.CreateHeightAttr(.4)
            light.CreateIntensityAttr(6000. * light_scale); light.CreateColorAttr((1., .97, .92))
            # RectLight already emits along local -Z (down): no rotation
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, H - .15))


def build(map_yaml, out_dir, seed=0, sides=24, light_scale=1.):
    from pxr import Usd, UsdGeom
    out = Path(out_dir); (out / 'textures').mkdir(parents=True, exist_ok=True)
    geometry = extract(map_yaml)
    rng = np.random.default_rng(seed)
    Image.fromarray(concrete_albedo(seed=seed + 1)).save(out / 'textures/floor.png')
    Image.fromarray(hose_albedo(seed=seed)).save(out / 'textures/hose_albedo.png')
    Image.fromarray(corrugation_normal_map()).save(out / 'textures/hose_normal.png')
    usd_path = out / 'track.usd'
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    M = lambda name, **kw: _material(stage, f'/World/Looks/{name}', **kw)
    materials = dict(
        floor=M('Floor', color=(.5, .5, .48), roughness=.35, texture='./textures/floor.png', uv_scale=(1, 1)),
        wall=M('Wall', color=(.78, .76, .70), roughness=.9),
        roof=M('Roof', color=(.62, .64, .66), roughness=.5, metallic=.5),
        steel=M('Steel', color=(.12, .12, .13), roughness=.55, metallic=.6),
        yellow=M('Yellow', color=(.85, .62, .05), roughness=.5),
        blue=M('Blue', color=(.04, .16, .55), roughness=.45),
        wood=M('Wood', color=(.55, .42, .26), roughness=.8),
        dark=M('Dark', color=(.05, .05, .06), roughness=.6),
        grey=M('Grey', color=(.55, .56, .58), roughness=.6))
    hose = M('Hose', color=(.86, .86, .83), roughness=.75, texture='./textures/hose_albedo.png',
             normal='./textures/hose_normal.png')
    build_hall(stage, geometry, materials, rng, light_scale)
    UsdGeom.Xform.Define(stage, '/World/Track')
    for i, wall in enumerate(geometry.walls):
        for layer, mesh in enumerate(stacked_hose_layers(wall, rng, HOSE_RADIUS, layers=HOSE_LAYERS, sides=sides)):
            _mesh(stage, f'/World/Track/Hose{i}_L{layer}', mesh, hose)
    stage.GetRootLayer().Save()

    thin = lambda p, step: np.asarray(p)[::step].tolist()
    scenario = dict(
        name=f'synthetic {Path(map_yaml).stem}', usd='track.usd', map=str(map_yaml),
        camera=CAMERA, sensors={'camera': CAMERA_MOUNT},
        rates={'dynamics': 50, 'camera': 20},
        reference_path=thin(geometry.centerline, 2), reference_spacing_m=.2,
        half_width=thin(geometry.half_width, 2),
        outer_boundary=thin(geometry.outer, 2), inner_boundaries=[thin(h, 2) for h in geometry.holes],
        hose=dict(radius_m=HOSE_RADIUS, layers=HOSE_LAYERS),
        provenance=dict(hall='placeholder geometry, not a scan', walls_from='skeleton of map wall lines',
                        seed=seed))
    (out / 'scenario.json').write_text(json.dumps(scenario))
    return geometry, usd_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='ROS map YAML (image: PNG)')
    parser.add_argument('--out', default='outputs/iros2026')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--light-scale', type=float, default=1., help='multiplies all light intensities (raise if frames are dark)')
    args = parser.parse_args()
    geometry, usd = build(args.map, args.out, args.seed, light_scale=args.light_scale)
    length = np.linalg.norm(np.diff(np.vstack([geometry.centerline, geometry.centerline[:1]]), axis=0), axis=1).sum()
    print(f'wrote {usd} | centerline {length:.1f} m | {len(geometry.walls)} hose segments')


if __name__ == '__main__':
    main()
