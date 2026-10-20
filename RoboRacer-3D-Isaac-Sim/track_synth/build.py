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
HOSE_FLATTEN = .72     # soft hoses sag: vertical / horizontal radius
HALL_MARGIN = (7., 7.)  # hall extends this far past the map bounds (x, y)
HALL_HEIGHT = 9.
# Camera intrinsics and mount rotation from configs/racing_142014.json (OAK-D Pro W on Car02).
CAMERA = dict(width=640, height=400, fx=283.4349068198737, fy=291.87822087105116,
              cx=320.97935, cy=194.30535)
# Height: reference/vehicle/vehicle_parameters.txt §6b corrects the camera to z=0.200 m above
# base_link (the 0.066 m in the racing config is the uncorrected value); base_link sits at axle
# height (~0.05 m tyre radius), so the optical centre is ~0.25 m above the floor.
CAMERA_MOUNT = dict(translation=[0.29267, 0.06987, 0.200],
                    quaternion=[-0.4743109522221859, 0.4989799497372533, -0.5079499488336963, 0.5177209478494539])
BASE_Z = .05


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
    from pxr import UsdGeom, UsdShade, Vt, Sdf
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


def _rod(stage, path, p0, p1, radius, material=None):
    from pxr import UsdGeom, UsdShade, Gf
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0; length = float(np.linalg.norm(axis))
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius); cyl.CreateHeightAttr(length); cyl.CreateAxisAttr('Z')
    matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(axis / length))))
    matrix.SetTranslateOnly(Gf.Vec3d(*((p0 + p1) / 2)))
    UsdGeom.Xformable(cyl).AddTransformOp().Set(matrix)
    if material is not None:
        UsdShade.MaterialBindingAPI(cyl).Bind(material)


def _cylinder(stage, path, center, radius, height, material=None):
    from pxr import UsdGeom, UsdShade, Gf
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius); cyl.CreateHeightAttr(height); cyl.CreateAxisAttr('Z')
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    if material is not None:
        UsdShade.MaterialBindingAPI(cyl).Bind(material)


def build_hall(stage, geometry, materials, rng, light_scale=1.):
    """Placeholder industrial hall: floor, walls, roof, beams, columns and clutter outside the track."""
    from pxr import UsdGeom, UsdLux, Gf, Sdf
    x0, y0, x1, y1 = geometry.bounds
    hx, hy = HALL_MARGIN
    X0, Y0, X1, Y1 = x0 - hx, y0 - hy, x1 + hx, y1 + hy
    H = HALL_HEIGHT
    UsdGeom.Xform.Define(stage, '/World/Hall')
    _mesh(stage, '/World/Hall/Floor', _quad(X0, Y0, X1, Y1, 0., ((X1 - X0) / 4., (Y1 - Y0) / 4.)), materials['floor'])
    _mesh(stage, '/World/Hall/Roof', _quad(X0, Y0, X1, Y1, H, ((X1 - X0) / 3., (Y1 - Y0) / 3.)), materials['roof'])
    # roof faces down; it must not block the dome light, which stands in for skylights + bounce light
    roof = stage.GetPrimAtPath('/World/Hall/Roof')
    UsdGeom.Mesh(roof).CreateNormalsAttr([(0, 0, -1)] * 4); UsdGeom.Mesh(roof).CreateDoubleSidedAttr(True)
    roof.CreateAttribute('primvars:doNotCastShadows', Sdf.ValueTypeNames.Bool).Set(True)
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
    # clutter: outside the free band and clear of the hoses, but close enough to be seen
    tree = cKDTree(np.vstack(geometry.walls)); centre_tree = cKDTree(geometry.centerline)
    objects = [(materials['blue'], (1.0, .6, 2.0)), (materials['wood'], (1.2, .8, .9)),
               (materials['dark'], (1.6, 1.0, 1.4)), (materials['grey'], (.9, .7, 1.8)),
               (materials['wood'], (.6, .4, .5)), (materials['grey'], (2.0, .8, 1.0))]
    def clear(x, y, radius):
        d_centre, i = centre_tree.query([x, y])
        return (tree.query([x, y])[0] > radius + HOSE_RADIUS + .4 and
                d_centre > geometry.half_width[i] + HOSE_RADIUS + radius + .4)
    placed = 0
    for _ in range(3000):
        if placed >= 60:
            break
        x = rng.uniform(X0 + 1, X1 - 1); y = rng.uniform(Y0 + 1, Y1 - 1)
        material, size = objects[rng.integers(len(objects))]
        if not clear(x, y, np.hypot(size[0], size[1]) / 2):
            continue
        _box(stage, f'/World/Hall/Clutter{placed}', (x, y, size[2] / 2), size, rng.uniform(0, np.pi), material)
        placed += 1
    # mocap tripods: three legs + mast + camera head, just outside the hoses
    placed = 0
    for _ in range(3000):
        if placed >= 16:
            break
        x = rng.uniform(x0, x1); y = rng.uniform(y0, y1)
        if not clear(x, y, .6) or tree.query([x, y])[0] > 2.5:
            continue
        top = rng.uniform(1.8, 3.0); base = f'/World/Hall/Tripod{placed}'
        UsdGeom.Xform.Define(stage, base)
        for leg, angle in enumerate(np.linspace(0, 2 * np.pi, 3, endpoint=False) + rng.uniform(0, 2)):
            _rod(stage, f'{base}/Leg{leg}', (x + .55 * np.cos(angle), y + .55 * np.sin(angle), 0.),
                 (x, y, 1.1), .012, materials['dark'])
        _rod(stage, f'{base}/Mast', (x, y, 1.0), (x, y, top), .018, materials['dark'])
        _box(stage, f'{base}/Head', (x, y, top + .06), (.12, .1, .1), 0., materials['dark'])
        placed += 1
    # lighting: bright dome (skylights + bounce), ceiling panels, daylight windows high on the walls
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/Ambient')
    dome.CreateIntensityAttr(1000. * light_scale); dome.CreateColorAttr((.95, .97, 1.))
    for i, y in enumerate(np.linspace(Y0 + 6., Y1 - 6., 5)):
        for j, x in enumerate(np.linspace(X0 + 8., X1 - 8., 7)):
            light = UsdLux.RectLight.Define(stage, f'/World/Lighting/Panel{i}_{j}')
            light.CreateWidthAttr(2.4); light.CreateHeightAttr(.4)
            light.CreateIntensityAttr(20000. * light_scale); light.CreateColorAttr((1., .97, .92))
            # RectLight already emits along local -Z (down): no rotation
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, H - .15))
    for w, (p0, p1, inward) in enumerate(walls):
        p0, p1, inward = np.array(p0), np.array(p1), np.array(inward, float)
        length = np.linalg.norm(p1 - p0); along = (p1 - p0) / length
        heading = np.degrees(np.arctan2(inward[1], inward[0])) - 90.
        for k, t in enumerate(np.arange(3., length - 2., 5.)):
            centre = p0 + along * t + inward * .05
            light = UsdLux.RectLight.Define(stage, f'/World/Lighting/Window{w}_{k}')
            light.CreateWidthAttr(2.0); light.CreateHeightAttr(1.6)
            light.CreateIntensityAttr(8000. * light_scale); light.CreateColorAttr((.88, .93, 1.))
            xf = UsdGeom.Xformable(light)
            xf.AddTranslateOp().Set(Gf.Vec3d(centre[0], centre[1], H - 2.2))
            xf.AddRotateZOp().Set(float(heading)); xf.AddRotateXOp().Set(90.)   # -Z -> inward


def build(map_yaml, out_dir, seed=0, sides=24, light_scale=1., hose_layers=HOSE_LAYERS):
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
        floor=M('Floor', color=(.5, .5, .48), roughness=.4, texture='./textures/floor.png', uv_scale=(1, 1)),
        wall=M('Wall', color=(.82, .81, .78), roughness=.9),
        roof=M('Roof', color=(.62, .64, .66), roughness=.5, metallic=.5),
        steel=M('Steel', color=(.12, .12, .13), roughness=.55, metallic=.6),
        yellow=M('Yellow', color=(.85, .62, .05), roughness=.5),
        blue=M('Blue', color=(.04, .16, .55), roughness=.45),
        wood=M('Wood', color=(.55, .42, .26), roughness=.8),
        dark=M('Dark', color=(.05, .05, .06), roughness=.6),
        grey=M('Grey', color=(.55, .56, .58), roughness=.6))
    hose = M('Hose', color=(.86, .86, .83), roughness=.6, texture='./textures/hose_albedo.png',
             normal='./textures/hose_normal.png')
    build_hall(stage, geometry, materials, rng, light_scale)
    UsdGeom.Xform.Define(stage, '/World/Track')
    for i, wall in enumerate(geometry.walls):
        for layer, mesh in enumerate(stacked_hose_layers(wall, rng, HOSE_RADIUS, HOSE_FLATTEN, layers=hose_layers, sides=sides)):
            _mesh(stage, f'/World/Track/Hose{i}_L{layer}', mesh, hose)
    stage.GetRootLayer().Save()

    thin = lambda p, step: np.asarray(p)[::step].tolist()
    scenario = dict(
        name=f'synthetic {Path(map_yaml).stem}', usd='track.usd', map=str(map_yaml),
        camera=CAMERA, sensors={'camera': CAMERA_MOUNT}, base_z=BASE_Z,
        rates={'dynamics': 50, 'camera': 20},
        reference_path=thin(geometry.centerline, 2), reference_spacing_m=.2,
        half_width=thin(geometry.half_width, 2),
        outer_boundary=thin(geometry.outer, 2), inner_boundaries=[thin(h, 2) for h in geometry.holes],
        hose=dict(radius_m=HOSE_RADIUS, layers=hose_layers, flatten=HOSE_FLATTEN),
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
    parser.add_argument('--hose-layers', type=int, default=HOSE_LAYERS, choices=(1, 2))
    args = parser.parse_args()
    geometry, usd = build(args.map, args.out, args.seed, light_scale=args.light_scale, hose_layers=args.hose_layers)
    length = np.linalg.norm(np.diff(np.vstack([geometry.centerline, geometry.centerline[:1]]), axis=0), axis=1).sum()
    print(f'wrote {usd} | centerline {length:.1f} m | {len(geometry.walls)} hose segments')


if __name__ == '__main__':
    main()
