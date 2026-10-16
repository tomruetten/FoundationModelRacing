#!/usr/bin/env python3
"""Build a reference-informed, shaded F1TENTH Gaussian asset from AutoDRIVE CAD.

Uses named components and authored smooth normals, bounded splat footprints,
ray-traced contact occlusion, and modeled cables. No downloaded photograph is
used as texture. Run in roboracer; --preview requires CUDA.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'simulation/isaac_sim'))
from object_splats import ObjectSplat, SH_C0, render_preview, classify_part, wheel_parts_meta, contact_shadow

PALETTE = {
    'rubber': [0.105, .115, .13], 'plastic': [.16, .18, .21],
    'silver': [.72, .76, .8], 'spring': [.89, .9, .88],
    # Keep the component names, but use restrained black housings and cables.
    'red': [.10, .105, .11], 'blue': [.11, .12, .13],
    'orange': [.12, .125, .13], 'pcb': [.065, .10, .075],
    'gold': [.48, .45, .37], 'green': [.10, .12, .105],
    'glass': [.035, .095, .14], 'deck': [.26, .29, .32],
    'white': [.91, .92, .89], 'battery': [.105, .11, .12],
}


def material(name, original):
    n = name.lower()
    if n.startswith('wheel_'):
        part = int(n.rsplit('solid', 1)[1])
        return {1:'rubber',2:'plastic',3:'silver',4:'red'}.get(part, 'silver')
    if 'spring' in n: return 'spring'
    if 'hokuyo' in n:
        return {'solid1':'orange','solid2':'plastic','solid3':'silver','solid4':'glass'}.get(n.rsplit('_',1)[1], 'plastic')
    if 'screw_terminal' in n: return 'green'
    if 'standoff' in n: return 'gold'
    if 'bolt' in n or 'pin_' in n or 'turnbuckle' in n: return 'silver'
    if 'battery' in n: return 'battery'
    if 'platform_deck' in n: return 'deck'
    if n == 'chassis_1_solid1': return 'deck'
    if 'top_mount' in n or 'lower_mount' in n: return 'blue'
    if 'logo_silk' in n: return 'white'
    if 'heatsink' in n or 'fan' in n: return 'plastic'
    if 'pcb_1_pcb' in n or '142_13509_1000_a01_sku0000_stp' in n or 'module_board' in n: return 'pcb'
    if 'antenna' in n: return 'plastic'
    if 'bldc_motor' in n or 'vesc_1_solid1' == n: return 'blue'
    if 'jetson' in n and np.mean(original) > .4: return 'silver'
    if np.max(original)-np.min(original) > .18: return None
    if 'shock_tower' in n: return 'blue'
    return 'plastic' if np.mean(original) < .35 else 'silver'


def load_components(path):
    from pxr import Usd, UsdGeom, UsdShade
    stage = Usd.Stage.Open(str(path))
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    # Source is Y up, +X forward. Object is +X forward, +Y left, +Z up.
    basis = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    components = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh = UsdGeom.Mesh(prim)
        # The imported CAD rear tower forms a large solid wall/wing. Replace it
        # with the open rear body support seen on the physical reference cars.
        if prim.GetName() == 'Rear_Shock_Tower_1_solid1': continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible': continue
        vertices = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if not len(vertices): continue
        if not np.all(counts == 3): raise ValueError('Expected triangulated source CAD')
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        v = ((np.c_[vertices,np.ones(len(vertices))] @ xf)[:,:3]*scale) @ basis.T
        triangles = v[faces]
        raw = np.asarray(mesh.GetNormalsAttr().Get(), dtype=float)
        if mesh.GetNormalsInterpolation() == 'faceVarying' and raw.size == len(faces)*9:
            normals = (raw.reshape(-1,3) @ np.linalg.inv(xf[:3,:3]).T) @ basis.T
            normals = normals.reshape(-1,3,3)
        else:
            normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
            normals = np.repeat(normals[:,None,:],3,axis=1)
        normals /= np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-12)
        original = np.array([.3,.3,.3])
        mat,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            for shader in Usd.PrimRange(mat.GetPrim()):
                if shader.IsA(UsdShade.Shader):
                    inp = UsdShade.Shader(shader).GetInput('diffuse_color_constant')
                    if inp and inp.Get() is not None: original = np.array(inp.Get())
        kind = material(prim.GetName(), original)
        color = np.array(PALETTE[kind]) if kind else original
        if kind is None:
            # Unclassified coloured CAD materials should not restore bright accents.
            color = np.full(3, min(float(np.mean(original)), .18))
        components.append((prim.GetName(),triangles,normals,color,kind or 'original'))
    wheel = np.concatenate([t.reshape(-1,3) for name,t,*_ in components if name.startswith('Wheel_')])
    origin = (wheel.min(0)+wheel.max(0))/2; origin[2] = wheel[:,2].min()
    return [(name,t-origin,n,c,k) for name,t,n,c,k in components], origin


def tube(points, radius=.0022, rings=70, sides=10):
    curve = CubicSpline(np.linspace(0,1,len(points)),np.array(points))
    t = np.linspace(0,1,rings); centers=curve(t); tangents=curve(t,1)
    tangents /= np.linalg.norm(tangents,axis=1,keepdims=True)
    first = np.cross(tangents,np.array([0.,1.,0.]))
    bad = np.linalg.norm(first,axis=1)<.01
    first[bad] = np.cross(tangents[bad],np.array([1.,0.,0.]))
    first /= np.linalg.norm(first,axis=1,keepdims=True)
    second=np.cross(tangents,first)
    a=np.arange(sides)*2*np.pi/sides
    normals=np.cos(a)[None,:,None]*first[:,None]+np.sin(a)[None,:,None]*second[:,None]
    v=(centers[:,None]+radius*normals).reshape(-1,3); ns=normals.reshape(-1,3)
    faces=[]
    for i in range(rings-1):
        for j in range(sides):
            x=i*sides+j; y=i*sides+(j+1)%sides
            faces.extend([[x,y,x+sides],[y,y+sides,x+sides]])
    f=np.array(faces)
    return v[f],ns[f]


def details(components, origin):
    # These routings join the source CAD's electronics locations; mm-scale wires
    # are new geometry, so camera and LiDAR see the same additions.
    paths = [
        ('power_red','red',[[-.14,.055,.065],[-.20,.081,.08],[-.185,.079,.151],[-.13,.021,.135]]),
        ('power_black','rubber',[[-.14,.045,.065],[-.195,.071,.08],[-.18,.069,.151],[-.128,.012,.135]]),
        ('usb_blue','blue',[[-.04,.067,.121],[.01,.09,.143],[.04,.065,.16],[.028,.038,.132]]),
        ('usb_black','rubber',[[-.062,.07,.126],[-.08,.105,.07],[-.19,.09,.09],[-.15,.045,.131]]),
        ('motor_red','red',[[-.15,.01,.058],[-.2,-.005,.082],[-.18,.021,.119],[-.14,.028,.118]]),
    ]
    # Coordinates above: source x, positive object-left = -source z, source height.
    for name,kind,pts in paths:
        tri,norm=tube(np.array(pts)-origin)
        components.append((name,tri,norm,np.array(PALETTE[kind]),kind))
    for side in [-1,1]:
        pts=np.array([[-.238,.038+side*.054,.068],[-.242,.038+side*.047,.133],
                      [-.242,.038,.14]])-origin
        tri,norm=tube(pts,radius=.003,rings=30)
        components.append(('rear_support',tri,norm,np.array(PALETTE['plastic']),'plastic'))
    import trimesh

    def add_mesh(name,mesh,kind):
        tri=np.asarray(mesh.triangles)
        norm=np.asarray(mesh.vertex_normals)[mesh.faces]
        components.append((name,tri,norm,np.array(PALETTE[kind]),kind))

    def box(name,extent,center,kind):
        m=trimesh.creation.box(extents=extent);m.apply_translation(np.array(center)-origin)
        # Boxes must retain sharp corner normals, not smoothed cube normals.
        components.append((name,np.asarray(m.triangles),
                           np.repeat(m.face_normals[:,None,:],3,axis=1),np.array(PALETTE[kind]),kind))
    box('stereo_camera', [.02,.077,.021], [.073,.038,.11], 'silver')
    box('camera_face', [.001,.067,.015], [.0835,.038,.11], 'plastic')
    for y in [-.024,0,.024]:
        m=trimesh.creation.cylinder(radius=.0045,height=.0015,sections=24)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        m.apply_translation(np.array([.085,.038+y,.11])-origin)
        add_mesh('camera_lens',m,'glass')
    box('rear_number_plate',[.002,.062,.028],[-.251,.038,.132],'white')
    box('battery_connector',[.013,.02,.01],[-.172,.079,.14],'orange')


def sample_components(components, spacing, seed):
    rng=np.random.default_rng(seed); sampled=[]; full=[]; stats={}; part_ids=[]; part_names=['body']
    for name,tri,norm,color,kind in components:
        full.append(tri)
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        expected=2*area/spacing**2
        counts=np.floor(expected).astype(int)+(rng.random(len(area))<expected%1)
        ids=np.repeat(np.arange(len(area)),counts)
        if not len(ids): continue
        a=np.sqrt(rng.random(len(ids))); b=rng.random(len(ids))
        weights=np.c_[1-a,a*(1-b),a*b]
        points=np.einsum('ni,nij->nj',weights,tri[ids])
        normal=np.einsum('ni,nij->nj',weights,norm[ids])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        colors=np.tile(color,(len(ids),1))
        # Surface variation stays small; preserve broad painted/material regions.
        grain=1+.025*np.sin(points[:,0]*970)*np.sin(points[:,1]*1130+points[:,2]*713)
        colors*=grain[:,None]
        if kind=='battery':
            stripe=(points[:,0]>.015)&(points[:,0]<.065)
            colors[stripe]=[.88,.9,.9]
        if name=='rear_number_plate':
            from PIL import Image,ImageDraw,ImageFont
            label=Image.new('RGB',(310,140),(233,237,236));draw=ImageDraw.Draw(label)
            draw.rectangle((0,0,309,139),outline=(28,30,32),width=12)
            draw.text((73,6),'02',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',100),fill=(15,23,28))
            lo=tri.reshape(-1,3).min(0);hi=tri.reshape(-1,3).max(0)
            u=np.clip(((points[:,1]-lo[1])/(hi[1]-lo[1])*309).astype(int),0,309)
            v=np.clip(((hi[2]-points[:,2])/(hi[2]-lo[2])*139).astype(int),0,139)
            colors=np.asarray(label)[v,u]/255.
        sampled.append((points,normal,colors))
        stats[kind]=stats.get(kind,0)+len(ids)
        pname=classify_part(name)
        if pname not in part_names: part_names.append(pname)
        part_ids.append(np.full(len(ids),part_names.index(pname),np.int16))
    return [np.concatenate([p[i] for p in sampled]) for i in range(3)], np.concatenate(full), stats, (np.concatenate(part_ids), part_names)


def shade(points,normals,colors,triangles):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    # Duplicate vertices are acceptable to the ray tracer and preserve sharp edges.
    verts=triangles.reshape(-1,3).astype(np.float32)
    inds=np.arange(len(verts),dtype=np.uint32).reshape(-1,3)
    scene.add_triangles(o3d.core.Tensor(verts),o3d.core.Tensor(inds))

    def cast(dirs):
        rays=np.c_[points+normals*.0006,dirs].astype(np.float32)
        return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()

    ambient=np.zeros(len(points)); rng=np.random.default_rng(14)
    for _ in range(8):
        dirs=rng.normal(size=points.shape)
        dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        dirs*=np.where((dirs*normals).sum(1)>0,1,-1)[:,None]
        distances=cast(dirs)
        ambient+=np.where(np.isfinite(distances),np.clip(distances/.04,0,1),1)/8
    light=np.array([-.4,-.5,1.]); light/=np.linalg.norm(light)
    light2=np.array([.6,.3,.7]);light2/=np.linalg.norm(light2)
    light_visibility=(cast(np.broadcast_to(light,points.shape))>.5).astype(float)
    diffuse=np.maximum(normals@light,0)
    fill=np.maximum(normals@light2,0)
    irradiance=.44+.36*ambient+.45*diffuse*(.4+.6*light_visibility)+.14*fill
    # Small broad sheen makes smooth housings and rims readable in baked splats.
    highlight=.07*np.maximum(normals@light,0)**12*ambient
    rgb=np.clip(colors*irradiance[:,None]+highlight[:,None],0,1)
    return rgb,ambient


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--out',type=Path,default=ROOT/'data/local/objects/f1tenth_detailed.npz')
    ap.add_argument('--spacing-mm',type=float,default=2.)
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args()
    source=args.source or Path(json.loads((ROOT/'data/local/objects/f1tenth.json').read_text())['source'])
    spacing=args.spacing_mm/1000
    if not .0005 <= spacing <= .005: raise ValueError('spacing must be 0.5–5 mm')
    components,origin=load_components(source)
    details(components,origin)
    (points,normals,colors),triangles,stats,(parts,part_names)=sample_components(components,spacing,0)
    print('Samples:',len(points),'materials:',stats,flush=True)
    rgb,ao=shade(points,normals,colors,triangles)
    # Isotropic tangent discs avoid CAD triangle aspect ratios creating 24 mm splats.
    tangent=np.cross(normals,np.array([0.,0.,1.]))
    bad=np.linalg.norm(tangent,axis=1)<.01
    tangent[bad]=np.cross(normals[bad],np.array([0.,1.,0.]))
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    bitangent=np.cross(normals,tangent)
    rot=np.stack([tangent,bitangent,normals],axis=2)
    quats=Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]]
    scales=np.tile([spacing*.68,spacing*.68,spacing*.12],(len(points),1))
    sh=np.zeros((len(points),16,3),np.float32);sh[:,0]=(rgb-.5)/SH_C0
    meta={'name':'f1tenth_detailed','source':str(source),'frame':'metres, +X forward, +Y left, +Z up; wheel-ground origin',
          'provenance':'Reference-informed AutoDRIVE CAD with component materials, new wires, smooth normals, ray-traced baked ambient occlusion; not a photographic reconstruction',
          'appearance_baked':True,'palette':'restrained black and grey, metallic hardware',
          'n':len(points),'spacing_mm':args.spacing_mm,'material_counts':stats,
          'source_origin':origin.tolist(),'ao_mean':float(ao.mean()),
          'references':['https://f1tenth-doc.readthedocs.io/en/latest/getting_started/build_car/index.html',
                        'https://f1tenth-doc.readthedocs.io/en/latest/getting_started/build_car/upper_level_chassis.html',
                        'https://www.raspitt.org/projects/f1tenth','reference/vehicle/fotos/fahrzeug_foto2.jpg'],
          'license_note':'Derived from local AutoDRIVE F1TENTH CAD; retain its original license. Reference photos were inspected, not copied into textures.'}
    meta['parts']=wheel_parts_meta(points,parts,part_names)
    obj=ObjectSplat(points,np.log(scales),quats,np.full(len(points),np.log(99)),sh,meta,parts=parts)
    lo,hi=obj.bbox()
    obj=obj.with_part(contact_shadow(hi[0]-lo[0],hi[1]-lo[1]),{'name':'shadow','kind':'shadow','lidar':False})
    meta=obj.meta
    assert np.isfinite(points).all() and np.isfinite(sh).all()
    assert np.max(scales)<.004 and np.max(np.abs(np.linalg.norm(normals,axis=1)-1))<1e-5
    args.out.parent.mkdir(parents=True,exist_ok=True);obj.save(args.out)
    args.out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print('Saved',args.out,'bounds',obj.bbox(),flush=True)
    if args.preview: render_preview(obj,args.out.with_name(args.out.stem+'_preview.png'),size_px=640)


if __name__=='__main__': main()
