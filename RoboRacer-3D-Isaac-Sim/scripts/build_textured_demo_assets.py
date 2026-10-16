#!/usr/bin/env python3
"""Convert licensed GLB demo assets to smooth, textured Gaussian surfaces.

Run with roboracer and DracoPy available. Inputs live in data/local/objects/sources.
UVs are sampled at each Gaussian, preserving texture detail on large triangles.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from build_f1tenth_appearance import shade
from render_learned_demo import ObjectSplat, ROOT
from object_splats import SH_C0, linear_to_srgb


def sample_texture(texture, uv):
    """Bilinear glTF repeat sampling at texel centres, retaining alpha."""
    image=np.asarray(texture.convert('RGBA'));h,w=image.shape[:2]
    x=uv[:,0]*w-.5;y=(1-uv[:,1])*h-.5
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    tx=(x-ix)[:,None];ty=(y-iy)[:,None]
    a=image[iy%h,ix%w].astype(np.float32)
    b=image[iy%h,(ix+1)%w].astype(np.float32)
    c=image[(iy+1)%h,ix%w].astype(np.float32)
    d=image[(iy+1)%h,(ix+1)%w].astype(np.float32)
    return ((1-ty)*((1-tx)*a+tx*b)+ty*((1-tx)*c+tx*d))/255.


def build(name, spacing, length=None, height=None, high=False):
    src=ROOT/'data/local/objects/sources'/f'{name}.glb'
    if name=='plastic_crate' and high:
        src=src.parent/'plastic_crate_original/plastic_crate_02_4k.gltf'
    scene=trimesh.load(src,force='scene')
    # glTF Y-up, Porsche front +Z -> object +X front, +Y left, +Z up.
    basis=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    parts=[]
    for node in scene.graph.nodes_geometry:
        xf,key=scene.graph[node];mesh=scene.geometry[key];mat=mesh.visual.material
        # Clearcoat is a duplicate transparent shell, not a second opaque surface.
        if name=='porsche_911' and (node=='Plane_0' or mat.name=='coat'):continue
        v=trimesh.transform_points(mesh.vertices,xf)@basis.T
        normal=mesh.vertex_normals@np.linalg.inv(xf[:3,:3])@basis.T
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        parts.append((v,mesh.faces,normal,mesh.visual,mat.name,node))
    vertices=np.concatenate([v for v,*_ in parts]);lo,hi=vertices.min(0),vertices.max(0)
    scale=length/(hi[0]-lo[0]) if length else (height/(hi[2]-lo[2]) if height else 1.)
    center=(lo+hi)/2;center[2]=lo[2]
    rng=np.random.default_rng(17);points=[];normals=[];colors=[];triangles=[];part_ids=[]
    for verts,faces,norm,vis,matname,node in parts:
        tri=((verts-center)*scale)[faces];triangles.append(tri)
        areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        expected=2*areas/spacing**2
        counts=np.floor(expected).astype(int)+(rng.random(len(faces))<expected%1)
        ids=np.repeat(np.arange(len(faces)),counts)
        if not len(ids):continue
        a=np.sqrt(rng.random(len(ids)));b=rng.random(len(ids));w=np.c_[1-a,a*(1-b),a*b]
        p=np.einsum('ni,nij->nj',w,tri[ids]);n=np.einsum('ni,nij->nj',w,norm[faces[ids]])
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        mat=vis.material;tex=getattr(mat,'baseColorTexture',None);uv=getattr(vis,'uv',None)
        if tex is not None and uv is not None:
            q=np.einsum('ni,nij->nj',w,uv[faces[ids]])
            rgba=sample_texture(tex,q);rgb=rgba[:,:3]
            # Honour cutout decals rather than turning transparent pixels black.
            valid=rgba[:,3]>.5;p,n,rgb=p[valid],n[valid],rgb[valid]
        else:
            factor=getattr(mat,'baseColorFactor',None)
            c=np.array(factor[:3],float)/255 if factor is not None else np.array([.35]*3)
            rgb=np.tile(linear_to_srgb(c),(len(p),1))
        ids_part=np.zeros(len(p),np.int16)
        if name=='porsche_911' and node.startswith('Cylinder.') and matname!='Material.001':
            ids_part=1+(p[:,1]<0).astype(np.int16)+2*(p[:,0]<0).astype(np.int16)
        part_ids.append(ids_part)
        points.append(p);normals.append(n);colors.append(rgb)
    p,n,c=map(np.concatenate,(points,normals,colors));tris=np.concatenate(triangles)
    print(name,'samples',len(p),'bounds',np.ptp(p,axis=0),flush=True)
    rgb,ao=shade(p,n,c,tris)
    tangent=np.cross(n,[0,0,1.]);bad=np.linalg.norm(tangent,axis=1)<.01
    tangent[bad]=np.cross(n[bad],[0,1.,0]);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    q=Rotation.from_matrix(np.stack([tangent,np.cross(n,tangent),n],axis=2)).as_quat()[:,[3,0,1,2]]
    sh=np.zeros((len(p),16,3),np.float32);sh[:,0]=(rgb-.5)/SH_C0
    scales=np.tile([spacing*.72,spacing*.72,spacing*.12],(len(p),1))
    if name=='porsche_911':
        source='https://sketchfab.com/3d-models/free-porsche-911-carrera-4s-d01b254483794de3819786d93e0e1ebf'
        author='Lionsharp Studios';license='CC BY 4.0'
    else:
        metadata=ROOT/'data/local/objects/sources'/(name+'_metadata.json')
        if name=='cone_photoscan':metadata=metadata.with_name('cone_photoscan_blenderkit_metadata.json')
        m=json.loads(metadata.read_text())
        source='https://www.blenderkit.com/get-blenderkit/'+m['assetBaseId']+'/'
        author=m.get('author',{});license='CC0'
    meta=dict(name=name,source=str(src),source_url=source,author=author,license=license,
              source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),appearance_baked=True,
              frame='metres +X forward +Y left +Z up; ground origin',spacing_mm=spacing*1000,
              quality='high' if high else 'standard',texture_filter='bilinear',
              modifications='UV surface sampling, smooth normals, baked diffuse/contact lighting'+
              ('; Porsche scaled to 0.46 m, duplicate clearcoat and display ground removed' if name=='porsche_911' else ''),
              reconstruction='Textured mesh converted to Gaussians; not a photo-trained splat',
              dimensions_m=np.ptp(p,axis=0).tolist())
    labels=np.concatenate(part_ids)
    if name=='porsche_911':
        meta['parts']=[dict(name='body',kind='body')]
        for j,part_name in enumerate(['wheel_fl','wheel_fr','wheel_rl','wheel_rr'],1):
            wheel=p[labels==j];wl,wh=wheel.min(0),wheel.max(0)
            meta['parts'].append(dict(name=part_name,kind='wheel',centre=((wl+wh)/2).tolist(),
                                      axis=[0,1,0],radius=float((wh[2]-wl[2])/2),steer=j<3))
    obj=ObjectSplat(p,np.log(scales),q,np.full(len(p),np.log(99)),sh,meta,
                    parts=labels if name=='porsche_911' else None)
    assert np.isfinite(p).all() and np.isfinite(sh).all()
    out=ROOT/'data/local/objects'/(name+('_hd' if high else '')+'.npz');obj.save(out)
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print('saved',out,flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset',choices=['all','obstacles','porsche_911','wooden_pallet','wooden_crate','plastic_crate','cone_photoscan'],default='all')
    ap.add_argument('--quality',choices=['standard','high'],default='high')
    args=ap.parse_args()
    for name,spacing,length,height in [('porsche_911',.0017,.46,None),('wooden_pallet',.005,None,None),('wooden_crate',.005,None,None),('plastic_crate',.004,None,None),('cone_photoscan',.002,None,.33)]:
        if args.asset in ('all',name) or (args.asset=='obstacles' and name!='porsche_911'):
            high=args.quality=='high' and name!='porsche_911'
            if high:spacing=.001 if name=='cone_photoscan' else .002
            build(name,spacing,length,height,high)
