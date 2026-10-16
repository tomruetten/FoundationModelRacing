"""Continuous contact for rigid rectangles following linear pose interpolation.

Conservative advancement uses a bound on every corner's speed. It therefore
checks the entire sweep, including rotation, rather than sampling endpoints.
Contact tolerance is one micrometre; unresolved convergence stops conservatively.
"""
import math
import numpy as np

LENGTH = .42
WIDTH = .26
_EPS = 1e-6


def angle_delta(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def interpolate(a, b, fraction):
    from .core import VehicleState
    values = {key: getattr(a, key) + fraction * (getattr(b, key)-getattr(a, key))
              for key in vars(a)}
    values['yaw'] = a.yaw + fraction * angle_delta(a.yaw, b.yaw)
    return VehicleState(**values)


def rectangle(state, length=LENGTH, width=WIDTH):
    local = np.array([[length/2,width/2],[-length/2,width/2],
                      [-length/2,-width/2],[length/2,-width/2]])
    c,s=math.cos(state.yaw),math.sin(state.yaw)
    return local @ np.array([[c,s],[-s,c]]) + [state.x,state.y]


def _point_segment_distances(points, vertices):
    ends=np.roll(vertices,-1,axis=0)
    edges=ends-vertices
    delta=points[:,None,:]-vertices[None,:,:]
    t=np.clip(np.sum(delta*edges,axis=2)/np.maximum(np.sum(edges*edges,axis=1),1e-30),0,1)
    return np.linalg.norm(delta-t[:,:,None]*edges,axis=2)


def _edge_distance(a,b):
    # Segment crossings require zero distance even when no vertex is nearby.
    u=np.roll(a,-1,axis=0)-a
    v=np.roll(b,-1,axis=0)-b
    w=b[None,:,:]-a[:,None,:]
    def cross(x,y): return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denom=cross(u[:,None,:],v[None,:,:])
    safe=np.where(abs(denom)>1e-15,denom,1)
    s=cross(w,v[None,:,:])/safe
    t=cross(w,u[:,None,:])/safe
    if np.any((abs(denom)>1e-15)&(s>=0)&(s<=1)&(t>=0)&(t<=1)):
        return 0.
    return float(min(_point_segment_distances(a,b).min(),_point_segment_distances(b,a).min()))


def point_in_polygon(point, polygon):
    x,y=point
    a=np.asarray(polygon); b=np.roll(a,-1,axis=0)
    dy=b[:,1]-a[:,1]
    crossings=((a[:,1]>y)!=(b[:,1]>y)) & (x < (b[:,0]-a[:,0])*(y-a[:,1])/np.where(abs(dy)>1e-15,dy,1)+a[:,0])
    return bool(np.count_nonzero(crossings)%2)


def _rectangle_distance(a,b):
    # SAT gap is a lower bound on Euclidean distance, sufficient for advancement.
    edges=np.concatenate((np.roll(a,-1,axis=0)-a,np.roll(b,-1,axis=0)-b))
    axes=np.column_stack((-edges[:,1],edges[:,0]))
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    pa=a@axes.T; pb=b@axes.T
    return max(0.,float(np.max(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)))))


def _motion_bound(a,b):
    return math.hypot(b.x-a.x,b.y-a.y)+math.hypot(LENGTH,WIDTH)/2*abs(angle_delta(a.yaw,b.yaw))


def _advance(distance, speed):
    t=0.
    for _ in range(256):
        gap=distance(t)
        if gap<=_EPS: return t
        if speed<=1e-15: return None
        advance=gap/speed
        if t+advance>1: return None
        t+=advance
    # Treat a nonconvergent near-contact conservatively; never tunnel.
    return t


def swept_vehicle_contact(a0,a1,b0,b1):
    """Earliest normalized contact time in [0,1], or None."""
    return _advance(lambda t:_rectangle_distance(rectangle(interpolate(a0,a1,t)),
        rectangle(interpolate(b0,b1,t))),_motion_bound(a0,a1)+_motion_bound(b0,b1))


def swept_boundary_contact(start,end,outer_boundary=None,inner_boundaries=()):
    """Contact with outside of outer polygon or interior of any island."""
    outer=None if outer_boundary is None else np.asarray(outer_boundary,dtype=float)
    inners=[np.asarray(p,dtype=float) for p in inner_boundaries]
    if outer is None and not inners: return None
    def distance(t):
        rect=rectangle(interpolate(start,end,t))
        gaps=[]
        if outer is not None:
            if not point_in_polygon(rect[0],outer): return 0.
            gaps.append(_edge_distance(rect,outer))
        for inner in inners:
            if point_in_polygon(rect[0],inner) or point_in_polygon(inner[0],rect): return 0.
            gaps.append(_edge_distance(rect,inner))
        return min(gaps)
    return _advance(distance,_motion_bound(start,end))
