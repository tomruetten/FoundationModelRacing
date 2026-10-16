"""GPU ray intersection against the same detailed vehicle mesh Isaac displays."""
import numpy as np
import warp as wp
from .config import pose_matrix
from .lidar import RaySamples


@wp.kernel
def _cast(mesh:wp.uint64,origins:wp.array(dtype=wp.vec3),directions:wp.array(dtype=wp.vec3),
          maximum:float,ranges:wp.array(dtype=float),hits:wp.array(dtype=wp.int32)):
    i=wp.tid()
    query=wp.mesh_query_ray(mesh,origins[i],directions[i],maximum)
    if query.result:
        ranges[i]=query.t
        hits[i]=1
    else:
        ranges[i]=maximum
        hits[i]=0


class VehicleRaycaster:
    def __init__(self,triangles,device='cuda:0',maximum=30.):
        self.device=device; self.maximum=maximum
        vertices=np.asarray(triangles,dtype=np.float32).reshape(-1,3)
        self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),
                          indices=wp.array(np.arange(len(vertices),dtype=np.int32),dtype=wp.int32,device=device))

    def render(self,lidar_pose,rays,vehicle_state,intensity=.5,return_probability=1.):
        transform=np.linalg.inv(pose_matrix(vehicle_state))@lidar_pose
        directions=np.asarray(rays)@transform[:3,:3].T
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        origins=np.broadcast_to(transform[:3,3],directions.shape).copy()
        distances=wp.empty(len(rays),dtype=float,device=self.device)
        hits=wp.empty(len(rays),dtype=wp.int32,device=self.device)
        wp.launch(_cast,dim=len(rays),inputs=[self.mesh.id,
            wp.array(origins,dtype=wp.vec3,device=self.device),wp.array(directions,dtype=wp.vec3,device=self.device),
            self.maximum,distances,hits],device=self.device)
        mask=hits.numpy().astype(bool)
        return RaySamples(distances.numpy(),mask,np.full(len(rays),intensity),np.full(len(rays),return_probability))
