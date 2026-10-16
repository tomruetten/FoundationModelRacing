"""Deterministic, fixed-step race state independent of a rendering backend."""
from dataclasses import dataclass
import math
from time import monotonic
from types import MappingProxyType
from typing import Mapping
import numpy as np
from .geometry import angle_delta, interpolate, swept_boundary_contact, swept_vehicle_contact


@dataclass(frozen=True)
class VehicleState:
    x: float = 0.
    y: float = 0.
    yaw: float = 0.
    vx: float = 0.
    vy: float = 0.
    yr: float = 0.
    ax: float = 0.
    ay: float = 0.


@dataclass(frozen=True)
class Command:
    speed: float = 0.
    steering: float = 0.


@dataclass(frozen=True)
class Collision:
    kind: str
    vehicles: tuple
    fraction: float


@dataclass(frozen=True)
class Snapshot:
    episode_id: int
    step_index: int
    time_s: float
    vehicles: Mapping[str, VehicleState]
    status: str
    collision: Collision | None
    laps: Mapping[str, int]
    reason: str = ''
    control_mode: str = 'manual'


class RaceEngine:
    """Single-owner stepping API. Runtime queues inputs onto this owner thread.

    Each injected model must implement reset() and predict(state, command, dt).
    Prediction failure leaves the published state unchanged and requires reset:
    model histories may have advanced, so resuming that episode is unsafe.
    """
    def __init__(self,models,reference_path,spawns,outer_boundary=None,
                 inner_boundaries=None,dt=.02,command_timeout=.3,
                 ego_id='ego',opponent_speed=1.5,max_speed=2.5,
                 max_steering=.29,lookahead=.9):
        if set(models)!=set(spawns) or ego_id not in spawns:
            raise ValueError('models and spawns must share IDs and include ego_id')
        if len({id(model) for model in models.values()})!=len(models):
            raise ValueError('Each vehicle requires an independent model instance')
        if not all(math.isfinite(v) and v>0 for v in
                   (dt,command_timeout,max_speed,max_steering,lookahead)):
            raise ValueError('Time steps, limits, and lookahead must be positive and finite')
        self.models=dict(models); self.spawns=dict(spawns)
        for state in self.spawns.values(): self._validate_state(state)
        self.ego_id=ego_id; self.dt=float(dt); self.command_timeout=command_timeout
        self.max_speed=max_speed; self.max_steering=max_steering
        self.opponent_speed=opponent_speed; self.lookahead=lookahead
        path=np.asarray(reference_path,dtype=float).copy()
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<3 or not np.isfinite(path).all():
            raise ValueError('reference_path must contain at least three finite XY points')
        if np.linalg.norm(path[0]-path[-1])<1e-9: path=path[:-1]
        self.path=path
        self._edges=np.roll(path,-1,axis=0)-path
        self._lengths=np.linalg.norm(self._edges,axis=1)
        if len(path)<3 or np.any(self._lengths<1e-9):
            raise ValueError('reference_path contains duplicate consecutive points')
        self._cumulative=np.r_[0.,np.cumsum(self._lengths)]
        self.path_length=float(self._cumulative[-1])
        self.outer_boundary=self._polygon(outer_boundary)
        self.inner_boundaries=[self._polygon(p) for p in (inner_boundaries or [])]
        self.control_mode='manual'; self.episode_id=-1
        self.reset()

    @staticmethod
    def _polygon(polygon):
        if polygon is None: return None
        p=np.asarray(polygon,dtype=float).copy()
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
            raise ValueError('boundary must be a finite polygon with at least three vertices')
        return p

    @staticmethod
    def _validate_state(state):
        if not isinstance(state,VehicleState) or not all(math.isfinite(v) for v in vars(state).values()):
            raise ValueError('Dynamics must return a finite VehicleState')

    def _project(self,state):
        point=np.array([state.x,state.y])
        t=np.clip(np.sum((point-self.path)*self._edges,axis=1)/self._lengths**2,0,1)
        nearest=self.path+t[:,None]*self._edges
        i=int(np.argmin(np.sum((nearest-point)**2,axis=1)))
        return float(self._cumulative[i]+t[i]*self._lengths[i])%self.path_length

    def _autopilot(self,state):
        progress=(self._project(state)+self.lookahead)%self.path_length
        i=min(int(np.searchsorted(self._cumulative,progress,side='right')-1),len(self.path)-1)
        target=self.path[i]+(progress-self._cumulative[i])/self._lengths[i]*self._edges[i]
        dx,dy=target-[state.x,state.y]
        lateral=-math.sin(state.yaw)*dx+math.cos(state.yaw)*dy
        steering=math.atan2(2*.3225*lateral,max(dx*dx+dy*dy,1e-9))
        return self._clamp(Command(self.opponent_speed,steering))

    def _clamp(self,command):
        if not isinstance(command,Command) or not all(math.isfinite(v) for v in (command.speed,command.steering)):
            raise ValueError('Command must contain finite speed and steering')
        return Command(float(np.clip(command.speed,0.,self.max_speed)),
                       float(np.clip(command.steering,-self.max_steering,self.max_steering)))

    def reset(self,seed=0):
        # No randomization is used; seed is reserved for explicit future spawn jitter.
        self.seed=int(seed)
        for model in self.models.values(): model.reset()
        self.episode_id+=1; self.step_index=0; self.time_s=0.
        self.vehicles=dict(self.spawns); self.status='paused'; self.reason='reset'
        self.collision=None; self._faulted=False
        self._command=None; self._command_time=None
        self._progress={key:self._project(s) for key,s in self.vehicles.items()}
        self._distance={key:0. for key in self.vehicles}
        self.laps={key:0 for key in self.vehicles}
        return self.snapshot()

    def set_control_mode(self,mode):
        if mode not in ('manual','ros','autopilot'): raise ValueError('Unknown control mode')
        self.control_mode=mode
        self._command=None; self._command_time=None
        return self.snapshot()

    def set_command(self,command,source,wall_time=None):
        if source!=self.control_mode or source=='autopilot': return False
        command=self._clamp(command)
        now=monotonic() if wall_time is None else float(wall_time)
        if not math.isfinite(now): raise ValueError('wall_time must be finite')
        self._command=command; self._command_time=now
        return True

    def pause(self,reason='paused'):
        if self.status!='finished': self.status='paused'; self.reason=str(reason)
        return self.snapshot()

    def resume(self):
        if self.status!='finished' and not self._faulted:
            self.status='running'; self.reason=''
        return self.snapshot()

    def snapshot(self):
        return Snapshot(self.episode_id,self.step_index,self.time_s,
            MappingProxyType(dict(self.vehicles)),self.status,self.collision,
            MappingProxyType(dict(self.laps)),self.reason,self.control_mode)

    def step(self,wall_time=None):
        if self.status!='running': return self.snapshot()
        now=monotonic() if wall_time is None else float(wall_time)
        if not math.isfinite(now): return self.pause('invalid wall clock')
        if self.control_mode!='autopilot' and (self._command is None or
                not 0<=now-self._command_time<=self.command_timeout):
            return self.pause('command timeout')
        try:
            proposed={}
            for key,state in self.vehicles.items():
                command=self._command if key==self.ego_id and self.control_mode!='autopilot' else self._autopilot(state)
                proposed[key]=self.models[key].predict(state,command,self.dt)
                self._validate_state(proposed[key])
            contacts=[]
            keys=list(self.vehicles)
            for i,key in enumerate(keys):
                t=swept_boundary_contact(self.vehicles[key],proposed[key],self.outer_boundary,self.inner_boundaries)
                if t is not None: contacts.append(Collision('boundary',(key,),t))
                for other in keys[i+1:]:
                    t=swept_vehicle_contact(self.vehicles[key],proposed[key],self.vehicles[other],proposed[other])
                    if t is not None: contacts.append(Collision('vehicle',(key,other),t))
            collision=min(contacts,key=lambda c:c.fraction) if contacts else None
            fraction=collision.fraction if collision else 1.
            updated={key:interpolate(self.vehicles[key],state,fraction) for key,state in proposed.items()}
            progress={}; distances=dict(self._distance); laps=dict(self.laps)
            for key,state in updated.items():
                progress[key]=self._project(state)
                delta=(progress[key]-self._progress[key]+self.path_length/2)%self.path_length-self.path_length/2
                old=self.vehicles[key]
                # Reject projection jumps across nearby, unrelated track segments.
                if abs(delta)<=2*math.hypot(state.x-old.x,state.y-old.y)+.1:
                    distances[key]+=delta
                    laps[key]=max(laps[key],int(max(0.,distances[key])/self.path_length))
        except Exception as exc:
            self._faulted=True
            return self.pause(f'dynamics/step error: {exc}; reset required')
        self.vehicles=updated; self._progress=progress; self._distance=distances; self.laps=laps
        self.step_index+=1; self.time_s+=self.dt*fraction
        if collision:
            self.collision=collision; self.status='finished'; self.reason=f'{collision.kind} contact'
        return self.snapshot()
