"""Independent learned dynamics histories, plus an explicit diagnostic model."""
from collections import deque
import math
from pathlib import Path


def _integrate(state, vx, vy, yr, ax, ay, dt):
    from .core import VehicleState
    heading=state.yaw+.5*yr*dt
    return VehicleState(x=state.x+(vx*math.cos(heading)-vy*math.sin(heading))*dt,
        y=state.y+(vx*math.sin(heading)+vy*math.cos(heading))*dt,
        yaw=(state.yaw+yr*dt+math.pi)%(2*math.pi)-math.pi,
        vx=vx,vy=vy,yr=yr,ax=ax,ay=ay)


class KinematicDynamics:
    """Bicycle dynamics for tests/diagnostics only; never an automatic fallback."""
    def reset(self): pass

    def predict(self,state,command,dt):
        yr=command.speed*math.tan(command.steering)/.3225
        return _integrate(state,command.speed,0.,yr,(command.speed-state.vx)/dt,
                          command.speed*yr,dt)


class GRUDynamics:
    """Run30 GRU, CPU inference, private ten-sample history per vehicle.

    Implements the shipped 15_gru_inference.py architecture and feature order.
    Prediction returns a whole world-pose state; no additional integration is
    required by callers. Checkpoints are loaded with weights_only=True.
    """
    def __init__(self,checkpoint):
        import torch
        import numpy as np
        from torch import nn
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru1=nn.GRU(7,100,batch_first=True)
                self.fc1=nn.Linear(100,100)
                self.fc3=nn.Linear(100,100)
                self.fc5=nn.Linear(100,5)
                self.LeakyReLU=nn.LeakyReLU(.1)
            def forward(self,x):
                x,_=self.gru1(x)
                return self.fc5(self.LeakyReLU(self.fc3(self.LeakyReLU(self.fc1(x[:,-1,:])))))
        self._torch=torch
        self.checkpoint=str(Path(checkpoint).resolve())
        self.model=Network().cpu()
        # Shipped training metadata contains NumPy arrays; allow only their
        # reconstruction types, keeping arbitrary pickle globals disallowed.
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct,
                (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
                np.ndarray,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]):
            weights=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if isinstance(weights,dict) and 'model_state_dict' in weights:
            weights=weights['model_state_dict']
        self.model.load_state_dict(weights,strict=True)
        self.model.eval()
        self.reset()

    def reset(self):
        self.history=deque([[0.]*7 for _ in range(10)],maxlen=10)

    def predict(self,state,command,dt):
        if not math.isclose(dt,.02,rel_tol=0,abs_tol=1e-10):
            raise ValueError('Run30 checkpoint requires dt=0.02 seconds')
        self.history.append([state.vx,state.vy,state.yr,state.ax,state.ay,command.speed,command.steering])
        torch=self._torch
        with torch.inference_mode():
            values=self.model(torch.tensor(list(self.history),dtype=torch.float32)[None]).flatten().tolist()
        if not all(math.isfinite(v) for v in values):
            raise ValueError('GRU returned non-finite state')
        return _integrate(state,*values,dt)
