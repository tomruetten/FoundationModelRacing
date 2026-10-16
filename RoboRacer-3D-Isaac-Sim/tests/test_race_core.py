import math
from pathlib import Path
import numpy as np
import pytest
from racing.core import Command, RaceEngine, VehicleState
from racing.dynamics import KinematicDynamics, GRUDynamics


def circle(radius=4):
    a = np.linspace(0, 2*math.pi, 240, endpoint=False)
    return np.column_stack((radius*np.cos(a), radius*np.sin(a)))


def engine(models=None, **kwargs):
    return RaceEngine(models=models or {'ego':KinematicDynamics(), 'opponent':KinematicDynamics()},
        reference_path=circle(), spawns={'ego':VehicleState(x=4,yaw=math.pi/2),
        'opponent':VehicleState(x=-4,yaw=-math.pi/2)}, **kwargs)


def test_timeout_mode_and_immutable_snapshot():
    race = engine(command_timeout=.3)
    race.set_command(Command(1, 0), 'manual', wall_time=0)
    race.resume()
    assert race.step(wall_time=.1).step_index == 1
    old = race.snapshot()
    with pytest.raises(TypeError):
        old.vehicles['ego'] = VehicleState()
    assert race.step(wall_time=.4).status == 'paused'
    assert race.snapshot().step_index == 1
    race.set_control_mode('ros')
    assert race.set_command(Command(1,0), 'manual', wall_time=.5) is False
    race.resume()
    assert race.step(wall_time=.5).step_index == 1
    race.set_command(Command(1,0), 'ros', wall_time=.5)
    race.resume()
    assert race.step(wall_time=.5).step_index == 2
    assert old.step_index == 1


def test_reset_deterministic_and_models_independent():
    class History(KinematicDynamics):
        def reset(self): self.history=[]
        def predict(self, state, command, dt):
            self.history.append(command)
            return super().predict(state,command,dt)
    a,b = History(),History()
    race=engine({'ego':a,'opponent':b})
    race.set_control_mode('autopilot'); race.resume()
    first=race.step().vehicles
    assert len(a.history)==len(b.history)==1 and a.history is not b.history
    race.reset(seed=0)
    assert not a.history and not b.history
    race.resume()
    assert race.step().vehicles == first
    with pytest.raises(ValueError): engine({'ego':a, 'opponent':a})


def test_inference_failure_is_atomic_and_requires_reset():
    class Broken(KinematicDynamics):
        def predict(self,*args): raise RuntimeError('bad model')
    race=engine({'ego':KinematicDynamics(),'opponent':Broken()})
    race.set_control_mode('autopilot'); race.resume()
    before=race.snapshot()
    after=race.step()
    assert after.vehicles==before.vehicles and after.step_index==0
    assert after.status=='paused' and 'bad model' in after.reason
    race.resume()
    assert race.snapshot().status=='paused'


def test_collision_finishes_at_first_contact():
    class Jump(KinematicDynamics):
        def predict(self,state,command,dt): return VehicleState(x=2 if state.x<0 else -2)
    race=RaceEngine(models={'ego':Jump(),'opponent':Jump()},reference_path=circle(),
        spawns={'ego':VehicleState(x=-2), 'opponent':VehicleState(x=2)})
    race.set_control_mode('autopilot'); race.resume()
    snap=race.step()
    assert snap.status=='finished' and snap.collision.kind=='vehicle'
    assert snap.vehicles['ego'].x == pytest.approx(-.21,abs=1e-5)
    assert snap.vehicles['opponent'].x == pytest.approx(.21,abs=1e-5)
    assert race.step()==snap


def test_diagnostic_autopilot_three_laps():
    race=engine(); race.set_control_mode('autopilot'); race.resume()
    for _ in range(2800):
        snap=race.step()
        assert snap.status=='running', snap.reason
        if min(snap.laps.values())>=3: break
    assert min(snap.laps.values())>=3
    race.reset()
    assert set(race.snapshot().laps.values())=={0}


def test_shipped_gru_rollout_and_reset():
    checkpoint=Path('dynamics/deployed/trained_models/gru_run30_best.pth')
    if not checkpoint.exists(): pytest.skip('shipped checkpoint unavailable')
    a,b=GRUDynamics(checkpoint),GRUDynamics(checkpoint)
    state=VehicleState()
    first=a.predict(state,Command(1.5,.05),.02)
    assert first == b.predict(state,Command(1.5,.05),.02)
    for _ in range(30):
        state=a.predict(state,Command(1.5,.05),.02)
        assert all(math.isfinite(v) for v in vars(state).values())
    b.reset()
    assert b.predict(VehicleState(),Command(1.5,.05),.02)==first


def test_commands_are_forward_only_and_bounded():
    class Capture(KinematicDynamics):
        def predict(self,state,command,dt):
            self.command=command
            return super().predict(state,command,dt)
    model=Capture()
    race=engine({'ego':model,'opponent':KinematicDynamics()})
    race.set_command(Command(-10,10),'manual',wall_time=0)
    race.resume(); race.step(wall_time=0)
    assert model.command==Command(0,.29)


def test_boundary_collision_and_nonfinite_prediction():
    outer=np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
    class Jump(KinematicDynamics):
        def predict(self,state,command,dt): return VehicleState(x=10)
    race=engine({'ego':Jump(),'opponent':KinematicDynamics()},outer_boundary=outer)
    race.set_control_mode('autopilot');race.resume()
    snap=race.step()
    assert snap.status=='finished' and snap.collision.kind=='boundary'
    class NaN(KinematicDynamics):
        def predict(self,*args): return VehicleState(vx=float('nan'))
    race=engine({'ego':NaN(),'opponent':KinematicDynamics()})
    race.set_control_mode('autopilot');race.resume()
    assert race.step().step_index==0


def test_reverse_gate_toggling_does_not_count_laps():
    class Toggle(KinematicDynamics):
        def reset(self): self.i=0
        def predict(self,state,command,dt):
            self.i+=1
            angle=.005 if self.i%2 else -.005
            return VehicleState(x=4*math.cos(angle),y=4*math.sin(angle),yaw=angle+math.pi/2)
    race=RaceEngine(models={'ego':Toggle()},reference_path=circle(),
        spawns={'ego':VehicleState(x=4,yaw=math.pi/2)})
    race.set_control_mode('autopilot');race.resume()
    for _ in range(100): snap=race.step()
    assert snap.laps['ego']==0
