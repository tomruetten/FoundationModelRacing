import math
import numpy as np
import pytest
from racing.geometry import swept_vehicle_contact, swept_boundary_contact
from racing.core import VehicleState


def test_swept_crossing_and_touch():
    a, b = VehicleState(x=-2), VehicleState(x=2)
    t = swept_vehicle_contact(a, b, VehicleState(), VehicleState())
    assert t == pytest.approx((2 - .42) / 4, abs=1e-5)
    assert swept_vehicle_contact(a, a, b, b) is None
    assert swept_vehicle_contact(VehicleState(x=-.42), VehicleState(x=-.42), VehicleState(), VehicleState()) == 0


def test_rotation_contact_with_clear_endpoints():
    a = VehicleState(yaw=0)
    b = VehicleState(yaw=math.pi/2)
    other = VehicleState(x=.44)
    assert swept_vehicle_contact(a, a, other, other) is None
    assert swept_vehicle_contact(b, b, other, other) is None
    assert swept_vehicle_contact(a, b, other, other) is not None


def test_boundary_tunneling_and_inner_island():
    outer = np.array([[-3,-3],[3,-3],[3,3],[-3,3]])
    inner = np.array([[-.1,-1],[.1,-1],[.1,1],[-.1,1]])
    assert swept_boundary_contact(VehicleState(x=-2), VehicleState(x=2), outer, [inner]) is not None
    t = swept_boundary_contact(VehicleState(), VehicleState(x=5), outer, [])
    assert t == pytest.approx((3-.21)/5, abs=1e-5)
    assert swept_boundary_contact(VehicleState(x=-2), VehicleState(x=-1), outer, [inner]) is None
