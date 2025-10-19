import math
import numpy as np
from space_determination.elements import OrbitalElements
from space_determination.propagate import r_eci_at

MU = 3.986004418e14

def test_imports():
    import space_determination
    assert hasattr(space_determination, "__all__")

def test_two_body_radius_reasonable():
    el = OrbitalElements(
        a=7000e3, e=0.001, i=math.radians(51.6),
        raan=0.0, argp=0.0, M0=0.0, mu=MU, t0=0.0
    )
    r0 = r_eci_at(el, 0.0)
    assert np.isfinite(r0).all()
    # near perigee radius ~ a(1-e)
    assert 6993e3 < np.linalg.norm(r0) < 7007e3
