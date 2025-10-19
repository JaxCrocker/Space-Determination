import numpy as np
from .elements import OrbitalElements
from .kepler import mean_motion, solve_kepler, true_anomaly_from_E, radius_from_E
from .frames import pqw_to_eci_matrix
import math

def r_eci_at(elements: OrbitalElements, t: float) -> np.ndarray:
    n = mean_motion(elements.mu, elements.a)
    M = elements.M0 + n * (t - elements.t0)
    E = solve_kepler(M, elements.e)
    nu = true_anomaly_from_E(E, elements.e)
    r = radius_from_E(elements.a, elements.e, E)

    # position in orbital plane (PQW)
    r_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    C = pqw_to_eci_matrix(elements.raan, elements.i, elements.argp)
    return C @ r_pqw
