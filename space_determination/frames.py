import math
import numpy as np

def pqw_to_eci_matrix(raan: float, i: float, argp: float) -> np.ndarray:
    """Rotation from orbital-plane PQW to ECI by (Ω, i, ω)."""
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i),    math.sin(i)
    co, so = math.cos(argp), math.sin(argp)
    # R3(Ω) R1(i) R3(ω)
    return np.array([
        [ cO*co - sO*so*ci, -cO*so - sO*co*ci,  sO*si],
        [ sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
        [ so*si,             co*si,             ci   ],
    ])
