import math

def mean_motion(mu: float, a: float) -> float:
    return math.sqrt(mu / a**3)

def solve_kepler(M: float, e: float, tol: float = 1e-12, itmax: int = 50) -> float:
    """Return eccentric anomaly E given mean anomaly M and eccentricity e (elliptic)."""
    # Normalize M to [-pi, pi] for faster convergence
    M = (M + math.pi) % (2 * math.pi) - math.pi
    E = M if e < 0.8 else math.pi  # starter
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def true_anomaly_from_E(E: float, e: float) -> float:
    s = math.sqrt(1 - e**2) * math.sin(E) / (1 - e * math.cos(E))
    c = (math.cos(E) - e) / (1 - e * math.cos(E))
    return math.atan2(s, c)

def radius_from_E(a: float, e: float, E: float) -> float:
    return a * (1 - e * math.cos(E))
