from dataclasses import dataclass

@dataclass(slots=True)
class OrbitalElements:
    a: float         # semi-major axis [m]
    e: float         # eccentricity [-]
    i: float         # inclination [rad]
    raan: float      # right ascension of ascending node Ω [rad]
    argp: float      # argument of perigee ω [rad]
    M0: float        # mean anomaly at epoch t0 [rad]
    mu: float        # gravitational parameter [m^3/s^2]
    t0: float        # epoch [s since some reference]
