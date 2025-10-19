import argparse
import math
from .elements import OrbitalElements
from .propagate import r_eci_at

MU_EARTH = 3.986004418e14  # m^3/s^2

def main() -> None:
    p = argparse.ArgumentParser(description="Space Determination: simple two-body demo")
    p.add_argument("--a", type=float, required=True, help="semi-major axis [m]")
    p.add_argument("--e", type=float, required=True, help="eccentricity")
    p.add_argument("--i", type=float, required=True, help="inclination [deg]")
    p.add_argument("--raan", type=float, required=True, help="RAAN Ω [deg]")
    p.add_argument("--argp", type=float, required=True, help="arg. of perigee ω [deg]")
    p.add_argument("--M0", type=float, required=True, help="mean anomaly at epoch [deg]")
    p.add_argument("--t0", type=float, default=0.0, help="epoch seconds (ref=0)")
    p.add_argument("--t",  type=float, default=0.0, help="target time [s]")
    args = p.parse_args()

    el = OrbitalElements(
        a=args.a, e=args.e,
        i=math.radians(args.i),
        raan=math.radians(args.raan),
        argp=math.radians(args.argp),
        M0=math.radians(args.M0),
        mu=MU_EARTH, t0=args.t0
    )
    r = r_eci_at(el, args.t)
    print(f"r_ECI(t={args.t:.1f}s) = [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}] m")
