import math
from kepler_solver import KeplerSolver

def test_circular_orbit():
    solver = KeplerSolver()
    M = math.pi / 2
    e = 0.0
    E = solver.solve_kepler_equation(M, e)
    assert math.isclose(E, M, rel_tol=1e-10)

def test_round_trip_conversion():
    solver = KeplerSolver()
    nu = 2.0
    e = 0.3
    M = solver.true_to_mean_anomaly(nu, e)
    nu_recovered = solver.mean_to_true_anomaly(M, e)
    assert math.isclose(nu, nu_recovered, abs_tol=1e-6)
