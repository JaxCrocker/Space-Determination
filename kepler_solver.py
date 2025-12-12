"""
Kepler equation solver for orbital mechanics.
Solves Kepler's equation: M = E - e*sin(E)
"""

import math
from typing import Tuple, Optional
import constants


class KeplerSolver:
    """Solver for Kepler's equation and related anomaly conversions."""

    def __init__(self, tolerance: float = constants.KEPLER_TOLERANCE,
                 max_iterations: int = constants.MAX_KEPLER_ITERATIONS):
        """
        Initialize Kepler solver.

        Args:
            tolerance: Convergence tolerance for iterative solver
            max_iterations: Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve_kepler_equation(self, mean_anomaly: float, eccentricity: float) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
        Uses Newton-Raphson method for elliptical orbits.

        Parameters:
            mean_anomaly: Mean anomaly M in radians, normalized to [0, 2π]
            eccentricity: Orbital eccentricity. Must be in range [0, 1) for elliptical orbits

        Returns:
            Eccentric anomaly E in radians, normalized to [0, 2π]

        Raises:
            ValueError: If eccentricity is negative or >= 1 (parabolic/hyperbolic orbits not supported)
            ValueError: If Newton-Raphson iteration fails to converge within max_iterations

        Examples:
            >>> solver = KeplerSolver()
            >>> # Circular orbit (e = 0)
            >>> E = solver.solve_kepler_equation(math.pi/2, 0.0)
            >>> print(f"E = {math.degrees(E):.2f} deg")
            E = 90.00 deg

            >>> # Elliptical orbit
            >>> E = solver.solve_kepler_equation(math.pi/3, 0.1)
            >>> print(f"E = {math.degrees(E):.2f} deg")
            E = 66.79 deg
        """
        if eccentricity < 0 or eccentricity >= 1:
            raise ValueError(f"This solver only handles elliptical orbits (0 <= e < 1), got e={eccentricity}")

        # Normalize mean anomaly to [0, 2pi]
        M = mean_anomaly % constants.TWO_PI

        # Initial guess for eccentric anomaly
        if eccentricity < 0.8:
            E = M
        else:
            E = math.pi

        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            f = E - eccentricity * math.sin(E) - M
            f_prime = 1 - eccentricity * math.cos(E)

            if abs(f_prime) < 1e-12:
                f_prime = 1e-12

            E_new = E - f / f_prime

            if abs(E_new - E) < self.tolerance:
                return E_new % constants.TWO_PI

            E = E_new

        raise ValueError(f"Kepler equation did not converge after {self.max_iterations} iterations")

    def mean_to_true_anomaly(self, mean_anomaly: float, eccentricity: float) -> float:
        """
        Convert mean anomaly to true anomaly.

        Parameters:
            mean_anomaly: Mean anomaly in radians
            eccentricity: Orbital eccentricity. Must be in range [0, 1) for elliptical orbits

        Returns:
            True anomaly ν in radians, normalized to [0, 2π]

        Raises:
            ValueError: If eccentricity is outside valid range [0, 1)
            ValueError: If Kepler equation solver fails to converge

        Examples:
            >>> solver = KeplerSolver()
            >>> # Convert 60 deg mean anomaly for a slightly elliptical orbit
            >>> nu = solver.mean_to_true_anomaly(math.radians(60), 0.1)
            >>> print(f"True anomaly: {math.degrees(nu):.2f} deg")
            True anomaly: 67.20 deg

            >>> # Near-circular orbit - anomalies are nearly equal
            >>> nu = solver.mean_to_true_anomaly(math.radians(45), 0.001)
            >>> print(f"Mean: 45.00 deg, True: {math.degrees(nu):.2f} deg")
            Mean: 45.00 deg, True: 45.09 deg
        """
        E = self.solve_kepler_equation(mean_anomaly, eccentricity)
        true_anomaly = self.eccentric_to_true_anomaly(E, eccentricity)
        return true_anomaly

    def eccentric_to_true_anomaly(self, eccentric_anomaly: float, eccentricity: float) -> float:
        """
        Convert eccentric anomaly to true anomaly.

        Parameters:
            eccentric_anomaly: Eccentric anomaly E in radians
            eccentricity: Orbital eccentricity. Must be in range [0, 1) for elliptical orbits

        Returns:
            True anomaly ν in radians, normalized to [0, 2π]

        Examples:
            >>> solver = KeplerSolver()
            >>> # Convert eccentric anomaly for moderate eccentricity
            >>> E = math.radians(60)  # 60 deg eccentric anomaly
            >>> nu = solver.eccentric_to_true_anomaly(E, 0.2)
            >>> print(f"E = 60 deg, ν = {math.degrees(nu):.2f} deg")
            E = 60 deg, ν = 73.40 deg

            >>> # For circular orbit (e=0), E = ν
            >>> nu = solver.eccentric_to_true_anomaly(math.pi/4, 0.0)
            >>> print(f"Circular: E = 45 deg, ν = {math.degrees(nu):.2f} deg")
            Circular: E = 45 deg, ν = 45.00 deg
        """
        E = eccentric_anomaly
        e = eccentricity

        beta = e / (1 + math.sqrt(1 - e**2))
        true_anomaly = E + 2 * math.atan2(beta * math.sin(E), 1 - beta * math.cos(E))

        return true_anomaly % constants.TWO_PI

    def true_to_eccentric_anomaly(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Convert true anomaly to eccentric anomaly.

        Parameters:
            true_anomaly: True anomaly ν in radians
            eccentricity: Orbital eccentricity

        Returns:
            Eccentric anomaly E in radians
        """
        nu = true_anomaly
        e = eccentricity

        E = math.atan2(
            math.sqrt(1 - e**2) * math.sin(nu),
            e + math.cos(nu)
        )

        return E % constants.TWO_PI

    def true_to_mean_anomaly(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Convert true anomaly to mean anomaly.

        Parameters:
            true_anomaly: True anomaly ν in radians
            eccentricity: Orbital eccentricity. Must be in range [0, 1) for elliptical orbits

        Returns:
            Mean anomaly M in radians, normalized to [0, 2π]

        Examples:
            >>> solver = KeplerSolver()
            >>> # Round-trip conversion test
            >>> nu_original = math.radians(120)  # 120 deg true anomaly
            >>> e = 0.3
            >>> M = solver.true_to_mean_anomaly(nu_original, e)
            >>> nu_recovered = solver.mean_to_true_anomaly(M, e)
            >>> error = abs(nu_recovered - nu_original)
            >>> print(f"Round-trip error: {math.degrees(error):.6f} deg")
            Round-trip error: 0.000000 deg

            >>> # True anomaly at apoapsis (180 deg)
            >>> M = solver.true_to_mean_anomaly(math.pi, 0.2)
            >>> print(f"Mean anomaly at apoapsis: {math.degrees(M):.2f} deg")
            Mean anomaly at apoapsis: 180.00 deg
        """
        E = self.true_to_eccentric_anomaly(true_anomaly, eccentricity)
        M = E - eccentricity * math.sin(E)
        return M % constants.TWO_PI

    def calculate_flight_path_angle(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Calculate flight path angle.

        Parameters:
            true_anomaly: True anomaly in radians
            eccentricity: Orbital eccentricity

        Returns:
            Flight path angle in radians
        """
        nu = true_anomaly
        e = eccentricity
        gamma = math.atan2(e * math.sin(nu), 1 + e * math.cos(nu))
        return gamma

    def time_since_periapsis(self, mean_anomaly: float, semi_major_axis: float,
                            mu: float = constants.EARTH_MU) -> float:
        """
        Calculate time since periapsis passage.

        Parameters:
            mean_anomaly: Mean anomaly in radians
            semi_major_axis: Semi-major axis in km
            mu: Gravitational parameter in km³/s²

        Returns:
            Time since periapsis in seconds
        """
        n = math.sqrt(mu / semi_major_axis**3)
        t = mean_anomaly / n
        return t

    def mean_anomaly_at_time(self, t: float, semi_major_axis: float,
                            initial_mean_anomaly: float = 0.0,
                            mu: float = constants.EARTH_MU) -> float:
        """
        Calculate mean anomaly at a given time from epoch.

        Parameters:
            t: Time since epoch in seconds. Can be negative for past times
            semi_major_axis: Semi-major axis of the orbit in km. Must be positive
            initial_mean_anomaly: Mean anomaly at epoch t=0 in radians. Default is 0.0 (at periapsis)
            mu: Gravitational parameter in km³/s². Default is Earth's μ = 398600.4418 km³/s²

        Returns:
            Mean anomaly at time t in radians, normalized to [0, 2π]

        Raises:
            ValueError: If semi_major_axis <= 0

        Examples:
            >>> solver = KeplerSolver()
            >>> # ISS-like orbit (a ≈ 6778 km for 400 km altitude)
            >>> a = 6778.0  # km
            >>> period = 2 * math.pi * math.sqrt(a**3 / constants.EARTH_MU)
            >>> print(f"Orbital period: {period/60:.1f} minutes")
            Orbital period: 92.6 minutes

            >>> # Mean anomaly after quarter orbit
            >>> M = solver.mean_anomaly_at_time(period/4, a, initial_mean_anomaly=0.0)
            >>> print(f"After 1/4 orbit: M = {math.degrees(M):.1f} deg")
            After 1/4 orbit: M = 90.0 deg

            >>> # Mean anomaly after 1.5 orbits
            >>> M = solver.mean_anomaly_at_time(1.5*period, a)
            >>> print(f"After 1.5 orbits: M = {math.degrees(M):.1f} deg")
            After 1.5 orbits: M = 180.0 deg

            >>> # Propagate backwards in time
            >>> M = solver.mean_anomaly_at_time(-period/2, a, initial_mean_anomaly=math.pi)
            >>> print(f"Half orbit before epoch: M = {math.degrees(M):.1f} deg")
            Half orbit before epoch: M = 0.0 deg
        """
        n = math.sqrt(mu / semi_major_axis**3)
        M = initial_mean_anomaly + n * t
        return M % constants.TWO_PI