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
        
        Uses Newton-Raphson method for elliptical orbits (e < 1).
        
        Args:
            mean_anomaly: Mean anomaly M in radians
            eccentricity: Orbital eccentricity e
            
        Returns:
            float: Eccentric anomaly E in radians
            
        Raises:
            ValueError: If eccentricity is invalid or convergence fails
        """
        if eccentricity < 0 or eccentricity >= 1:
            raise ValueError(f"This solver only handles elliptical orbits (0 <= e < 1), got e={eccentricity}")
        
        # Normalize mean anomaly to [0, 2π]
        M = mean_anomaly % constants.TWO_PI
        
        # Initial guess for eccentric anomaly
        # Use a better initial guess based on eccentricity
        if eccentricity < 0.8:
            E = M  # Good for low eccentricity
        else:
            E = math.pi  # Better for high eccentricity
        
        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Calculate f(E) = E - e*sin(E) - M
            f = E - eccentricity * math.sin(E) - M
            
            # Calculate f'(E) = 1 - e*cos(E)
            f_prime = 1 - eccentricity * math.cos(E)
            
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                f_prime = 1e-12
            
            # Newton-Raphson step
            E_new = E - f / f_prime
            
            # Check convergence
            if abs(E_new - E) < self.tolerance:
                return E_new % constants.TWO_PI
            
            E = E_new
        
        # If we reach here, convergence failed
        raise ValueError(f"Kepler equation did not converge after {self.max_iterations} iterations")
    
    def mean_to_true_anomaly(self, mean_anomaly: float, eccentricity: float) -> float:
        """
        Convert mean anomaly to true anomaly.
        
        Args:
            mean_anomaly: Mean anomaly in radians
            eccentricity: Orbital eccentricity
            
        Returns:
            float: True anomaly in radians
        """
        # First, get eccentric anomaly
        E = self.solve_kepler_equation(mean_anomaly, eccentricity)
        
        # Convert eccentric anomaly to true anomaly
        true_anomaly = self.eccentric_to_true_anomaly(E, eccentricity)
        
        return true_anomaly
    
    def eccentric_to_true_anomaly(self, eccentric_anomaly: float, eccentricity: float) -> float:
        """
        Convert eccentric anomaly to true anomaly.
        
        Uses the relation:
        tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        
        Args:
            eccentric_anomaly: Eccentric anomaly E in radians
            eccentricity: Orbital eccentricity
            
        Returns:
            float: True anomaly ν in radians
        """
        E = eccentric_anomaly
        e = eccentricity
        
        # Method 1: Using half-angle formula (more stable)
        beta = e / (1 + math.sqrt(1 - e**2))
        true_anomaly = E + 2 * math.atan2(beta * math.sin(E), 1 - beta * math.cos(E))
        
        return true_anomaly % constants.TWO_PI
    
    def true_to_eccentric_anomaly(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Convert true anomaly to eccentric anomaly.
        
        Args:
            true_anomaly: True anomaly ν in radians
            eccentricity: Orbital eccentricity
            
        Returns:
            float: Eccentric anomaly E in radians
        """
        nu = true_anomaly
        e = eccentricity
        
        # Using atan2 for correct quadrant
        E = math.atan2(
            math.sqrt(1 - e**2) * math.sin(nu),
            e + math.cos(nu)
        )
        
        return E % constants.TWO_PI
    
    def true_to_mean_anomaly(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Convert true anomaly to mean anomaly.
        
        Args:
            true_anomaly: True anomaly in radians
            eccentricity: Orbital eccentricity
            
        Returns:
            float: Mean anomaly in radians
        """
        # First convert to eccentric anomaly
        E = self.true_to_eccentric_anomaly(true_anomaly, eccentricity)
        
        # Then convert to mean anomaly using Kepler's equation
        M = E - eccentricity * math.sin(E)
        
        return M % constants.TWO_PI
    
    def calculate_flight_path_angle(self, true_anomaly: float, eccentricity: float) -> float:
        """
        Calculate the flight path angle (angle between velocity and local horizon).
        
        Args:
            true_anomaly: True anomaly in radians
            eccentricity: Orbital eccentricity
            
        Returns:
            float: Flight path angle in radians
        """
        nu = true_anomaly
        e = eccentricity
        
        # Flight path angle formula
        gamma = math.atan2(e * math.sin(nu), 1 + e * math.cos(nu))
        
        return gamma
    
    def time_since_periapsis(self, mean_anomaly: float, semi_major_axis: float,
                            mu: float = constants.EARTH_MU) -> float:
        """
        Calculate time since periapsis passage.
        
        Args:
            mean_anomaly: Mean anomaly in radians
            semi_major_axis: Semi-major axis in km
            mu: Gravitational parameter in km³/s²
            
        Returns:
            float: Time since periapsis in seconds
        """
        # Mean motion
        n = math.sqrt(mu / semi_major_axis**3)
        
        # Time since periapsis
        t = mean_anomaly / n
        
        return t
    
    def mean_anomaly_at_time(self, t: float, semi_major_axis: float,
                            initial_mean_anomaly: float = 0.0,
                            mu: float = constants.EARTH_MU) -> float:
        """
        Calculate mean anomaly at a given time from epoch.
        
        Args:
            t: Time since epoch in seconds
            semi_major_axis: Semi-major axis in km
            initial_mean_anomaly: Mean anomaly at epoch in radians
            mu: Gravitational parameter in km³/s²
            
        Returns:
            float: Mean anomaly at time t in radians
        """
        # Mean motion
        n = math.sqrt(mu / semi_major_axis**3)
        
        # Propagated mean anomaly
        M = initial_mean_anomaly + n * t
        
        return M % constants.TWO_PI


def test_kepler_solver():
    """Test the Kepler solver with some known cases."""
    solver = KeplerSolver()
    
    print("Testing Kepler Solver...")
    print("-" * 40)
    
    # Test case 1: Circular orbit (e = 0)
    print("Test 1: Circular orbit (e = 0)")
    M = math.pi / 2  # 90 degrees
    e = 0.0
    E = solver.solve_kepler_equation(M, e)
    print(f"  M = {math.degrees(M):.2f}°, E = {math.degrees(E):.2f}°")
    print(f"  Expected E = M for circular orbit: {abs(E - M) < 1e-10}")
    
    # Test case 2: Low eccentricity
    print("\nTest 2: Low eccentricity (e = 0.1)")
    M = math.pi / 3  # 60 degrees
    e = 0.1
    E = solver.solve_kepler_equation(M, e)
    nu = solver.mean_to_true_anomaly(M, e)
    print(f"  M = {math.degrees(M):.2f}°")
    print(f"  E = {math.degrees(E):.2f}°")
    print(f"  ν = {math.degrees(nu):.2f}°")
    
    # Test case 3: High eccentricity
    print("\nTest 3: High eccentricity (e = 0.8)")
    M = math.pi  # 180 degrees
    e = 0.8
    E = solver.solve_kepler_equation(M, e)
    nu = solver.mean_to_true_anomaly(M, e)
    print(f"  M = {math.degrees(M):.2f}°")
    print(f"  E = {math.degrees(E):.2f}°")
    print(f"  ν = {math.degrees(nu):.2f}°")
    
    # Test case 4: Round-trip conversion
    print("\nTest 4: Round-trip conversion")
    nu_original = 2.0  # radians
    e = 0.3
    M = solver.true_to_mean_anomaly(nu_original, e)
    nu_recovered = solver.mean_to_true_anomaly(M, e)
    print(f"  Original ν = {math.degrees(nu_original):.4f}°")
    print(f"  Recovered ν = {math.degrees(nu_recovered):.4f}°")
    print(f"  Error = {math.degrees(abs(nu_recovered - nu_original)):.6f}°")
    
    print("-" * 40)
    print("Tests complete!")


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_kepler_solver()
