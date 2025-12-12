"""
Output generator for orbital propagation results.
Handles formatting and exporting results in various formats.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from orbital_elements import OrbitalElements
from propagator import PropagationState
import constants


class OutputGenerator:
    def __init__(self):
        """Initialize output generator."""
        self.supported_formats = ['txt', 'csv', 'json']
    
    def generate_report(self, initial_elements: OrbitalElements,
                       final_state: PropagationState,
                       propagation_time: float,
                       output_file: str = "orbit_output.txt",
                       format: str = "txt") -> None:
        """
        Generate output report in specified format.
        
        Args:
            initial_elements: Initial orbital elements
            final_state: Final propagation state
            propagation_time: Total propagation time in seconds
            output_file: Output file path
            format: Output format ('txt', 'csv', or 'json')
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Use one of {self.supported_formats}")
        
        if format == 'txt':
            self._write_text_report(initial_elements, final_state, propagation_time, output_file)
        elif format == 'csv':
            self._write_csv_report(initial_elements, final_state, propagation_time, output_file)
        elif format == 'json':
            self._write_json_report(initial_elements, final_state, propagation_time, output_file)
    
    def _write_text_report(self, initial_elements: OrbitalElements,
                          final_state: PropagationState,
                          propagation_time: float,
                          output_file: str) -> None:
        """
        Write human-readable text report
        
        Args:
            initial_elements: Initial orbital elements
            final_state: Final propagation state
            propagation_time: Total propagation time in seconds
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ORBIT PROPAGATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Propagation summary
            f.write("PROPAGATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start Epoch: {initial_elements.epoch.isoformat()}\n")
            f.write(f"End Epoch:   {final_state.time.isoformat()}\n")
            f.write(f"Duration:    {propagation_time:.1f} seconds ({propagation_time/3600:.2f} hours)\n")
            f.write(f"Orbits:      {propagation_time / initial_elements.period:.2f}\n" if initial_elements.period else "")
            f.write("\n")
            
            # Initial orbital elements
            f.write("INITIAL ORBITAL ELEMENTS\n")
            f.write("-" * 40 + "\n")
            f.write(self._format_orbital_elements(initial_elements))
            f.write("\n")
            
            # Final orbital elements
            f.write("FINAL ORBITAL ELEMENTS\n")
            f.write("-" * 40 + "\n")
            f.write(self._format_orbital_elements(final_state.orbital_elements))
            f.write("\n")
            
            # Final state vectors
            f.write("FINAL STATE VECTORS (ECI)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Position [km]:\n")
            f.write(f"  X: {final_state.position[0]:12.3f}\n")
            f.write(f"  Y: {final_state.position[1]:12.3f}\n")
            f.write(f"  Z: {final_state.position[2]:12.3f}\n")
            f.write(f"  Magnitude: {self._magnitude(final_state.position):12.3f}\n\n")
            
            f.write(f"Velocity [km/s]:\n")
            f.write(f"  VX: {final_state.velocity[0]:12.6f}\n")
            f.write(f"  VY: {final_state.velocity[1]:12.6f}\n")
            f.write(f"  VZ: {final_state.velocity[2]:12.6f}\n")
            f.write(f"  Magnitude: {self._magnitude(final_state.velocity):12.6f}\n\n")
            
            # Orbit characteristics
            f.write("ORBIT CHARACTERISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Orbit Type:     {initial_elements.orbit_type}\n")
            f.write(f"Altitude (peri): {initial_elements.perigee - constants.EARTH_RADIUS:.1f} km\n")
            if initial_elements.apogee:
                f.write(f"Altitude (apo):  {initial_elements.apogee - constants.EARTH_RADIUS:.1f} km\n")
            if initial_elements.period:
                f.write(f"Orbital Period:  {initial_elements.period/3600:.2f} hours\n")
                f.write(f"Mean Motion:     {initial_elements.mean_motion * 86400 / constants.TWO_PI:.4f} rev/day\n")
            f.write("\n")
            
            # Footer
            f.write("=" * 60 + "\n")
            f.write(f"Report generated: {datetime.now().isoformat()}\n")
            f.write("APC 524 Orbit Propagation Tool\n")
    
    def _write_csv_report(self, initial_elements: OrbitalElements,
                         final_state: PropagationState,
                         propagation_time: float,
                         output_file: str) -> None:
        """
        Write CSV format report.
        
        Args:
            initial_elements: Initial orbital elements
            final_state: Final propagation state
            propagation_time: Total propagation time in seconds
            output_file: Output file path
        """
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Parameter', 'Initial', 'Final', 'Units'])
            
            # Epoch
            writer.writerow(['Epoch', 
                           initial_elements.epoch.isoformat(),
                           final_state.time.isoformat(),
                           'UTC'])
            
            # Orbital elements
            writer.writerow(['Semi-major axis',
                           f'{initial_elements.a:.3f}',
                           f'{final_state.orbital_elements.a:.3f}',
                           'km'])
            
            writer.writerow(['Eccentricity',
                           f'{initial_elements.e:.6f}',
                           f'{final_state.orbital_elements.e:.6f}',
                           '-'])
            
            writer.writerow(['Inclination',
                           f'{initial_elements.i * constants.RAD_TO_DEG:.3f}',
                           f'{final_state.orbital_elements.i * constants.RAD_TO_DEG:.3f}',
                           'degrees'])
            
            writer.writerow(['RAAN',
                           f'{initial_elements.raan * constants.RAD_TO_DEG:.3f}',
                           f'{final_state.orbital_elements.raan * constants.RAD_TO_DEG:.3f}',
                           'degrees'])
            
            writer.writerow(['Arg of Perigee',
                           f'{initial_elements.arg_perigee * constants.RAD_TO_DEG:.3f}',
                           f'{final_state.orbital_elements.arg_perigee * constants.RAD_TO_DEG:.3f}',
                           'degrees'])
            
            writer.writerow(['Mean Anomaly',
                           f'{initial_elements.mean_anomaly * constants.RAD_TO_DEG:.3f}',
                           f'{final_state.orbital_elements.mean_anomaly * constants.RAD_TO_DEG:.3f}',
                           'degrees'])
            
            # State vectors
            writer.writerow([])
            writer.writerow(['State Vectors', 'X', 'Y', 'Z', 'Magnitude'])
            writer.writerow(['Position [km]',
                           f'{final_state.position[0]:.3f}',
                           f'{final_state.position[1]:.3f}',
                           f'{final_state.position[2]:.3f}',
                           f'{self._magnitude(final_state.position):.3f}'])
            
            writer.writerow(['Velocity [km/s]',
                           f'{final_state.velocity[0]:.6f}',
                           f'{final_state.velocity[1]:.6f}',
                           f'{final_state.velocity[2]:.6f}',
                           f'{self._magnitude(final_state.velocity):.6f}'])
            
            # Propagation info
            writer.writerow([])
            writer.writerow(['Propagation Time', f'{propagation_time:.1f}', 'seconds'])
            writer.writerow(['Propagation Time', f'{propagation_time/3600:.2f}', 'hours'])
            if initial_elements.period:
                writer.writerow(['Number of Orbits', f'{propagation_time/initial_elements.period:.2f}', '-'])
    
    def _write_json_report(self, initial_elements: OrbitalElements,
                          final_state: PropagationState,
                          propagation_time: float,
                          output_file: str) -> None:
        """
        Write JSON format report.
        
        Args:
            initial_elements: Initial orbital elements
            final_state: Final propagation state
            propagation_time: Total propagation time in seconds
            output_file: Output file path
        """
        report = {
            'propagation_summary': {
                'start_epoch': initial_elements.epoch.isoformat(),
                'end_epoch': final_state.time.isoformat(),
                'duration_seconds': propagation_time,
                'duration_hours': propagation_time / 3600,
                'number_of_orbits': propagation_time / initial_elements.period if initial_elements.period else None
            },
            'initial_state': {
                'orbital_elements': initial_elements.to_dict()
            },
            'final_state': final_state.to_dict(),
            'orbit_characteristics': {
                'orbit_type': initial_elements.orbit_type,
                'perigee_altitude_km': initial_elements.perigee - constants.EARTH_RADIUS,
                'apogee_altitude_km': initial_elements.apogee - constants.EARTH_RADIUS if initial_elements.apogee else None,
                'period_hours': initial_elements.period / 3600 if initial_elements.period else None,
                'mean_motion_rev_per_day': initial_elements.mean_motion * 86400 / constants.TWO_PI
            },
            'metadata': {
                'generated': datetime.now().isoformat(),
                'tool': 'APC 524 Orbit Propagation Tool',
                'version': '1.0.0'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def export_ground_track(self, ground_track: List[tuple], 
                           output_file: str = "ground_track.csv") -> None:
        """
        Export ground track to CSV file.
        
        Args:
            ground_track: List of (latitude, longitude) tuples
            output_file: Output file path
        """
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Point', 'Latitude (deg)', 'Longitude (deg)'])
            
            for i, (lat, lon) in enumerate(ground_track):
                writer.writerow([i, f'{lat:.4f}', f'{lon:.4f}'])
        
        print(f"Ground track exported to {output_file}")
    
    def export_state_history(self, states: List[PropagationState],
                           output_file: str = "state_history.csv") -> None:
        """
        Export propagation state history to CSV.
        
        Args:
            states: List of propagation states
            output_file: Output file path
        """
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'Time (UTC)', 'Elapsed (s)',
                'X (km)', 'Y (km)', 'Z (km)',
                'VX (km/s)', 'VY (km/s)', 'VZ (km/s)',
                'Mean Anomaly (deg)', 'Radius (km)'
            ])
            
            start_time = states[0].time if states else datetime.now()
            
            for state in states:
                elapsed = (state.time - start_time).total_seconds()
                radius = self._magnitude(state.position)
                
                writer.writerow([
                    state.time.isoformat(),
                    f'{elapsed:.1f}',
                    f'{state.position[0]:.3f}',
                    f'{state.position[1]:.3f}',
                    f'{state.position[2]:.3f}',
                    f'{state.velocity[0]:.6f}',
                    f'{state.velocity[1]:.6f}',
                    f'{state.velocity[2]:.6f}',
                    f'{state.orbital_elements.mean_anomaly * constants.RAD_TO_DEG:.3f}',
                    f'{radius:.3f}'
                ])
        
        print(f"State history exported to {output_file}")
    
    def _format_orbital_elements(self, elements: OrbitalElements) -> str:
        """
        Format orbital elements for text output.
        
        Args:
            elements: Orbital elements to format
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Semi-major axis:     {elements.a:12.3f} km")
        lines.append(f"Eccentricity:        {elements.e:12.6f}")
        lines.append(f"Inclination:         {elements.i * constants.RAD_TO_DEG:12.3f} degrees")
        lines.append(f"RAAN:                {elements.raan * constants.RAD_TO_DEG:12.3f} degrees")
        lines.append(f"Argument of Perigee: {elements.arg_perigee * constants.RAD_TO_DEG:12.3f} degrees")
        lines.append(f"Mean Anomaly:        {elements.mean_anomaly * constants.RAD_TO_DEG:12.3f} degrees")
        lines.append(f"Epoch:               {elements.epoch.isoformat()}")
        
        return '\n'.join(lines)
    
    def _magnitude(self, vector) -> float:
        """Calculate magnitude of a vector."""
        import numpy as np
        return np.linalg.norm(vector)
    
    def create_kml_ground_track(self, ground_track: List[tuple],
                               satellite_name: str = "Satellite",
                               output_file: str = "ground_track.kml") -> None:
        """
        Create KML file for Google Earth visualization.
        
        Args:
            ground_track: List of (latitude, longitude) tuples
            satellite_name: Name of the satellite
            output_file: Output KML file path
        """
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>{satellite_name} Ground Track</name>
    <Style id="groundTrack">
        <LineStyle>
            <color>ff0000ff</color>
            <width>2</width>
        </LineStyle>
    </Style>
    <Placemark>
        <name>{satellite_name} Orbit</name>
        <styleUrl>#groundTrack</styleUrl>
        <LineString>
            <coordinates>
"""
        
        # Add coordinates
        for lat, lon in ground_track:
            kml_content += f"                {lon:.6f},{lat:.6f},0\n"
        
        kml_content += """            </coordinates>
        </LineString>
    </Placemark>
</Document>
</kml>"""
        
        with open(output_file, 'w') as f:
            f.write(kml_content)
        
        print(f"KML ground track exported to {output_file}")


def test_output_generator():
    """Test the output generator."""
    import math
    from datetime import datetime
    
    print("Testing Output Generator...")
    print("-" * 40)
    
    # Create sample data
    initial_elements = OrbitalElements(
        a=7000.0,
        e=0.001,
        i=math.radians(45.0),
        raan=math.radians(30.0),
        arg_perigee=math.radians(60.0),
        mean_anomaly=math.radians(0.0),
        epoch=datetime(2025, 1, 15, 12, 0, 0)
    )
    
    # Create a simple final state
    import numpy as np
    final_state = PropagationState(
        time=datetime(2025, 1, 15, 13, 30, 0),
        position=np.array([6500.0, 2000.0, 1000.0]),
        velocity=np.array([1.5, 7.0, 0.5]),
        orbital_elements=OrbitalElements(
            a=7000.0,
            e=0.001,
            i=math.radians(45.0),
            raan=math.radians(30.0),
            arg_perigee=math.radians(60.0),
            mean_anomaly=math.radians(97.0),  # Propagated
            epoch=datetime(2025, 1, 15, 13, 30, 0)
        )
    )
    
    # Test output generation
    output_gen = OutputGenerator()
    
    print("Generating text report...")
    output_gen.generate_report(
        initial_elements, final_state, 5400.0,
        "test_output.txt", "txt"
    )
    print("  Created: test_output.txt")
    
    print("\nGenerating CSV report...")
    output_gen.generate_report(
        initial_elements, final_state, 5400.0,
        "test_output.csv", "csv"
    )
    print("  Created: test_output.csv")
    
    print("\nGenerating JSON report...")
    output_gen.generate_report(
        initial_elements, final_state, 5400.0,
        "test_output.json", "json"
    )
    print("  Created: test_output.json")
    
    # Test ground track export
    print("\nGenerating sample ground track...")
    ground_track = [
        (0.0, 0.0),
        (10.0, -15.0),
        (20.0, -30.0),
        (25.0, -45.0),
        (20.0, -60.0)
    ]
    output_gen.export_ground_track(ground_track, "test_ground_track.csv")
    
    print("-" * 40)
    print("Tests complete!")


if __name__ == "__main__":
    test_output_generator()
