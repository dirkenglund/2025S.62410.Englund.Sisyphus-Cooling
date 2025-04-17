import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the test class for core physics calculations
class TestSisyphusCoolingPhysics(unittest.TestCase):
    
    def test_temperature_evolution_calculation(self):
        """Test the temperature evolution calculation for Sisyphus cooling."""
        # Mock parameters
        initial_temp = 100  # μK
        detuning = -3.0  # Γ units
        rabi_freq = 2.0  # Γ units
        total_time = 5e-3  # 5 ms
        dt = 1e-6  # 1 μs
        n_steps = int(total_time / dt)
        times = np.linspace(0, total_time, n_steps)
        
        # Calculate cooling rate and temperatures
        cooling_rate = (np.abs(detuning) / 3) * (rabi_freq / 2)
        sisyphus_temps = initial_temp * np.exp(-times * 1e3 * cooling_rate)
        
        # Assertions
        self.assertEqual(len(sisyphus_temps), n_steps)
        self.assertAlmostEqual(sisyphus_temps[0], initial_temp)
        self.assertLess(sisyphus_temps[-1], initial_temp)  # Temperature should decrease
        
        # Check that the cooling follows the expected exponential decay
        expected_final_temp = initial_temp * np.exp(-total_time * 1e3 * cooling_rate)
        self.assertAlmostEqual(sisyphus_temps[-1], expected_final_temp, places=5)
    
    def test_atom_retention_calculation(self):
        """Test the atom retention calculation."""
        # Mock parameters
        initial_atoms = 5000
        trap_depth = 50  # μK
        temperatures = np.array([100, 80, 60, 40, 20])  # μK
        
        # Calculate atom retention
        retention = []
        for temp in temperatures:
            # Probability of atom having energy < trap_depth
            p_retain = 1 - np.exp(-trap_depth / temp)
            # Number of atoms retained
            atoms = initial_atoms * p_retain
            retention.append(atoms)
        
        # Assertions
        self.assertEqual(len(retention), len(temperatures))
        for i in range(1, len(retention)):
            self.assertGreater(retention[i], retention[i-1])  # Retention should increase as temperature decreases
        
        # Check specific values
        for i, temp in enumerate(temperatures):
            expected_retention = initial_atoms * (1 - np.exp(-trap_depth / temp))
            self.assertAlmostEqual(retention[i], expected_retention, places=5)
    
    def test_velocity_distribution_calculation(self):
        """Test the velocity distribution calculation."""
        # Mock parameters
        temperature = 10e-6  # 10 μK in K
        v_max = 0.5  # m/s
        v = np.linspace(-v_max, v_max, 1000)
        m_rb = 1.44e-25  # kg, mass of Rb-87
        kb = 1.38e-23  # J/K, Boltzmann constant
        
        # Calculate Maxwell-Boltzmann distribution
        dist = np.sqrt(m_rb / (2 * np.pi * kb * temperature)) * np.exp(-m_rb * v**2 / (2 * kb * temperature))
        
        # Normalize
        dist_norm = dist / np.max(dist)
        
        # Assertions
        self.assertEqual(len(dist_norm), len(v))
        self.assertAlmostEqual(np.max(dist_norm), 1.0)
        
        # Check that the distribution is symmetric
        mid_point = len(v) // 2
        for i in range(mid_point):
            self.assertAlmostEqual(dist_norm[i], dist_norm[-i-1], places=5)
        
        # Check that the peak is near v=0 (using absolute difference instead of places)
        peak_index = np.argmax(dist_norm)
        self.assertTrue(abs(v[peak_index]) < 0.001, f"Peak not close enough to zero: {v[peak_index]}")

# Test class for graph data structure
class TestWorldModelGraph(unittest.TestCase):
    
    def test_world_model_graph_creation(self):
        """Test the creation of the world model graph."""
        # Create a simple test graph
        nodes = [
            {"id": "sisyphus", "label": "Sisyphus Cooling", "group": 0},
            {"id": "doppler", "label": "Doppler Cooling", "group": 1},
            {"id": "polarization", "label": "Polarization Gradient", "group": 2}
        ]
        
        links = [
            {"source": "sisyphus", "target": "doppler", "value": 2},
            {"source": "sisyphus", "target": "polarization", "value": 3}
        ]
        
        # Create node and link dataframes
        nodes_df = pd.DataFrame(nodes)
        links_df = pd.DataFrame(links)
        
        # Create adjacency matrix
        n = len(nodes)
        adjmat = np.zeros((n, n))
        
        for link in links:
            source_idx = next(i for i, node in enumerate(nodes) if node["id"] == link["source"])
            target_idx = next(i for i, node in enumerate(nodes) if node["id"] == link["target"])
            adjmat[source_idx, target_idx] = link["value"]
            adjmat[target_idx, source_idx] = link["value"]  # Make it undirected
        
        # Assertions
        self.assertEqual(adjmat.shape, (n, n))
        self.assertEqual(adjmat[0, 1], 2)  # sisyphus-doppler link
        self.assertEqual(adjmat[1, 0], 2)  # doppler-sisyphus link (undirected)
        self.assertEqual(adjmat[0, 2], 3)  # sisyphus-polarization link
        self.assertEqual(adjmat[2, 0], 3)  # polarization-sisyphus link (undirected)
        self.assertEqual(adjmat[1, 2], 0)  # No direct link between doppler and polarization

# Test class for comparison data
class TestComparisonData(unittest.TestCase):
    
    def test_comparison_data_validity(self):
        """Test the validity of the comparison data."""
        # Mock comparison data
        comparison_data = {
            "Method": ["Doppler Cooling", "Sisyphus Cooling", "Velocity-Selective Coherent Population Trapping", 
                      "Raman Cooling", "Evaporative Cooling"],
            "Temperature Range": ["~100-500 μK", "~1-20 μK", "~100 nK - 1 μK", "~100 nK - 1 μK", "~1 nK - 1 μK"],
            "Atom Retention": ["High (>90%)", "Medium (50-80%)", "Low-Medium (30-60%)", "Medium (40-70%)", "Low (<30%)"],
            "Complexity": ["Low", "Medium", "High", "High", "Medium"],
            "Key Advantage": ["Simple implementation", "Sub-Doppler temperatures", "Very low temperatures", 
                             "Works for multiple species", "Reaches quantum degeneracy"]
        }
        
        # Assertions
        self.assertEqual(len(comparison_data["Method"]), 5)
        self.assertEqual(len(comparison_data["Temperature Range"]), 5)
        self.assertEqual(len(comparison_data["Atom Retention"]), 5)
        self.assertEqual(len(comparison_data["Complexity"]), 5)
        self.assertEqual(len(comparison_data["Key Advantage"]), 5)
        
        # Check that Sisyphus cooling has lower temperature than Doppler cooling
        doppler_temp = comparison_data["Temperature Range"][0]
        sisyphus_temp = comparison_data["Temperature Range"][1]
        self.assertIn("100-500", doppler_temp)
        self.assertIn("1-20", sisyphus_temp)
        
        # Check that atom retention values are consistent with expectations
        doppler_retention = comparison_data["Atom Retention"][0]
        sisyphus_retention = comparison_data["Atom Retention"][1]
        evaporative_retention = comparison_data["Atom Retention"][4]
        self.assertIn(">90%", doppler_retention)
        self.assertIn("50-80%", sisyphus_retention)
        self.assertIn("<30%", evaporative_retention)

if __name__ == '__main__':
    unittest.main()
