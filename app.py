from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import json
import os
from scipy.constants import hbar, k as k_B, atomic_mass, c
import sys
import time

# Physical constants for Rubidium-87
RB87_MASS = 86.909180527 * atomic_mass  # kg
WAVELENGTH_D2 = 780.241e-9  # m (D2 transition wavelength)
WAVENUMBER_D2 = 2 * np.pi / WAVELENGTH_D2  # m^-1
GAMMA_D2 = 2 * np.pi * 6.065e6  # s^-1 (natural linewidth of D2 transition)
RECOIL_ENERGY = (hbar * WAVENUMBER_D2)**2 / (2 * RB87_MASS)  # J
RECOIL_TEMPERATURE = RECOIL_ENERGY / k_B  # K
DOPPLER_TEMPERATURE = hbar * GAMMA_D2 / (2 * k_B)  # K

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    """Serve the main HTML interface."""
    with open('sisyphus_cooling_web_interface.html', 'r') as f:
        content = f.read()
    return content

@app.route('/api/constants')
def get_constants():
    """Return the physical constants used in the simulation."""
    return jsonify({
        'RB87_MASS': RB87_MASS,
        'WAVELENGTH_D2': WAVELENGTH_D2,
        'WAVENUMBER_D2': WAVENUMBER_D2,
        'GAMMA_D2': GAMMA_D2,
        'RECOIL_ENERGY': RECOIL_ENERGY,
        'RECOIL_TEMPERATURE': RECOIL_TEMPERATURE,
        'DOPPLER_TEMPERATURE': DOPPLER_TEMPERATURE
    })

@app.route('/api/world-model')
def get_world_model():
    """Return the world model data for visualization."""
    # Nodes representing physical quantities, equations, and concepts
    nodes = [
        {'id': 'RB87_MASS', 'label': 'Rb-87 Mass', 'type': 'variable'},
        {'id': 'WAVELENGTH_D2', 'label': 'D2 Wavelength', 'type': 'variable'},
        {'id': 'WAVENUMBER_D2', 'label': 'D2 Wavenumber', 'type': 'variable'},
        {'id': 'GAMMA_D2', 'label': 'D2 Linewidth', 'type': 'variable'},
        {'id': 'RECOIL_ENERGY', 'label': 'Recoil Energy', 'type': 'variable'},
        {'id': 'RECOIL_TEMPERATURE', 'label': 'Recoil Temperature', 'type': 'variable'},
        {'id': 'DOPPLER_TEMPERATURE', 'label': 'Doppler Temperature', 'type': 'variable'},
        {'id': 'WAVENUMBER_EQUATION', 'label': 'k = 2π/λ', 'type': 'equation'},
        {'id': 'RECOIL_ENERGY_EQUATION', 'label': 'E_r = (ħk)²/(2m)', 'type': 'equation'},
        {'id': 'RECOIL_TEMPERATURE_EQUATION', 'label': 'T_r = E_r/k_B', 'type': 'equation'},
        {'id': 'DOPPLER_TEMPERATURE_EQUATION', 'label': 'T_D = ħΓ/(2k_B)', 'type': 'equation'},
        {'id': 'Sisyphus_Cooling', 'label': 'Sisyphus Cooling', 'type': 'concept'},
        {'id': 'Optical_Lattice', 'label': 'Optical Lattice', 'type': 'concept'},
        {'id': 'Polarization_Gradient', 'label': 'Polarization Gradient', 'type': 'concept'},
        {'id': 'Optical_Pumping', 'label': 'Optical Pumping', 'type': 'concept'},
        {'id': 'D2_Transition', 'label': 'D2 Transition', 'type': 'concept'}
    ]
    
    # Links representing relationships between nodes
    links = [
        {'source': 'WAVENUMBER_EQUATION', 'target': 'WAVENUMBER_D2'},
        {'source': 'WAVENUMBER_EQUATION', 'target': 'WAVELENGTH_D2'},
        {'source': 'RECOIL_ENERGY_EQUATION', 'target': 'RECOIL_ENERGY'},
        {'source': 'RECOIL_ENERGY_EQUATION', 'target': 'WAVENUMBER_D2'},
        {'source': 'RECOIL_ENERGY_EQUATION', 'target': 'RB87_MASS'},
        {'source': 'RECOIL_TEMPERATURE_EQUATION', 'target': 'RECOIL_TEMPERATURE'},
        {'source': 'RECOIL_TEMPERATURE_EQUATION', 'target': 'RECOIL_ENERGY'},
        {'source': 'DOPPLER_TEMPERATURE_EQUATION', 'target': 'DOPPLER_TEMPERATURE'},
        {'source': 'DOPPLER_TEMPERATURE_EQUATION', 'target': 'GAMMA_D2'},
        {'source': 'Sisyphus_Cooling', 'target': 'RECOIL_TEMPERATURE'},
        {'source': 'Sisyphus_Cooling', 'target': 'DOPPLER_TEMPERATURE'},
        {'source': 'Optical_Lattice', 'target': 'WAVELENGTH_D2'},
        {'source': 'Optical_Lattice', 'target': 'WAVENUMBER_D2'},
        {'source': 'Polarization_Gradient', 'target': 'Optical_Lattice'},
        {'source': 'Optical_Pumping', 'target': 'GAMMA_D2'},
        {'source': 'Optical_Pumping', 'target': 'Sisyphus_Cooling'},
        {'source': 'D2_Transition', 'target': 'WAVELENGTH_D2'},
        {'source': 'D2_Transition', 'target': 'GAMMA_D2'},
    ]
    
    return jsonify({
        'nodes': nodes,
        'links': links
    })

@app.route('/api/lattice-potential')
def get_lattice_potential():
    """Return the optical lattice potential data for visualization."""
    lattice_period = WAVELENGTH_D2 / 2  # λ/2
    x = np.linspace(0, 2 * lattice_period, 100)
    
    # Potential depth (typical value for Sisyphus cooling)
    U0 = 10 * RECOIL_ENERGY
    
    # Calculate potentials for different ground state sublevels
    potentials = []
    for m_F in range(-2, 3):  # m_F from -2 to 2
        phase = m_F * np.pi / 4
        potential = U0 * (1 - np.cos(2 * np.pi * x / lattice_period + phase))
        potentials.append(potential.tolist())
    
    return jsonify({
        'x': x.tolist(),
        'potentials': potentials,
        'lattice_period': lattice_period
    })

@app.route('/api/polarization-gradient')
def get_polarization_gradient():
    """Return the polarization gradient data for visualization."""
    lattice_period = WAVELENGTH_D2 / 2  # λ/2
    x = np.linspace(0, 2 * lattice_period, 100)
    
    # Calculate polarization components
    sigma_plus = np.sin(np.pi * x / lattice_period)**2
    sigma_minus = np.cos(np.pi * x / lattice_period)**2
    pi_pol = np.abs(np.sin(2 * np.pi * x / lattice_period))
    
    # Normalize
    total = sigma_plus + sigma_minus + pi_pol
    sigma_plus = sigma_plus / total
    sigma_minus = sigma_minus / total
    pi_pol = pi_pol / total
    
    return jsonify({
        'x': x.tolist(),
        'sigma_plus': sigma_plus.tolist(),
        'sigma_minus': sigma_minus.tolist(),
        'pi': pi_pol.tolist(),
        'lattice_period': lattice_period
    })

@app.route('/api/simulate-cooling')
def simulate_cooling():
    """Simulate Sisyphus cooling with the given parameters."""
    # Get parameters from request
    initial_temp = float(request.args.get('initialTemp', 100))  # μK
    detuning = float(request.args.get('detuning', -3))  # Γ
    rabi_freq = float(request.args.get('rabiFreq', 1))  # Γ
    
    # Convert to SI units
    initial_temp_K = initial_temp * 1e-6  # Convert μK to K
    
    # Simulate cooling
    total_time = 5e-3  # 5 ms
    dt = 1e-6  # 1 μs
    n_steps = int(total_time / dt)
    times = np.linspace(0, total_time, n_steps)
    
    # Simplified cooling rate model
    cooling_rate = (np.abs(detuning) / 3) * (rabi_freq / 2)
    sisyphus_temps = initial_temp_K * np.exp(-cooling_rate * times)
    
    # Simplified Doppler cooling model for comparison
    doppler_limit = DOPPLER_TEMPERATURE
    doppler_rate = 0.1 * cooling_rate  # Slower than Sisyphus cooling
    doppler_temps = doppler_limit + (initial_temp_K - doppler_limit) * np.exp(-doppler_rate * times)
    
    return jsonify({
        'times': times.tolist(),
        'sisyphus_temps': (sisyphus_temps / 1e-6).tolist(),  # Convert to μK
        'doppler_temps': (doppler_temps / 1e-6).tolist()  # Convert to μK
    })

@app.route('/api/entity-info/<entity_id>')
def get_entity_info(entity_id):
    """Return detailed information about an entity in the world model."""
    entity_info = {
        'RB87_MASS': {
            'name': 'Rubidium-87 Mass',
            'value': f'{RB87_MASS:.3e} kg',
            'description': 'The mass of a Rubidium-87 atom.',
            'related': ['RECOIL_ENERGY']
        },
        'WAVELENGTH_D2': {
            'name': 'D2 Transition Wavelength',
            'value': f'{WAVELENGTH_D2 * 1e9:.3f} nm',
            'description': 'The wavelength of the D2 transition in Rubidium-87.',
            'related': ['WAVENUMBER_D2', 'D2_Transition']
        },
        'WAVENUMBER_D2': {
            'name': 'D2 Transition Wavenumber',
            'value': f'{WAVENUMBER_D2:.3e} m⁻¹',
            'description': 'The wavenumber (k) of the D2 transition in Rubidium-87.',
            'related': ['WAVELENGTH_D2', 'RECOIL_ENERGY']
        },
        'GAMMA_D2': {
            'name': 'D2 Transition Linewidth',
            'value': f'{GAMMA_D2 / (2 * np.pi) * 1e-6:.3f} MHz',
            'description': 'The natural linewidth of the D2 transition in Rubidium-87.',
            'related': ['DOPPLER_TEMPERATURE', 'D2_Transition']
        },
        'RECOIL_ENERGY': {
            'name': 'Recoil Energy',
            'value': f'{RECOIL_ENERGY:.3e} J',
            'description': 'The energy transferred to an atom when it absorbs or emits a single photon.',
            'related': ['RECOIL_TEMPERATURE', 'RB87_MASS', 'WAVENUMBER_D2']
        },
        'RECOIL_TEMPERATURE': {
            'name': 'Recoil Temperature',
            'value': f'{RECOIL_TEMPERATURE * 1e6:.3f} μK',
            'description': 'The temperature corresponding to the recoil energy, sets a fundamental limit for laser cooling.',
            'related': ['RECOIL_ENERGY', 'Sisyphus_Cooling']
        },
        'DOPPLER_TEMPERATURE': {
            'name': 'Doppler Temperature',
            'value': f'{DOPPLER_TEMPERATURE * 1e6:.3f} μK',
            'description': 'The lowest temperature achievable with Doppler cooling, a fundamental limit for this technique.',
            'related': ['GAMMA_D2', 'Sisyphus_Cooling']
        },
        'Sisyphus_Cooling': {
            'name': 'Sisyphus Cooling',
            'value': 'N/A',
            'description': 'A sub-Doppler laser cooling technique that allows atoms to reach temperatures below the Doppler cooling limit.',
            'related': ['Optical_Lattice', 'Polarization_Gradient', 'Optical_Pumping']
        },
        'Optical_Lattice': {
            'name': 'Optical Lattice',
            'value': 'N/A',
            'description': 'A periodic potential created by interfering laser beams, forms the basis for Sisyphus cooling.',
            'related': ['Polarization_Gradient', 'WAVELENGTH_D2']
        },
        'Polarization_Gradient': {
            'name': 'Polarization Gradient',
            'value': 'N/A',
            'description': 'A spatially varying polarization created by counter-propagating laser beams with orthogonal polarizations.',
            'related': ['Optical_Lattice', 'Optical_Pumping']
        },
        'Optical_Pumping': {
            'name': 'Optical Pumping',
            'value': 'N/A',
            'description': 'The process of using light to transfer atoms between different internal states.',
            'related': ['Sisyphus_Cooling', 'D2_Transition']
        },
        'D2_Transition': {
            'name': 'D2 Transition',
            'value': '5²S₁/₂, F=2 → 5²P₃/₂, F\'=3',
            'description': 'The atomic transition used for laser cooling of Rubidium-87.',
            'related': ['WAVELENGTH_D2', 'GAMMA_D2']
        }
    }
    
    if entity_id in entity_info:
        return jsonify(entity_info[entity_id])
    else:
        return jsonify({'error': 'Entity not found'}), 404

@app.route('/api/find-path')
def find_path():
    """Find a path between two entities in the world model."""
    source = request.args.get('source')
    target = request.args.get('target')
    
    # Simplified path finding (in a real implementation, this would use graph algorithms)
    paths = {
        ('RB87_MASS', 'RECOIL_TEMPERATURE'): [
            'RB87_MASS', 'RECOIL_ENERGY_EQUATION', 'RECOIL_ENERGY', 
            'RECOIL_TEMPERATURE_EQUATION', 'RECOIL_TEMPERATURE'
        ],
        ('WAVELENGTH_D2', 'DOPPLER_TEMPERATURE'): [
            'WAVELENGTH_D2', 'D2_Transition', 'GAMMA_D2', 
            'DOPPLER_TEMPERATURE_EQUATION', 'DOPPLER_TEMPERATURE'
        ],
        ('Optical_Pumping', 'RECOIL_TEMPERATURE'): [
            'Optical_Pumping', 'Sisyphus_Cooling', 'RECOIL_TEMPERATURE'
        ]
    }
    
    # Try both directions
    key = (source, target)
    reverse_key = (target, source)
    
    if key in paths:
        return jsonify({'path': paths[key]})
    elif reverse_key in paths:
        return jsonify({'path': list(reversed(paths[reverse_key]))})
    else:
        # Generate a simple path if not found in our predefined paths
        return jsonify({'path': [source, target]})

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = 8085
    print(f"Starting Sisyphus Cooling Simulation server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
