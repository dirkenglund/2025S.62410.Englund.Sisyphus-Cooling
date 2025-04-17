import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import time
import math
from scipy.constants import hbar, k as k_B
from scipy.stats import maxwell
import base64
from io import BytesIO
import json

# Set page configuration
st.set_page_config(
    page_title="Sisyphus Cooling Interactive Demo",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        color: #1e1e1e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6f0ff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a86e8;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stSlider > div > div {
        background-color: #4a86e8;
    }
    .highlight {
        background-color: #e6f0ff;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .comparison-table {
        width: 100%;
        text-align: center;
    }
    .comparison-table th {
        background-color: #4a86e8;
        color: white;
        padding: 10px;
    }
    .comparison-table td {
        padding: 8px;
    }
    .comparison-table tr:nth-child(even) {
        background-color: #e6f0ff;
    }
    .equation {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2em;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4a86e8;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a76d8;
    }
    .cooling-method {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 5px solid #4a86e8;
    }
</style>
""", unsafe_allow_html=True)

# Constants for Rubidium-87
RB87_MASS = 1.44316060e-25  # kg
WAVELENGTH = 780.24e-9  # m
WAVENUMBER = 2 * np.pi / WAVELENGTH
GAMMA = 2 * np.pi * 6.07e6  # Natural linewidth in rad/s
RECOIL_ENERGY = (hbar * WAVENUMBER) ** 2 / (2 * RB87_MASS)
RECOIL_TEMPERATURE = RECOIL_ENERGY / k_B
DOPPLER_TEMPERATURE = hbar * GAMMA / (2 * k_B)

# Create tabs for different sections
tabs = st.tabs(["Introduction", "Theory", "Simulation", "World Model", "Comparison"])

# Introduction Tab
with tabs[0]:
    st.title("Sisyphus Cooling Interactive Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Interactive Sisyphus Cooling Demo
        
        This educational tool demonstrates the principles of Sisyphus cooling, a sub-Doppler laser cooling technique used in atomic physics to cool atoms to extremely low temperatures. The demo allows you to explore the physics behind this cooling method, visualize the cooling process, and compare it with other cooling techniques.
        
        ### What you can do with this demo:
        
        - Understand the physics principles behind Sisyphus cooling
        - Visualize the cooling process and atomic motion
        - Experiment with different parameters to see their effects
        - Compare Sisyphus cooling with other cooling methods
        - Explore the relationships between different physical concepts
        
        Use the tabs above to navigate through different sections of the demo.
        """)
    
    with col2:
        st.markdown("### Rubidium-87 Properties")
        st.markdown("""
        - **Atomic Number:** 37
        - **Mass:** 86.909 u
        - **D2 Transition Wavelength:** 780.24 nm
        - **Natural Linewidth (Γ):** 2π × 6.07 MHz
        - **Doppler Cooling Limit:** 146 μK
        - **Recoil Temperature:** 362 nK
        """)
    
    st.markdown("---")
    
    st.markdown("## Sisyphus Effect")
    st.markdown("""
    The key cooling mechanism works as follows:
    
    1. An atom in a specific ground state sublevel moves up a potential hill
    2. Near the top of the hill, it's optically pumped to another sublevel with lower potential energy
    3. The energy difference is carried away by the scattered photon
    4. This process repeats, continuously removing energy from the atom's motion
    
    This mechanism allows cooling to temperatures well below the Doppler limit, approaching the recoil limit.
    """)
    
    st.markdown("## Key Equations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recoil Energy:")
        st.latex(r"E_r = \frac{(\hbar k)^2}{2m}")
        
        st.markdown("### Recoil Temperature:")
        st.latex(r"T_r = \frac{E_r}{k_B}")
    
    with col2:
        st.markdown("### Doppler Temperature:")
        st.latex(r"T_D = \frac{\hbar \Gamma}{2k_B}")
        
        st.markdown("### Optical Pumping Rate:")
        st.latex(r"R_{ij} = \frac{\Omega^2}{4\Delta^2} \Gamma |C_{ij}|^2")

# Theory Tab
with tabs[1]:
    st.title("Theory of Sisyphus Cooling")
    
    st.markdown("""
    ## Optical Lattice and Polarization Gradients
    
    Sisyphus cooling relies on the creation of an optical lattice with a spatially varying polarization gradient. This is typically achieved using two counter-propagating laser beams with orthogonal polarizations.
    """)
    
    st.markdown("### Polarization Gradient Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        In the lin⊥lin configuration, the polarization changes from:
        - Linear (x) → Circular (σ⁺) → Linear (y) → Circular (σ⁻) → Linear (x)
        
        This creates a periodic potential landscape for different magnetic sublevels of the ground state.
        """)
        
        st.markdown("### Potential Energy Landscape")
        st.latex(r"U_g(z) = U_0 \cos^2(kz)")
        st.markdown("where $U_0$ is the potential depth and $k$ is the wavenumber of the laser light.")
    
    with col2:
        # Placeholder for polarization gradient visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        z = np.linspace(0, 2*np.pi, 1000)
        ax.plot(z, np.cos(z)**2, 'b-', label='$m_g = +1/2$')
        ax.plot(z, np.sin(z)**2, 'r-', label='$m_g = -1/2$')
        ax.set_xlabel('Position (z/λ)')
        ax.set_ylabel('Potential Energy')
        ax.set_title('Optical Lattice Potential for Different Sublevels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.markdown("""
    ## Cooling Mechanism
    
    The Sisyphus cooling mechanism can be understood through the following steps:
    
    1. **Optical Pumping**: Atoms are optically pumped between different ground state sublevels based on their position in the optical lattice.
    
    2. **Energy Loss**: When an atom moves up a potential hill and is then optically pumped to a state with lower potential energy, it loses kinetic energy.
    
    3. **Repeated Cycles**: This process repeats many times, leading to a continuous reduction in the atom's kinetic energy.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Optical Pumping Rates")
        st.markdown("""
        The rate of optical pumping between ground state sublevels depends on:
        - Laser intensity (via Rabi frequency Ω)
        - Detuning from resonance (Δ)
        - Clebsch-Gordan coefficients (C_ij)
        
        The pumping rate is given by:
        """)
        st.latex(r"R_{ij} = \frac{\Omega^2}{4\Delta^2} \Gamma |C_{ij}|^2")
    
    with col2:
        st.markdown("### Cooling Limit")
        st.markdown("""
        The final temperature achievable with Sisyphus cooling is limited by:
        - The depth of the optical lattice potential (U_0)
        - The recoil energy (E_r)
        
        For optimal parameters, the temperature can reach:
        """)
        st.latex(r"T_{final} \approx \frac{U_0}{10 k_B}")
        st.markdown("which can be much lower than the Doppler cooling limit.")
    
    st.markdown("---")
    
    st.markdown("""
    ## Quantum Description
    
    A full quantum mechanical description of Sisyphus cooling involves:
    
    1. **Dressed States**: The combined atom-light system forms dressed states that depend on the atom's position.
    
    2. **Density Matrix**: The evolution of the atomic density matrix under the influence of the light field and spontaneous emission.
    
    3. **Fokker-Planck Equation**: The atomic motion can be described by a Fokker-Planck equation with position-dependent diffusion and friction coefficients.
    """)
    
    st.markdown("""
    ### Friction Coefficient
    
    The friction coefficient in Sisyphus cooling is given by:
    """)
    st.latex(r"\alpha \approx \frac{8\hbar k^2 \Delta \Omega^2}{\Gamma^2 \Delta^2 + 2\Omega^2\Gamma^2}")
    
    st.markdown("""
    ### Diffusion Coefficient
    
    The momentum diffusion coefficient has two contributions:
    1. Spontaneous emission: $D_{spont} \approx \hbar^2 k^2 R_{scatt}$
    2. Fluctuations in the dipole force: $D_{dip} \approx \hbar^2 k^2 \Omega^2 / \Gamma$
    
    The total diffusion coefficient is:
    """)
    st.latex(r"D \approx D_{spont} + D_{dip}")

# Simulation Tab
with tabs[2]:
    st.title("Interactive Simulation")
    
    st.markdown("""
    Adjust the parameters below to see how they affect the Sisyphus cooling process. The simulation shows the temperature evolution, velocity distribution, and atom retention over time.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detuning = st.slider("Detuning (Γ)", -10.0, -0.1, -3.0, 0.1)
        rabi_freq = st.slider("Rabi Frequency (Γ)", 0.1, 5.0, 1.0, 0.1)
    
    with col2:
        initial_temp = st.slider("Initial Temperature (μK)", 100, 1000, 500, 10)
        lattice_depth = st.slider("Lattice Depth (E_r)", 10, 500, 100, 10)
    
    with col3:
        atom_number = st.slider("Initial Atom Number", 1000, 10000, 5000, 100)
        simulation_time = st.slider("Simulation Time (ms)", 1, 50, 20, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("Start Simulation", key="start_sim")
    
    with col2:
        reset_button = st.button("Reset Simulation", key="reset_sim")
    
    st.markdown("## Simulation Results")
    
    # Initialize session state for simulation results
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    
    if reset_button:
        st.session_state.pop('simulation_run', None)
        st.rerun()
    
    if start_button or st.session_state.simulation_run:
        st.session_state.simulation_run = True
        
        # Simulation parameters
        time_points = np.linspace(0, simulation_time, 100)
        
        # Sisyphus cooling simulation
        def sisyphus_cooling(t, initial_temp, detuning, rabi_freq, lattice_depth):
            # Simplified model of temperature evolution in Sisyphus cooling
            # Parameters in natural units (Γ, E_r)
            
            # Friction coefficient (simplified model)
            alpha = 8 * (rabi_freq**2) * abs(detuning) / (detuning**2 + 2*rabi_freq**2)
            
            # Diffusion coefficient (simplified model)
            D_spont = (rabi_freq**2) / (detuning**2 + 1)
            D_dip = (rabi_freq**2) / 4
            D = D_spont + D_dip
            
            # Equilibrium temperature in recoil units
            T_eq = D / (2 * alpha)
            
            # Temperature evolution
            T = T_eq + (initial_temp/RECOIL_TEMPERATURE - T_eq) * np.exp(-alpha * t * 1e-3)
            
            # Convert to μK
            T_uK = T * RECOIL_TEMPERATURE * 1e6
            
            # Calculate atom retention (simplified model)
            # Atoms are lost if their energy exceeds the lattice depth
            retention = 1.0 - np.exp(-(lattice_depth / (T * 3)))
            retention = retention * np.exp(-0.05 * t)  # Additional loss due to background collisions
            
            return T_uK, retention
        
        # Doppler cooling simulation for comparison
        def doppler_cooling(t, initial_temp):
            # Simplified model of temperature evolution in Doppler cooling
            T_eq = DOPPLER_TEMPERATURE * 1e6  # μK
            
            # Temperature evolution (simplified exponential approach)
            T = T_eq + (initial_temp - T_eq) * np.exp(-0.1 * t)
            
            # Calculate atom retention (simplified model)
            retention = 0.9 * np.exp(-0.1 * t)
            
            return T, retention
        
        # Run simulations
        sisyphus_temp, sisyphus_retention = sisyphus_cooling(time_points, initial_temp, detuning, rabi_freq, lattice_depth)
        doppler_temp, doppler_retention = doppler_cooling(time_points, initial_temp)
        
        # Calculate atom numbers
        sisyphus_atoms = atom_number * sisyphus_retention
        doppler_atoms = atom_number * doppler_retention
        
        # Create velocity distributions
        def maxwell_boltzmann(v, T):
            # Maxwell-Boltzmann distribution in 1D
            m = RB87_MASS
            return np.sqrt(m/(2*np.pi*k_B*T)) * np.exp(-m*v**2/(2*k_B*T))
        
        v = np.linspace(-1, 1, 1000)  # m/s
        
        # Initial distribution
        dist_initial = maxwell_boltzmann(v, initial_temp*1e-6)
        dist_initial_norm = dist_initial / np.max(dist_initial)
        
        # Final distributions
        dist_sisyphus = maxwell_boltzmann(v, sisyphus_temp[-1]*1e-6)
        dist_sisyphus_norm = dist_sisyphus / np.max(dist_sisyphus)
        
        dist_doppler = maxwell_boltzmann(v, doppler_temp[-1]*1e-6)
        dist_doppler_norm = dist_doppler / np.max(dist_doppler)
        
        # Create plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature evolution plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=sisyphus_temp, 
                mode='lines', 
                name='Sisyphus Cooling',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=doppler_temp, 
                mode='lines', 
                name='Doppler Cooling',
                line=dict(color='red', width=2)
            ))
            fig.add_shape(
                type="line",
                x0=0, y0=DOPPLER_TEMPERATURE*1e6, x1=simulation_time, y1=DOPPLER_TEMPERATURE*1e6,
                line=dict(color="red", width=1, dash="dash"),
                name="Doppler Limit"
            )
            fig.add_annotation(
                x=simulation_time/2, y=DOPPLER_TEMPERATURE*1e6*1.1,
                text="Doppler Limit",
                showarrow=False,
                font=dict(color="red")
            )
            fig.update_layout(
                title="Temperature Evolution",
                xaxis_title="Time (ms)",
                yaxis_title="Temperature (μK)",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Atom retention plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=sisyphus_atoms, 
                mode='lines', 
                name='Sisyphus Cooling',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=doppler_atoms, 
                mode='lines', 
                name='Doppler Cooling',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="Atom Retention",
                xaxis_title="Time (ms)",
                yaxis_title="Number of Atoms",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Velocity distribution plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=v, 
            y=dist_initial_norm, 
            mode='lines', 
            name='Initial',
            line=dict(color='gray', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=v, 
            y=dist_sisyphus_norm, 
            mode='lines', 
            name='Sisyphus Cooling',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=v, 
            y=dist_doppler_norm, 
            mode='lines', 
            name='Doppler Cooling',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="Velocity Distribution (Normalized)",
            xaxis_title="Velocity (m/s)",
            yaxis_title="Probability Density (Normalized)",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display final results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Final Temperature (Sisyphus)",
                value=f"{sisyphus_temp[-1]:.1f} μK",
                delta=f"{sisyphus_temp[-1] - initial_temp:.1f} μK"
            )
        
        with col2:
            st.metric(
                label="Final Temperature (Doppler)",
                value=f"{doppler_temp[-1]:.1f} μK",
                delta=f"{doppler_temp[-1] - initial_temp:.1f} μK"
            )
        
        with col3:
            st.metric(
                label="Temperature Ratio",
                value=f"{doppler_temp[-1]/sisyphus_temp[-1]:.1f}x",
                delta="Improvement with Sisyphus cooling"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Atoms Retained (Sisyphus)",
                value=f"{int(sisyphus_atoms[-1])}",
                delta=f"{int(sisyphus_atoms[-1] - atom_number)} atoms"
            )
        
        with col2:
            st.metric(
                label="Atoms Retained (Doppler)",
                value=f"{int(doppler_atoms[-1])}",
                delta=f"{int(doppler_atoms[-1] - atom_number)} atoms"
            )
        
        with col3:
            retention_ratio = sisyphus_atoms[-1]/doppler_atoms[-1] if doppler_atoms[-1] > 0 else float('inf')
            st.metric(
                label="Retention Ratio",
                value=f"{retention_ratio:.1f}x",
                delta="More atoms with Sisyphus cooling"
            )
        
        # Spatial distribution visualization
        st.markdown("### Spatial Distribution of Atoms")
        
        # Create a spatial distribution based on the optical lattice
        def generate_spatial_distribution(num_atoms, lattice_depth, temperature):
            # Position in units of wavelength
            z = np.linspace(0, 4, 1000)
            
            # Potential for the two ground state sublevels
            U_plus = lattice_depth * RECOIL_ENERGY * np.cos(2*np.pi*z)**2
            U_minus = lattice_depth * RECOIL_ENERGY * np.sin(2*np.pi*z)**2
            
            # Boltzmann distribution
            P_plus = np.exp(-U_plus/(k_B*temperature*1e-6))
            P_minus = np.exp(-U_minus/(k_B*temperature*1e-6))
            
            # Normalize
            P_plus = P_plus / np.sum(P_plus)
            P_minus = P_minus / np.sum(P_minus)
            
            # Sample positions
            num_plus = num_atoms // 2
            num_minus = num_atoms - num_plus
            
            positions_plus = np.random.choice(z, size=num_plus, p=P_plus)
            positions_minus = np.random.choice(z, size=num_minus, p=P_minus)
            
            return z, U_plus, U_minus, positions_plus, positions_minus
        
        # Generate distributions
        z, U_plus, U_minus, pos_plus, pos_minus = generate_spatial_distribution(
            int(sisyphus_atoms[-1]), 
            lattice_depth, 
            sisyphus_temp[-1]
        )
        
        # Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add potential curves
        fig.add_trace(
            go.Scatter(
                x=z, 
                y=U_plus/(k_B*1e-6), 
                mode='lines', 
                name='Potential (m=+1/2)',
                line=dict(color='rgba(0,0,255,0.5)', width=2)
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=z, 
                y=U_minus/(k_B*1e-6), 
                mode='lines', 
                name='Potential (m=-1/2)',
                line=dict(color='rgba(255,0,0,0.5)', width=2)
            ),
            secondary_y=True
        )
        
        # Add histogram of atom positions
        fig.add_trace(
            go.Histogram(
                x=np.concatenate([pos_plus, pos_minus]),
                name='Atom Distribution',
                marker_color='green',
                opacity=0.7,
                nbinsx=40
            )
        )
        
        fig.update_layout(
            title="Spatial Distribution of Atoms in Optical Lattice",
            xaxis_title="Position (λ)",
            yaxis_title="Number of Atoms",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        fig.update_yaxes(
            title_text="Potential Energy (μK)", 
            secondary_y=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# World Model Tab
with tabs[3]:
    st.title("World Model: Concept Relationships")
    
    st.markdown("""
    This interactive graph visualizes the relationships between different concepts in Sisyphus cooling. 
    You can explore how various physical quantities, equations, and cooling mechanisms are interconnected.
    """)
    
    # Create a graph data structure
    nodes = [
        {"id": "sisyphus", "label": "Sisyphus Cooling", "group": "method"},
        {"id": "doppler", "label": "Doppler Cooling", "group": "method"},
        {"id": "vscpt", "label": "VSCPT", "group": "method"},
        {"id": "raman", "label": "Raman Cooling", "group": "method"},
        {"id": "evaporative", "label": "Evaporative Cooling", "group": "method"},
        
        {"id": "optical_lattice", "label": "Optical Lattice", "group": "concept"},
        {"id": "polarization", "label": "Polarization Gradient", "group": "concept"},
        {"id": "optical_pumping", "label": "Optical Pumping", "group": "concept"},
        {"id": "dressed_states", "label": "Dressed States", "group": "concept"},
        {"id": "light_shift", "label": "Light Shift", "group": "concept"},
        
        {"id": "temperature", "label": "Temperature", "group": "quantity"},
        {"id": "energy", "label": "Energy", "group": "quantity"},
        {"id": "momentum", "label": "Momentum", "group": "quantity"},
        {"id": "detuning", "label": "Detuning", "group": "quantity"},
        {"id": "intensity", "label": "Laser Intensity", "group": "quantity"},
        
        {"id": "doppler_limit", "label": "Doppler Limit", "group": "limit"},
        {"id": "recoil_limit", "label": "Recoil Limit", "group": "limit"},
        
        {"id": "friction", "label": "Friction Coefficient", "group": "equation"},
        {"id": "diffusion", "label": "Diffusion Coefficient", "group": "equation"},
        {"id": "fokker_planck", "label": "Fokker-Planck Equation", "group": "equation"},
        
        {"id": "rb87", "label": "Rubidium-87", "group": "atom"},
        {"id": "cs133", "label": "Cesium-133", "group": "atom"},
        
        {"id": "mot", "label": "Magneto-Optical Trap", "group": "apparatus"},
        {"id": "optical_molasses", "label": "Optical Molasses", "group": "apparatus"},
        
        {"id": "bec", "label": "Bose-Einstein Condensate", "group": "application"},
        {"id": "quantum_computing", "label": "Quantum Computing", "group": "application"},
        {"id": "atomic_clock", "label": "Atomic Clocks", "group": "application"}
    ]
    
    # Define edges (connections between concepts)
    edges = [
        {"source": "sisyphus", "target": "optical_lattice", "value": 5},
        {"source": "sisyphus", "target": "polarization", "value": 5},
        {"source": "sisyphus", "target": "optical_pumping", "value": 4},
        {"source": "sisyphus", "target": "doppler_limit", "value": 3},
        {"source": "sisyphus", "target": "recoil_limit", "value": 2},
        {"source": "sisyphus", "target": "temperature", "value": 5},
        {"source": "sisyphus", "target": "friction", "value": 4},
        {"source": "sisyphus", "target": "diffusion", "value": 3},
        
        {"source": "doppler", "target": "optical_pumping", "value": 3},
        {"source": "doppler", "target": "doppler_limit", "value": 5},
        {"source": "doppler", "target": "temperature", "value": 4},
        {"source": "doppler", "target": "optical_molasses", "value": 5},
        
        {"source": "vscpt", "target": "recoil_limit", "value": 5},
        {"source": "vscpt", "target": "optical_pumping", "value": 4},
        {"source": "vscpt", "target": "momentum", "value": 5},
        
        {"source": "raman", "target": "recoil_limit", "value": 4},
        {"source": "raman", "target": "momentum", "value": 4},
        {"source": "raman", "target": "dressed_states", "value": 3},
        
        {"source": "evaporative", "target": "bec", "value": 5},
        {"source": "evaporative", "target": "temperature", "value": 4},
        
        {"source": "optical_lattice", "target": "polarization", "value": 4},
        {"source": "optical_lattice", "target": "light_shift", "value": 3},
        {"source": "optical_lattice", "target": "energy", "value": 3},
        
        {"source": "polarization", "target": "optical_pumping", "value": 4},
        
        {"source": "temperature", "target": "energy", "value": 4},
        {"source": "temperature", "target": "momentum", "value": 3},
        {"source": "temperature", "target": "doppler_limit", "value": 4},
        {"source": "temperature", "target": "recoil_limit", "value": 4},
        
        {"source": "friction", "target": "fokker_planck", "value": 4},
        {"source": "diffusion", "target": "fokker_planck", "value": 4},
        {"source": "friction", "target": "temperature", "value": 3},
        
        {"source": "rb87", "target": "sisyphus", "value": 4},
        {"source": "rb87", "target": "doppler", "value": 4},
        {"source": "rb87", "target": "mot", "value": 4},
        
        {"source": "cs133", "target": "sisyphus", "value": 3},
        {"source": "cs133", "target": "atomic_clock", "value": 5},
        
        {"source": "mot", "target": "optical_molasses", "value": 4},
        {"source": "mot", "target": "doppler", "value": 4},
        {"source": "mot", "target": "sisyphus", "value": 3},
        
        {"source": "sisyphus", "target": "bec", "value": 3},
        {"source": "vscpt", "target": "bec", "value": 3},
        {"source": "raman", "target": "bec", "value": 3},
        
        {"source": "bec", "target": "quantum_computing", "value": 4},
        {"source": "bec", "target": "atomic_clock", "value": 3}
    ]
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node["id"], label=node["label"], group=node["group"])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["value"])
    
    # Create a dictionary for node groups
    group_colors = {
        "method": "#4285F4",  # Blue
        "concept": "#EA4335",  # Red
        "quantity": "#FBBC05",  # Yellow
        "limit": "#34A853",  # Green
        "equation": "#8F00FF",  # Purple
        "atom": "#FF6D01",  # Orange
        "apparatus": "#00ACC1",  # Cyan
        "application": "#AB47BC"  # Pink
    }
    
    # Create node colors based on groups
    node_colors = [group_colors[G.nodes[node]["group"]] for node in G.nodes()]
    
    # Create a Plotly figure
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line_width=2))
    
    # Add node information for hover
    node_text = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        neighbor_text = "<br>".join([G.nodes[n]["label"] for n in neighbors])
        node_text.append(f"<b>{G.nodes[node]['label']}</b><br><br>Connected to:<br>{neighbor_text}")
    
    node_trace.text = node_text
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(text='Interactive Concept Map of Sisyphus Cooling', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Add a legend
    for group, color in group_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=group.capitalize(),
            showlegend=True
        ))
    
    # Display the graph
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanations for selected concepts
    st.markdown("## Concept Explanations")
    
    concept = st.selectbox(
        "Select a concept to learn more:",
        options=[node["label"] for node in nodes],
        index=0
    )
    
    # Concept explanations
    explanations = {
        "Sisyphus Cooling": """
        **Sisyphus Cooling** is a sub-Doppler laser cooling technique that uses polarization gradients to create a spatially varying light shift of atomic energy levels. 
        
        Named after the Greek myth of Sisyphus, who was condemned to roll a boulder up a hill only to have it roll back down, atoms in this cooling scheme repeatedly climb potential hills, lose energy through optical pumping, and repeat the process.
        
        This technique can cool atoms to temperatures significantly below the Doppler cooling limit, approaching the recoil limit.
        """,
        
        "Doppler Cooling": """
        **Doppler Cooling** is the most basic laser cooling technique, relying on the Doppler effect to create a velocity-dependent force on atoms.
        
        When an atom moves toward a laser beam, it sees the light frequency shifted higher (blue-shifted) due to the Doppler effect. Similarly, when moving away, it sees the frequency shifted lower (red-shifted).
        
        By tuning the laser slightly below the atomic resonance (red-detuned), atoms preferentially absorb photons when moving toward the laser beam, resulting in a net cooling force.
        
        The temperature limit for Doppler cooling is T_D = ħΓ/2k_B, which is 146 μK for Rubidium-87.
        """,
        
        "VSCPT": """
        **Velocity-Selective Coherent Population Trapping (VSCPT)** is a sub-recoil cooling technique that can cool atoms below the recoil limit.
        
        VSCPT works by creating quantum states that are decoupled from the light field for atoms with zero velocity. Atoms in these "dark states" no longer interact with the laser and thus stop being heated by photon recoil.
        
        Over time, atoms accumulate in these dark states, resulting in a very narrow velocity distribution centered at zero velocity.
        
        VSCPT can achieve temperatures significantly below the recoil limit, with the final temperature limited only by the interaction time.
        """,
        
        "Raman Cooling": """
        **Raman Cooling** is another sub-recoil cooling technique that uses stimulated Raman transitions between different ground state sublevels.
        
        The technique involves a series of velocity-selective Raman pulses that transfer atoms from higher momentum states to lower momentum states, followed by optical pumping to reset the internal state.
        
        By carefully designing the sequence of Raman pulses, atoms can be accumulated in a very narrow velocity distribution around zero.
        
        Raman cooling has been used to achieve temperatures as low as a few nanokelvin, well below the recoil limit.
        """,
        
        "Evaporative Cooling": """
        **Evaporative Cooling** is a technique used to reach ultra-low temperatures needed for Bose-Einstein condensation.
        
        The principle is similar to how a cup of coffee cools: by selectively removing the highest-energy atoms from a trapped ensemble, the remaining atoms rethermalize at a lower temperature.
        
        In practice, this is often achieved by lowering the depth of a magnetic or optical trap, allowing the most energetic atoms to escape.
        
        Evaporative cooling can achieve temperatures in the nanokelvin range but typically results in significant atom loss.
        """,
        
        "Optical Lattice": """
        **Optical Lattice** is a periodic potential created by the interference of laser beams.
        
        In the context of Sisyphus cooling, the optical lattice creates a spatially varying potential landscape for atoms in different ground state sublevels.
        
        The depth of the optical lattice potential is proportional to the laser intensity and inversely proportional to the detuning.
        
        Optical lattices are also used in quantum simulation, quantum computing, and precision measurements.
        """,
        
        "Polarization Gradient": """
        **Polarization Gradient** refers to the spatial variation of the polarization of the light field.
        
        In Sisyphus cooling, counter-propagating laser beams with orthogonal polarizations create a polarization that changes from linear to circular to orthogonal linear to opposite circular, and so on, over a distance of λ/4.
        
        This polarization gradient, combined with the different Clebsch-Gordan coefficients for different transitions, creates the spatially varying potential landscape that enables Sisyphus cooling.
        """,
        
        "Optical Pumping": """
        **Optical Pumping** is the process of using light to transfer atoms from one quantum state to another.
        
        In Sisyphus cooling, optical pumping transfers atoms between different ground state sublevels depending on their position in the optical lattice.
        
        The rate of optical pumping depends on the laser intensity, detuning, and the Clebsch-Gordan coefficients for the specific transitions.
        
        Optical pumping is a key mechanism in many laser cooling techniques, including Doppler cooling, Sisyphus cooling, and VSCPT.
        """,
        
        "Dressed States": """
        **Dressed States** are the eigenstates of the combined atom-light system.
        
        In the presence of a light field, the bare atomic states are "dressed" by the photons, resulting in new eigenstates that are superpositions of the bare states.
        
        The energies of these dressed states depend on the intensity and detuning of the light field, as well as the position of the atom in a spatially varying field.
        
        Dressed state formalism provides a powerful framework for understanding laser cooling and other light-atom interactions.
        """,
        
        "Light Shift": """
        **Light Shift**, also known as the AC Stark shift, is the shift in energy levels of an atom due to interaction with an oscillating electromagnetic field.
        
        In the context of Sisyphus cooling, the light shift creates the potential hills that atoms must climb, leading to energy loss.
        
        The magnitude of the light shift is proportional to the intensity of the light field and inversely proportional to the detuning from resonance.
        
        Different ground state sublevels experience different light shifts depending on their coupling to the excited states.
        """,
        
        "Temperature": """
        **Temperature** in the context of laser-cooled atoms refers to the average kinetic energy of the atoms.
        
        For a thermal distribution, the temperature T is related to the mean kinetic energy by E_kin = (3/2)k_B T.
        
        In laser cooling experiments, the temperature is typically measured by time-of-flight methods or by analyzing the velocity distribution.
        
        Different cooling techniques have different temperature limits, with Doppler cooling limited to ~100 μK, Sisyphus cooling to ~1 μK, and sub-recoil techniques to <100 nK.
        """,
        
        "Energy": """
        **Energy** in laser cooling can refer to several quantities:
        
        1. Kinetic energy of the atoms, which is directly related to temperature
        2. Potential energy in the optical lattice
        3. Internal energy of the atomic states
        4. Recoil energy from absorbing or emitting photons
        
        The goal of laser cooling is to reduce the kinetic energy of the atoms by converting it to potential energy and then dissipating it through spontaneous emission.
        """,
        
        "Momentum": """
        **Momentum** transfer is at the heart of laser cooling techniques.
        
        When an atom absorbs a photon, it receives a momentum kick in the direction of the photon's propagation. When it emits a photon spontaneously, it receives a recoil kick in a random direction.
        
        By carefully engineering the absorption process to be velocity-dependent, laser cooling creates a net force that reduces the atom's momentum.
        
        The minimum momentum spread achievable is limited by the recoil from a single photon, unless sub-recoil techniques are used.
        """,
        
        "Detuning": """
        **Detuning** refers to the difference between the laser frequency and the atomic resonance frequency.
        
        In Doppler and Sisyphus cooling, the laser is typically red-detuned (lower frequency than resonance) to create a cooling force.
        
        The optimal detuning for Sisyphus cooling is typically a few natural linewidths (Γ) below resonance.
        
        The detuning affects both the cooling rate and the final temperature, with larger detunings generally resulting in lower final temperatures but slower cooling.
        """,
        
        "Laser Intensity": """
        **Laser Intensity** is a critical parameter in all laser cooling techniques.
        
        In Sisyphus cooling, the intensity determines the depth of the optical lattice potential and the optical pumping rate.
        
        Higher intensities generally lead to faster cooling but can also increase the final temperature due to increased photon scattering.
        
        The intensity is often expressed in terms of the saturation intensity or through the Rabi frequency.
        """,
        
        "Doppler Limit": """
        **Doppler Limit** is the lowest temperature achievable with Doppler cooling, given by T_D = ħΓ/2k_B.
        
        For Rubidium-87, the Doppler limit is approximately 146 μK.
        
        This limit arises from the balance between the cooling force and the heating due to random photon recoil during spontaneous emission.
        
        Sub-Doppler cooling techniques like Sisyphus cooling can overcome this limit by using additional mechanisms beyond the Doppler effect.
        """,
        
        "Recoil Limit": """
        **Recoil Limit** is the temperature corresponding to the kinetic energy of an atom after absorbing or emitting a single photon, given by T_r = (ħk)²/2mk_B.
        
        For Rubidium-87, the recoil limit is approximately 362 nK.
        
        This was long thought to be the fundamental limit for laser cooling, but sub-recoil techniques like VSCPT and Raman cooling can overcome it.
        
        The recoil limit represents the quantum limit of laser cooling methods that rely on photon scattering.
        """,
        
        "Friction Coefficient": """
        **Friction Coefficient** characterizes the velocity-dependent force in laser cooling.
        
        In the low-velocity limit, the cooling force can be written as F = -αv, where α is the friction coefficient.
        
        For Sisyphus cooling, the friction coefficient is proportional to the laser intensity and inversely proportional to the detuning.
        
        A larger friction coefficient leads to faster cooling but may also increase the final temperature due to associated heating mechanisms.
        """,
        
        "Diffusion Coefficient": """
        **Diffusion Coefficient** characterizes the random heating due to the discrete nature of photon scattering.
        
        In laser cooling, there are two main contributions to momentum diffusion:
        1. Spontaneous emission, which causes random recoils
        2. Fluctuations in the dipole force due to the spatial variation of the light field
        
        The balance between friction (cooling) and diffusion (heating) determines the final temperature.
        """,
        
        "Fokker-Planck Equation": """
        **Fokker-Planck Equation** describes the evolution of the probability distribution of atomic positions and velocities under the influence of friction and diffusion.
        
        In laser cooling, it takes the form:
        ∂W/∂t = ∇_p·(αpW) + ∇_p·(D∇_pW)
        
        where W is the probability distribution, p is momentum, α is the friction coefficient, and D is the diffusion coefficient.
        
        The steady-state solution gives the final temperature as T = D/αk_B.
        """,
        
        "Rubidium-87": """
        **Rubidium-87** is one of the most commonly used atomic species in laser cooling experiments.
        
        Key properties:
        - Atomic number: 37
        - Mass: 86.909 u
        - D2 transition wavelength: 780.24 nm
        - Natural linewidth (Γ): 2π × 6.07 MHz
        - Doppler cooling limit: 146 μK
        - Recoil temperature: 362 nK
        
        Rubidium is popular due to its convenient transition wavelength (accessible with diode lasers) and favorable level structure.
        """,
        
        "Cesium-133": """
        **Cesium-133** is another commonly used atomic species in laser cooling experiments, particularly for atomic clocks.
        
        Key properties:
        - Atomic number: 55
        - Mass: 132.905 u
        - D2 transition wavelength: 852.35 nm
        - Natural linewidth (Γ): 2π × 5.22 MHz
        - Doppler cooling limit: 125 μK
        - Recoil temperature: 198 nK
        
        Cesium is used in the definition of the second and in high-precision atomic clocks.
        """,
        
        "Magneto-Optical Trap": """
        **Magneto-Optical Trap (MOT)** is a device that combines laser cooling with a magnetic field to trap and cool atoms.
        
        The MOT uses three pairs of counter-propagating laser beams along with a quadrupole magnetic field to create a position-dependent force that pushes atoms toward the center.
        
        MOTs can capture atoms from a room-temperature vapor and cool them to temperatures of a few hundred microkelvin.
        
        The MOT is typically the first stage in experiments requiring ultracold atoms, followed by further cooling techniques like Sisyphus cooling or evaporative cooling.
        """,
        
        "Optical Molasses": """
        **Optical Molasses** refers to the configuration of three pairs of counter-propagating laser beams used for Doppler cooling.
        
        The term "molasses" comes from the viscous damping force experienced by atoms in this light field.
        
        In optical molasses, atoms are cooled but not trapped, as there is no position-dependent force.
        
        Surprisingly, experiments in the 1980s found that optical molasses could cool atoms below the Doppler limit, which led to the discovery of Sisyphus cooling.
        """,
        
        "Bose-Einstein Condensate": """
        **Bose-Einstein Condensate (BEC)** is a state of matter formed when bosonic atoms are cooled to temperatures very close to absolute zero.
        
        In a BEC, a large fraction of the atoms occupy the lowest quantum state, creating a macroscopic quantum object.
        
        Creating a BEC typically requires:
        1. Initial laser cooling (Doppler, Sisyphus)
        2. Transfer to a magnetic or optical trap
        3. Evaporative cooling to reach nanokelvin temperatures
        
        BECs were first created in 1995 and have since become important tools for studying quantum physics.
        """,
        
        "Quantum Computing": """
        **Quantum Computing** leverages quantum mechanical phenomena to perform computations.
        
        Laser-cooled atoms can serve as qubits in quantum computers, with their internal states representing the 0 and 1 states.
        
        Optical lattices can be used to arrange atoms in regular arrays for quantum computing applications.
        
        The ultra-low temperatures achieved through laser cooling are essential for maintaining quantum coherence.
        """,
        
        "Atomic Clocks": """
        **Atomic Clocks** use the precise frequency of atomic transitions to keep time with extraordinary accuracy.
        
        Laser cooling is essential for atomic clocks because:
        1. It reduces Doppler broadening of spectral lines
        2. It allows for longer interrogation times
        3. It minimizes systematic shifts due to atomic motion
        
        The most advanced atomic clocks use laser-cooled atoms or ions and have fractional frequency uncertainties below 10^-18.
        
        These clocks are used for precision measurements, tests of fundamental physics, and as the basis for the definition of the second.
        """
    }
    
    # Display the selected concept explanation
    if concept in explanations:
        st.markdown(explanations[concept])
    else:
        st.markdown("Explanation not available for this concept.")

# Comparison Tab
with tabs[4]:
    st.title("Comparison of Cooling Methods")
    
    st.markdown("""
    This section compares Sisyphus cooling with other sub-Doppler and sub-recoil cooling techniques. 
    Understanding the strengths and limitations of each method is crucial for choosing the appropriate technique for specific applications.
    """)
    
    # Temperature ranges comparison
    st.markdown("## Temperature Ranges")
    
    # Create temperature range data
    cooling_methods = ["Doppler Cooling", "Sisyphus Cooling", "VSCPT", "Raman Cooling", "Evaporative Cooling"]
    temp_min = [100, 1, 0.05, 0.05, 0.001]  # μK
    temp_max = [1000, 20, 1, 1, 0.1]  # μK
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Method': cooling_methods,
        'Min Temperature (μK)': temp_min,
        'Max Temperature (μK)': temp_max
    })
    
    # Create a horizontal bar chart
    fig = px.timeline(
        df, x_start="Min Temperature (μK)", x_end="Max Temperature (μK)", y="Method",
        color="Method", opacity=0.8,
        title="Temperature Ranges Achievable with Different Cooling Methods"
    )
    
    # Update layout to use log scale for x-axis
    fig.update_xaxes(type="log", title="Temperature (μK)")
    fig.update_yaxes(title="")
    
    # Add vertical lines for reference temperatures
    fig.add_vline(x=DOPPLER_TEMPERATURE*1e6, line_dash="dash", line_color="red", 
                 annotation_text="Doppler Limit", annotation_position="top right")
    fig.add_vline(x=RECOIL_TEMPERATURE*1e6, line_dash="dash", line_color="green", 
                 annotation_text="Recoil Limit", annotation_position="top right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison table
    st.markdown("## Detailed Comparison")
    
    st.markdown("""
    <table class="comparison-table">
        <tr>
            <th>Method</th>
            <th>Temperature Range</th>
            <th>Atom Retention</th>
            <th>Complexity</th>
            <th>Key Advantages</th>
            <th>Limitations</th>
        </tr>
        <tr>
            <td>Doppler Cooling</td>
            <td>100 μK - 1 mK</td>
            <td>High (>90%)</td>
            <td>Low</td>
            <td>Simple implementation, robust, high capture velocity</td>
            <td>Limited by Doppler temperature (~146 μK for Rb87)</td>
        </tr>
        <tr>
            <td>Sisyphus Cooling</td>
            <td>1 μK - 20 μK</td>
            <td>Medium-High (70-90%)</td>
            <td>Medium</td>
            <td>Efficient sub-Doppler cooling, relatively simple implementation</td>
            <td>Limited by lattice depth, sensitive to magnetic fields</td>
        </tr>
        <tr>
            <td>VSCPT</td>
            <td>50 nK - 1 μK</td>
            <td>Low-Medium (30-60%)</td>
            <td>High</td>
            <td>Can reach sub-recoil temperatures, narrow momentum distribution</td>
            <td>Slow cooling rate, complex implementation, low efficiency</td>
        </tr>
        <tr>
            <td>Raman Cooling</td>
            <td>50 nK - 1 μK</td>
            <td>Medium (50-70%)</td>
            <td>High</td>
            <td>Can reach sub-recoil temperatures, works in 3D</td>
            <td>Requires complex laser systems, sensitive alignment</td>
        </tr>
        <tr>
            <td>Evaporative Cooling</td>
            <td>1 nK - 100 nK</td>
            <td>Very Low (1-10%)</td>
            <td>Medium</td>
            <td>Can reach quantum degeneracy (BEC), works with various species</td>
            <td>Significant atom loss, requires initial laser cooling</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Other Sub-Doppler Cooling Techniques
    st.markdown("## Other Sub-Doppler Cooling Techniques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="cooling-method">
            <h3>Velocity-Selective Coherent Population Trapping (VSCPT)</h3>
            <p>VSCPT uses quantum interference to create "dark states" that are decoupled from the light field for atoms with zero velocity.</p>
            <p><strong>Key features:</strong></p>
            <ul>
                <li>Can cool below the recoil limit</li>
                <li>Creates very narrow momentum distribution</li>
                <li>Based on quantum interference effects</li>
                <li>Requires specific atomic level structure</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cooling-method">
            <h3>Raman Cooling</h3>
            <p>Raman cooling uses stimulated Raman transitions to selectively transfer atoms from higher momentum states to lower momentum states.</p>
            <p><strong>Key features:</strong></p>
            <ul>
                <li>Can cool below the recoil limit</li>
                <li>Works with a variety of atomic species</li>
                <li>Requires precise control of laser frequencies</li>
                <li>Can be implemented in 1D, 2D, or 3D</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="cooling-method">
            <h3>Gray Molasses</h3>
            <p>Gray molasses combines Sisyphus cooling with velocity-selective coherent population trapping using blue-detuned lasers.</p>
            <p><strong>Key features:</strong></p>
            <ul>
                <li>Efficient cooling for atoms with complex level structure</li>
                <li>Works with blue-detuned lasers</li>
                <li>Can cool to a few μK</li>
                <li>Particularly useful for cooling molecules</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison of cooling efficiency
    st.markdown("## Cooling Efficiency Comparison")
    
    # Create data for cooling efficiency plot
    methods = ["Doppler", "Sisyphus", "VSCPT", "Raman", "Evaporative"]
    cooling_rate = [0.8, 0.6, 0.2, 0.3, 0.1]  # Relative cooling rate
    final_temp = [1.0, 0.1, 0.01, 0.01, 0.001]  # Relative to Doppler limit
    atom_retention = [0.9, 0.8, 0.5, 0.6, 0.1]  # Fraction of atoms retained
    
    # Create a bubble chart
    fig = go.Figure()
    
    # Add traces for each cooling method
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatter(
            x=[cooling_rate[i]],
            y=[final_temp[i]],
            mode='markers',
            marker=dict(
                size=atom_retention[i] * 50,
                color=colors[i],
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=method,
            text=[f"{method}<br>Cooling Rate: {cooling_rate[i]:.1f}<br>Final Temp: {final_temp[i]:.3f} T_D<br>Atom Retention: {atom_retention[i]:.1f}"],
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title="Cooling Method Efficiency Comparison",
        xaxis=dict(
            title="Relative Cooling Rate",
            type="linear"
        ),
        yaxis=dict(
            title="Final Temperature (T/T_Doppler)",
            type="log"
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        annotations=[
            dict(
                x=0.5,
                y=0.02,
                xref="paper",
                yref="paper",
                text="Bubble size represents atom retention",
                showarrow=False
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Applications
    st.markdown("## Applications of Ultra-Cold Atoms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Quantum Computing and Simulation
        
        Ultra-cold atoms can be used as qubits in quantum computers and quantum simulators. Different cooling techniques are suitable for different quantum computing architectures:
        
        - **Optical Lattice Quantum Computers**: Require Sisyphus cooling followed by evaporative cooling
        - **Ion Trap Quantum Computers**: Use Doppler cooling followed by sideband cooling
        - **Neutral Atom Arrays**: Use Sisyphus cooling and Raman cooling
        
        The ability to cool atoms to ultra-low temperatures is essential for maintaining quantum coherence during computation.
        """)
    
    with col2:
        st.markdown("""
        ### Precision Measurements and Metrology
        
        Ultra-cold atoms are used in various precision measurement devices:
        
        - **Atomic Clocks**: Use Sisyphus and Raman cooling to reduce Doppler shifts
        - **Atom Interferometers**: Use Sisyphus cooling to create coherent matter waves
        - **Gravitational Wave Detectors**: Use ultra-cold atoms as test masses
        
        The choice of cooling technique depends on the required temperature, coherence time, and atom number for the specific application.
        """)
    
    st.markdown("""
    ### Fundamental Physics Research
    
    Ultra-cold atoms provide a clean platform for studying fundamental physics:
    
    - **Bose-Einstein Condensates**: Require evaporative cooling after initial laser cooling
    - **Quantum Many-Body Physics**: Use various cooling techniques depending on the system
    - **Tests of Fundamental Symmetries**: Require the highest precision, often using multiple cooling stages
    
    The development of new cooling techniques continues to push the boundaries of what's possible in quantum physics research.
    """)

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Sisyphus Cooling Interactive Demo | Created for 6.2410 Quantum Physics Lab | MIT
    <br>
    For educational purposes only
</div>
""", unsafe_allow_html=True)
