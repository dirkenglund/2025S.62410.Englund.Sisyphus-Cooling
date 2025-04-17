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
                x=ti
(Content truncated due to size limit. Use line ranges to read in chunks)