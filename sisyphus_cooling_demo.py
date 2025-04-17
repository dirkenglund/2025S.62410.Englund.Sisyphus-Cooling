import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.constants import hbar, k as k_B, atomic_mass, c
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Sisyphus Cooling Interactive Demo",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0066cc;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 5px solid #0066cc;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .equation {
        background-color: #f9f9f9;
        padding: 0.5rem;
        text-align: center;
        margin: 1rem 0;
        font-style: italic;
    }
    .caption {
        text-align: center;
        font-style: italic;
        margin-top: 0.5rem;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f8ff;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Physical constants for Rubidium-87
RB87_MASS = 86.909180527 * atomic_mass  # kg
WAVELENGTH_D2 = 780.241e-9  # m (D2 transition wavelength)
WAVENUMBER_D2 = 2 * np.pi / WAVELENGTH_D2  # m^-1
GAMMA_D2 = 2 * np.pi * 6.065e6  # s^-1 (natural linewidth of D2 transition)
RECOIL_ENERGY = (hbar * WAVENUMBER_D2)**2 / (2 * RB87_MASS)  # J
RECOIL_TEMPERATURE = RECOIL_ENERGY / k_B  # K
DOPPLER_TEMPERATURE = hbar * GAMMA_D2 / (2 * k_B)  # K

# Conversion factors
uK_to_K = 1e-6  # Convert microkelvin to kelvin
mK_to_K = 1e-3  # Convert millikelvin to kelvin

# Header
st.markdown('<h1 class="main-header">Sisyphus Cooling Interactive Demo</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This interactive demonstration allows you to explore Sisyphus cooling, a sub-Doppler laser cooling technique 
    that enables atoms to reach temperatures significantly below the Doppler cooling limit. Named after the Greek 
    mythological figure Sisyphus, this technique exploits a repetitive energy loss mechanism where atoms climb 
    potential hills and lose energy through optical pumping processes.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["Introduction", "Theory", "Simulation", "World Model", "Comparison"])

# Introduction Tab
with tabs[0]:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">What is Sisyphus Cooling?</h2>', unsafe_allow_html=True)
        st.markdown("""
        Sisyphus cooling is a laser cooling technique that allows atoms to reach temperatures below the Doppler cooling limit.
        
        In this process, atoms move through a spatially varying light field created by counter-propagating laser beams with 
        orthogonal polarizations. This creates a periodic potential landscape where different ground state sublevels experience 
        different potentials.
        
        As atoms move through this landscape, they undergo optical pumping between different ground state sublevels at positions 
        where the polarization favors specific transitions. When an atom moves up a potential hill in one sublevel and is then 
        optically pumped to another sublevel with a lower potential energy, it loses kinetic energy.
        
        This process repeats, continuously removing energy from the atom's motion, much like Sisyphus in Greek mythology who was 
        condemned to roll a boulder up a hill, only to have it roll back down, repeating this process for eternity.
        """)
        
        st.markdown('<h2 class="sub-header">Key Concepts</h2>', unsafe_allow_html=True)
        st.markdown("""
        - **Polarization Gradient**: Spatially varying polarization created by counter-propagating laser beams
        - **Optical Lattice**: Periodic potential landscape experienced by atoms
        - **Optical Pumping**: Light-induced transitions between atomic ground state sublevels
        - **Sisyphus Effect**: Repetitive process of climbing potential hills and losing energy
        """)
    
    with col2:
        st.markdown('<h2 class="sub-header">Temperature Limits</h2>', unsafe_allow_html=True)
        
        # Create a dataframe for temperature limits
        temp_data = {
            'Limit': ['Doppler Limit', 'Recoil Limit', 'Typical Sisyphus Cooling'],
            'Temperature (μK)': [DOPPLER_TEMPERATURE/uK_to_K, RECOIL_TEMPERATURE/uK_to_K, 5.0],
            'Description': [
                'Minimum temperature achievable with Doppler cooling',
                'Fundamental limit due to photon recoil',
                'Typical temperature achieved with Sisyphus cooling'
            ]
        }
        st.dataframe(temp_data, hide_index=True)
        
        st.markdown('<h2 class="sub-header">Rubidium-87 Properties</h2>', unsafe_allow_html=True)
        
        # Create a dataframe for Rb-87 properties
        rb_data = {
            'Property': ['D2 Transition', 'Wavelength', 'Natural Linewidth', 'Mass'],
            'Value': ['5²S₁/₂ → 5²P₃/₂', f'{WAVELENGTH_D2*1e9:.3f} nm', f'{GAMMA_D2/(2*np.pi)*1e-6:.3f} MHz', f'{RB87_MASS*1e27:.5f} × 10⁻²⁷ kg']
        }
        st.dataframe(rb_data, hide_index=True)
        
        # Add a simple illustration
        st.markdown('<h2 class="sub-header">Sisyphus Effect Illustration</h2>', unsafe_allow_html=True)
        
        # Create a simple illustration of the Sisyphus effect
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create position array
        x = np.linspace(0, 2, 1000)
        
        # Create two potential curves with phase difference
        y1 = 0.5 - 0.4 * np.cos(2 * np.pi * x)
        y2 = 0.5 - 0.4 * np.cos(2 * np.pi * x + np.pi)
        
        # Plot potentials
        ax.plot(x, y1, 'b-', label='m_F = -1/2')
        ax.plot(x, y2, 'r-', label='m_F = +1/2')
        
        # Add atom and arrows
        ax.plot([0.25], [y1[250]], 'ko', markersize=10, label='Atom')
        ax.arrow(0.25, y1[250], 0.2, 0.1, head_width=0.03, head_length=0.05, fc='k', ec='k')
        ax.arrow(0.45, y1[450], 0.05, -0.3, head_width=0.03, head_length=0.05, fc='g', ec='g')
        ax.arrow(0.5, y2[500], 0.2, 0.1, head_width=0.03, head_length=0.05, fc='k', ec='k')
        ax.arrow(0.7, y2[700], 0.05, -0.3, head_width=0.03, head_length=0.05, fc='g', ec='g')
        
        ax.set_xlabel('Position (λ/2)')
        ax.set_ylabel('Potential Energy')
        ax.set_title('Sisyphus Cooling Mechanism')
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        
        st.pyplot(fig)
        st.markdown('<p class="caption">Illustration of the Sisyphus cooling mechanism. The atom climbs a potential hill, undergoes optical pumping to a lower potential state, and repeats the process, continuously losing energy.</p>', unsafe_allow_html=True)

# Theory Tab
with tabs[1]:
    st.markdown('<h2 class="sub-header">Theoretical Background</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Energy Level Structure</h3>', unsafe_allow_html=True)
        st.markdown("""
        For Rubidium-87, we focus on the D2 transition between the 5²S₁/₂ ground state and the 
        5²P₃/₂ excited state. In our simplified model, we consider:
        
        - Ground state: F=2 with magnetic sublevels m<sub>F</sub> = -2, -1, 0, 1, 2
        - Excited state: F'=3 with magnetic sublevels m<sub>F</sub> = -3, -2, -1, 0, 1, 2, 3
        
        The transitions between these levels are governed by selection rules and 
        Clebsch-Gordan coefficients, which determine the coupling strengths.
        """, unsafe_allow_html=True)
        
        # Create energy level diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Ground state (5²S₁/₂)
        ax.plot([-1, 1], [0, 0], 'k-', linewidth=2)
        ax.text(-1.5, 0, '5²S₁/₂, F=2', fontsize=12)
        
        # Ground state magnetic sublevels
        for m_F in range(-2, 3):
            ax.plot([m_F*0.2-0.1, m_F*0.2+0.1], [0.05, 0.05], 'b-', linewidth=2)
            ax.text(m_F*0.2, 0.1, f'm_F={m_F}', fontsize=10, ha='center')
        
        # Excited state (5²P₃/₂)
        ax.plot([-1, 1], [1, 1], 'k-', linewidth=2)
        ax.text(-1.5, 1, '5²P₃/₂, F=3', fontsize=12)
        
        # Excited state magnetic sublevels
        for m_F in range(-3, 4):
            ax.plot([m_F*0.15-0.05, m_F*0.15+0.05], [1.05, 1.05], 'r-', linewidth=2)
            ax.text(m_F*0.15, 1.1, f'm_F={m_F}', fontsize=10, ha='center')
        
        # Draw transitions
        # σ+ transitions (Δm_F = +1)
        for m_F in range(-2, 3):
            ax.arrow(m_F*0.2, 0.07, (m_F+1)*0.15-m_F*0.2, 0.93, 
                     head_width=0.03, head_length=0.05, fc='g', ec='g', alpha=0.5)
        
        # σ- transitions (Δm_F = -1)
        for m_F in range(-1, 3):
            ax.arrow(m_F*0.2, 0.07, (m_F-1)*0.15-m_F*0.2, 0.93, 
                     head_width=0.03, head_length=0.05, fc='b', ec='b', alpha=0.5)
        
        # π transitions (Δm_F = 0)
        for m_F in range(-2, 3):
            ax.arrow(m_F*0.2, 0.07, m_F*0.15-m_F*0.2, 0.93, 
                     head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.5)
        
        # Add legend
        ax.plot([], [], 'g-', label='σ+ transitions (Δm_F = +1)')
        ax.plot([], [], 'b-', label='σ- transitions (Δm_F = -1)')
        ax.plot([], [], 'r-', label='π transitions (Δm_F = 0)')
        
        ax.legend(loc='upper center')
        ax.set_title('Rubidium-87 D2 Transition Energy Levels')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        st.pyplot(fig)
        
        st.markdown('<h3>Optical Lattice Potential</h3>', unsafe_allow_html=True)
        st.markdown("""
        Counter-propagating laser beams with orthogonal linear polarizations create a standing wave 
        with a spatially varying polarization. This results in a periodic potential landscape for atoms, 
        where different ground state sublevels experience different potentials.
        """)
        
        st.markdown('<div class="equation">U(x, m_F) = U_0 [1 - cos(2kx + φ(m_F))]</div>', unsafe_allow_html=True)
        
        st.markdown("""
        where U<sub>0</sub> is the potential depth, k is the wavenumber, and φ(m<sub>F</sub>) is a 
        phase that depends on the magnetic sublevel.
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>Polarization Gradient</h3>', unsafe_allow_html=True)
        st.markdown("""
        The polarization of the light field varies spatially from σ<sup>+</sup> to linear to σ<sup>-</sup> 
        and back, with a period of λ/2. This variation is crucial for the Sisyphus cooling mechanism.
        """, unsafe_allow_html=True)
        
        # Create polarization gradient plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create position array
        x = np.linspace(0, 1, 1000)
        
        # Create polarization components
        sigma_plus = np.sin(np.pi * x)**2
        sigma_minus = np.cos(np.pi * x)**2
        pi = np.sin(2 * np.pi * x)
        
        # Plot polarization components
        ax.plot(x, sigma_plus, 'g-', label='σ+')
        ax.plot(x, np.abs(pi), 'r-', label='π')
        ax.plot(x, sigma_minus, 'b-', label='σ-')
        
        ax.set_xlabel('Position (λ/2)')
        ax.set_ylabel('Polarization Component')
        ax.set_title('Polarization Gradient Along the Optical Lattice')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        st.markdown('<h3>Optical Pumping</h3>', unsafe_allow_html=True)
        st.markdown("""
        As atoms move through the polarization gradient, they undergo optical pumping between different 
        ground state sublevels. The pumping rates depend on the local polarization and the Clebsch-Gordan 
        coefficients.
        """)
        
        st.markdown('<h3>Sisyphus Effect</h3>', unsafe_allow_html=True)
        st.markdown("""
        The key cooling mechanism works as follows:
        
        1. An atom in a specific ground state sublevel moves up a potential hill
        2. Near the top of the hill, it's optically pumped to another sublevel with lower potential energy
        3. The energy difference is carried away by the scattered photon
        4. This process repeats, continuously removing energy from the atom's motion
        
        This mechanism allows cooling to temperatures well below the Doppler limit, approaching the 
        recoil limit.
        """)
        
        st.markdown('<h3>Key Equations</h3>', unsafe_allow_html=True)
        
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.markdown('<p><strong>Recoil Energy:</strong></p>', unsafe_allow_html=True)
            st.latex(r"E_r = \frac{(\hbar k)^2}{2m}")
            
            st.markdown('<p><strong>Recoil Temperature:</strong></p>', unsafe_allow_html=True)
            st.latex(r"T_r = \frac{E_r}{k_B}")
        
        with col_eq2:
            st.markdown('<p><strong>Doppler Temperature:</strong></p>', unsafe_allow_html=True)
            st.latex(r"T_D = \frac{\hbar \Gamma}{2k_B}")
            
            st.markdown('<p><strong>Optical Pumping Rate:</strong></p>', unsafe_allow_html=True)
            st.latex(r"R_{ij} = \frac{\Omega^2}{4\Delta^2} \Gamma |C_{ij}|^2")

# Simulation Tab
with tabs[2]:
    st.markdown('<h2 class="sub-header">Interactive Simulation</h2>', unsafe_allow_html=True)
    
    # Simulation parameters
    st.markdown('<h3>Simulation Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detuning = st.slider("Detuning (Γ units)", min_value=-10.0, max_value=-1.0, value=-3.0, step=0.5, 
                            help="Detuning of the laser from resonance in units of natural linewidth Γ")
    
    with col2:
        rabi_freq = st.slider("Rabi Frequency (Γ units)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                             help="Rabi frequency in units of natural linewidth Γ")
    
    with col3:
        initial_temp = st.slider("Initial Temperature (μK)", min_value=50, max_value=500, value=100, step=50,
                                help="Initial temperature of the atoms in microkelvin")
    
    # Simulation control
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    
    with col_btn1:
        start_button = st.button("Start Simulation", type="primary")
    
    with col_btn2:
        reset_button = st.button("Reset Simulation")
    
    # Simulation results
    st.markdown('<h3>Simulation Results</h3>', unsafe_allow_html=True)
    
    col_temp, col_vel = st.columns(2)
    
    # Run simulation
    if start_button or 'simulation_run' in st.session_state:
        if start_button:
            st.session_state.simulation_run = True
        
        # Simulation parameters
        total_time = 5e-3  # 5 ms
        dt = 1e-6  # 1 μs
        n_steps = int(total_time / dt)
        times = np.linspace(0, total_time, n_steps)
        
        # Simulate Sisyphus cooling
        cooling_rate = (np.abs(detuning) / 3) * (rabi_freq / 2)  # Simplified cooling rate
        sisyphus_temps = initial_temp 
(Content truncated due to size limit. Use line ranges to read in chunks)