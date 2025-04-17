import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from streamlit_d3graph import d3graph
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Sisyphus Cooling Interactive Demo",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility and styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        color: #1e1e1e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .sub-header {
        font-size: 24px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .equation {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 18px;
    }
    .card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 10px;
        border-left: 3px solid #4e8df5;
        margin: 10px 0;
    }
    .comparison-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .comparison-title {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Sisyphus Cooling Interactive Demo")
st.markdown("An educational simulation of Sisyphus cooling for Rubidium-87 atoms")

# Create tabs
tabs = st.tabs(["Introduction", "Theory", "Simulation", "World Model", "Comparison"])

# Introduction Tab
with tabs[0]:
    st.markdown('<h2 class="sub-header">Introduction to Sisyphus Cooling</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Sisyphus cooling is a sub-Doppler laser cooling technique that allows atoms to be cooled to temperatures 
        significantly below the Doppler cooling limit. Named after the Greek mythological figure Sisyphus, 
        who was condemned to roll a boulder up a hill only to have it roll back down, this technique 
        exploits a similar repetitive process to remove energy from atoms.
        
        This interactive demo explores the principles and implementation of Sisyphus cooling for 
        Rubidium-87 atoms, demonstrating:
        
        - The quantum mechanical principles behind Sisyphus cooling
        - The optical lattice potential and polarization gradient
        - Temperature evolution during the cooling process
        - Atom retention efficiency during cooling
        - Comparison with other cooling methods
        
        Sisyphus cooling was first proposed by Claude Cohen-Tannoudji in 1989, who later received the 
        Nobel Prize in Physics in 1997 for this work.
        """)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("""
        **Key Advantages of Sisyphus Cooling:**
        - Achieves temperatures in the microkelvin range (μK)
        - Overcomes the Doppler cooling limit
        - Essential for many quantum physics experiments and applications
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Rubidium-87 Properties")
        st.markdown("""
        - **Atomic Number:** 37
        - **Mass:** 86.909 u
        - **D2 Transition Wavelength:** 780.24 nm
        - **Natural Linewidth (Γ):** 2π × 6.07 MHz
        - **Doppler Cooling Limit:** 146 μK
        - **Recoil Temperature:** 362 nK
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a simple illustration
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create position array
        x = np.linspace(0, 2, 1000)
        
        # Create potential
        potential1 = 0.5 * np.sin(2 * np.pi * x) + 1.5
        potential2 = 0.5 * np.sin(2 * np.pi * x + np.pi) + 1.5
        
        # Plot potentials
        ax.plot(x, potential1, 'b-', linewidth=2, label='$m_F = +1/2$')
        ax.plot(x, potential2, 'r-', linewidth=2, label='$m_F = -1/2$')
        
        # Add atom and transitions
        ax.plot([0.25], [potential1[250]], 'ko', markersize=10, label='Atom')
        ax.arrow(0.25, potential1[250], 0.2, 0, head_width=0.1, head_length=0.05, fc='k', ec='k')
        ax.arrow(0.45, potential1[450], 0, potential2[450]-potential1[450], head_width=0.05, head_length=0.1, fc='g', ec='g')
        
        ax.set_xlabel('Position (λ)')
        ax.set_ylabel('Potential Energy')
        ax.set_title('Sisyphus Cooling Principle')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', 'λ/4', 'λ/2', '3λ/4', 'λ'])
        ax.set_yticks([])
        
        st.pyplot(fig)

# Theory Tab
with tabs[1]:
    st.markdown('<h2 class="sub-header">Theoretical Background</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Sisyphus cooling relies on the interaction between atoms with multiple ground state sublevels 
    and a light field with a polarization gradient. This section explains the key theoretical 
    concepts behind this cooling mechanism.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Energy Level Structure</h3>', unsafe_allow_html=True)
        st.markdown("""
        For Rubidium-87, we focus on the D2 transition between:
        - Ground state: 5²S₁/₂, F=2 with magnetic sublevels m<sub>F</sub> = -2, -1, 0, 1, 2
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
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.3)
        ax.set_axis_off()
        
        st.pyplot(fig)
    
    with col2:
        st.markdown('<h3>Optical Lattice Potential</h3>', unsafe_allow_html=True)
        st.markdown("""
        The interference of two counter-propagating laser beams with orthogonal polarizations 
        creates a standing wave with a spatially varying polarization. This results in a periodic 
        potential landscape for the different magnetic sublevels of the ground state.
        """)
        
        # Create optical lattice potential plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create position array
        x = np.linspace(0, 2, 1000)
        
        # Create potentials for different m_F states
        potential_plus = 0.5 * np.sin(2 * np.pi * x) + 1.5
        potential_minus = 0.5 * np.sin(2 * np.pi * x + np.pi) + 1.5
        
        # Plot potentials
        ax.plot(x, potential_plus, 'b-', linewidth=2, label='m_F = +1/2')
        ax.plot(x, potential_minus, 'r-', linewidth=2, label='m_F = -1/2')
        
        # Add labels and legend
        ax.set_xlabel('Position (λ)')
        ax.set_ylabel('Potential Energy (U₀)')
        ax.set_title('Optical Lattice Potential for Different Ground State Sublevels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
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
    
    # Additional parameters for atom retention
    col4, col5 = st.columns(2)
    
    with col4:
        trap_depth = st.slider("Trap Depth (μK)", min_value=10, max_value=200, value=50, step=10,
                              help="Depth of the optical trap in microkelvin")
    
    with col5:
        initial_atoms = st.slider("Initial Atom Number", min_value=1000, max_value=10000, value=5000, step=1000,
                                 help="Initial number of atoms in the trap")
    
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
        sisyphus_temps = initial_temp * np.exp(-times * 1e3 * cooling_rate)
        
        # Simulate Doppler cooling for comparison
        doppler_cooling_rate = 0.2  # Simplified Doppler cooling rate
        doppler_temps = initial_temp * np.exp(-times * 1e3 * doppler_cooling_rate)
        
        # Calculate atom retention
        # Model: atoms with energy > trap_depth escape
        # Maxwell-Boltzmann distribution for energies
        retention_sisyphus = []
        retention_doppler = []
        
        for temp_s, temp_d in zip(sisyphus_temps, doppler_temps):
            # Probability of atom having energy < trap_depth
            p_retain_s = 1 - np.exp(-trap_depth / temp_s)
            p_retain_d = 1 - np.exp(-trap_depth / temp_d)
            
            # Number of atoms retained
            atoms_s = initial_atoms * p_retain_s
            ato
(Content truncated due to size limit. Use line ranges to read in chunks)