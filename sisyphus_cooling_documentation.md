# Sisyphus Cooling Model for Rubidium-87 Atoms

## Introduction

This document provides an explanation of the Sisyphus cooling model implemented for Rubidium-87 atoms using QuTiP (Quantum Toolbox in Python). The model demonstrates the key principles of Sisyphus cooling, a sub-Doppler laser cooling technique that allows atoms to reach temperatures below the Doppler cooling limit.

## Physics Principles

### Sisyphus Cooling Mechanism

Sisyphus cooling, also known as polarization gradient cooling, is a laser cooling technique named after the Greek mythological figure Sisyphus, who was condemned to roll a boulder up a hill, only to have it roll back down, repeating this process for eternity. In the atomic physics context, atoms repeatedly climb potential hills and lose energy in the process.

The key components of Sisyphus cooling are:

1. **Polarization Gradient**: Two counter-propagating laser beams with orthogonal polarizations create a standing wave with a spatially varying polarization (alternating between σ+, linear, and σ- polarization).

2. **State-Dependent Potential**: Different ground state magnetic sublevels experience different light shifts (AC Stark shifts) depending on the local polarization, creating a periodic potential landscape.

3. **Optical Pumping**: As atoms move through this landscape, they undergo optical pumping between different ground state sublevels at positions where the polarization favors specific transitions.

4. **Energy Loss Mechanism**: When an atom moves up a potential hill in one sublevel and is then optically pumped to another sublevel with a lower potential energy, it loses kinetic energy. This process repeats, continuously removing energy from the atom's motion.

### Rubidium-87 D2 Transition

For our model, we focus on the D2 transition of Rubidium-87, which involves:

- Ground state: 5²S₁/₂, F=2 (with magnetic sublevels m_F = -2, -1, 0, 1, 2)
- Excited state: 5²P₃/₂, F=3 (with magnetic sublevels m_F = -3, -2, -1, 0, 1, 2, 3)

This transition at 780.241 nm is commonly used in laser cooling experiments due to its favorable properties:
- Closed cycling transition (atoms that are excited tend to return to the same ground state)
- Accessible with readily available diode lasers
- Suitable hyperfine structure for optical pumping

### Cooling Limits

Sisyphus cooling can achieve temperatures below the Doppler cooling limit, which for Rubidium-87 is approximately 146 μK. The ultimate limit for Sisyphus cooling is the recoil temperature (about 0.18 μK for Rubidium-87), which corresponds to the energy transferred to an atom when it emits a single photon.

## Implementation Details

### Quantum System Model

The quantum system is modeled using the `RubidiumAtom` class, which:
- Defines the ground and excited state manifolds
- Creates quantum operators for transitions between states
- Implements the Clebsch-Gordan coefficients for the D2 transition

```python
class RubidiumAtom:
    def __init__(self):
        # Define dimensions of ground and excited state manifolds
        self.ground_dim = 5  # F=2 has 5 magnetic sublevels
        self.excited_dim = 7  # F'=3 has 7 magnetic sublevels
        self.total_dim = self.ground_dim + self.excited_dim
        
        # Create basis states and operators
        # ...
```

### Optical Lattice Potential

The optical lattice potential is implemented in the `SisyphusCooling` class:

```python
def optical_lattice_potential(self, position):
    # Normalized position in the lattice
    x_norm = position / self.lattice_period
    
    # Potential depth
    U0 = 10 * RECOIL_ENERGY  # Typical value for Sisyphus cooling
    
    # Potentials for different ground state sublevels
    potentials = np.zeros(self.atom.ground_dim)
    
    for m_F in range(-2, 3):  # m_F from -2 to 2
        idx = m_F + 2  # Convert m_F to index
        # Phase depends on m_F
        phase = m_F * np.pi / 4
        potentials[idx] = U0 * (1 - np.cos(2 * np.pi * x_norm + phase))
    
    return potentials
```

This creates a position-dependent potential for each ground state sublevel, with the potentials shifted in phase relative to each other.

### Atom-Light Interaction

The atom-light interaction includes:

1. **Polarization Gradient**: The polarization varies spatially along the standing wave:

```python
def polarization_at_position(self, position):
    # Normalized position in the lattice
    x_norm = position / self.lattice_period
    
    # Polarization components vary sinusoidally
    sigma_plus = np.sin(np.pi * x_norm)**2
    sigma_minus = np.cos(np.pi * x_norm)**2
    pi = np.sin(2 * np.pi * x_norm)
    
    # Normalize
    total = sigma_plus + sigma_minus + abs(pi)
    return sigma_plus/total, abs(pi)/total, sigma_minus/total
```

2. **Optical Pumping**: Atoms are pumped between ground state sublevels based on the local polarization:

```python
def update_atom_states(self, dt):
    for i in range(self.n_atoms):
        # Get pumping rates at this position
        rates = self.optical_pumping_rates(position)
        
        # Determine if a transition occurs
        if np.random.rand() < np.sum(probs[:, current_state]):
            # Choose the new state based on relative probabilities
            # ...
            
            # When an atom changes state, it loses energy (Sisyphus effect)
            old_potential = self.optical_lattice_potential(position)[current_state]
            new_potential = self.optical_lattice_potential(position)[self.atom_states[i]]
            
            # Energy difference goes to photon
            energy_diff = old_potential - new_potential
            
            # If energy difference is positive, atom loses energy
            if energy_diff > 0:
                # The key Sisyphus cooling effect
                self.potential_energies[i] -= energy_diff
                self.total_energies[i] -= energy_diff
```

### Simulation Dynamics

The simulation evolves the system in time steps, updating:
1. Atom internal states (optical pumping)
2. Atom positions (based on velocities)
3. Atom velocities (based on forces and spontaneous emission)

The temperature is calculated from the average kinetic energy of the atoms:
```python
temperature = np.mean(self.kinetic_energies) / k_B
```

## Results and Analysis

### Temperature Evolution

The simulation demonstrates that Sisyphus cooling can achieve significantly lower temperatures than Doppler cooling:

- **Sisyphus Cooling**: Temperature decreases rapidly from the initial 100 μK to near the recoil limit
- **Doppler Cooling**: Temperature stabilizes around the Doppler limit (~146 μK)

The cooling comparison plot shows this difference on a logarithmic scale, clearly demonstrating the advantage of Sisyphus cooling for achieving ultra-cold temperatures.

### Spatial Distribution

The spatial distribution plots show how atoms become localized in the optical lattice potential wells as they cool. Initially, atoms are distributed randomly with a thermal velocity distribution. As cooling progresses, atoms become trapped in the potential wells, with their positions corresponding to the minima of the potential for their particular ground state sublevel.

### Energy Levels and Transitions

The energy level diagram illustrates the D2 transition structure for Rubidium-87, showing:
- Ground state (5²S₁/₂, F=2) with 5 magnetic sublevels
- Excited state (5²P₃/₂, F=3) with 7 magnetic sublevels
- Different types of transitions (σ+, σ-, π) between these levels

### Optical Lattice and Polarization Gradient

The optical lattice potential plot shows how different ground state sublevels experience different potential landscapes, with the potentials shifted in phase relative to each other. This phase shift is crucial for the Sisyphus cooling mechanism.

The polarization gradient plot shows how the polarization components (σ+, σ-, π) vary along the optical lattice, creating the conditions for state-dependent optical pumping.

## Limitations and Future Improvements

### Model Limitations

1. **Simplified Quantum Structure**: The model uses a simplified representation of the Rubidium-87 energy levels, focusing only on the F=2 → F'=3 transition and neglecting other hyperfine levels.

2. **One-Dimensional Model**: The simulation is limited to one spatial dimension, whereas real experiments are three-dimensional.

3. **Semi-Classical Approach**: The model treats atomic motion classically while using quantum mechanics for internal states, which is an approximation.

4. **Damping Coefficient**: The model includes a velocity damping coefficient to ensure cooling dominates over heating, which is a simplification of the actual cooling mechanism.

### Potential Improvements

1. **Full Quantum Treatment**: Implement a fully quantum mechanical treatment of both internal states and atomic motion.

2. **Three-Dimensional Model**: Extend the model to three spatial dimensions for more realistic simulations.

3. **Additional Hyperfine Levels**: Include other hyperfine levels and transitions for a more complete model.

4. **Realistic Experimental Parameters**: Calibrate the model with parameters from actual experimental setups.

5. **Quantum Monte Carlo Approach**: Implement a quantum Monte Carlo wave function approach for more accurate modeling of spontaneous emission.

## Conclusion

This educational model successfully demonstrates the key principles of Sisyphus cooling for Rubidium-87 atoms. It shows how the combination of a polarization gradient, state-dependent potentials, and optical pumping leads to efficient cooling below the Doppler limit.

The visualizations provide clear insights into the cooling mechanism, the energy level structure, and the spatial dynamics of the atoms. Despite its simplifications, the model captures the essential physics of Sisyphus cooling and provides a foundation for understanding more complex laser cooling techniques.

## References

1. Dalibard, J.; Cohen-Tannoudji, C. (1989). "Laser cooling below the Doppler limit by polarization gradients: simple theoretical models". Journal of the Optical Society of America B. 6 (11): 2023.

2. Steck, D.A. (2001). "Rubidium 87 D Line Data". Available at: https://steck.us/alkalidata/rubidium87numbers.1.6.pdf

3. Metcalf, H.J.; van der Straten, P. (1999). "Laser Cooling and Trapping". Springer.

4. Foot, C.J. (2005). "Atomic Physics". Oxford University Press. Section 9.6.
