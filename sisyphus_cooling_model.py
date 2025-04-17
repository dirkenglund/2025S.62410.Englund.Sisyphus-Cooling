    def simulate_doppler_cooling(self, total_time, dt, callback=None):
        """
        Simulate Doppler cooling for comparison.
        
        This is a simplified model of Doppler cooling, where the cooling force
        is proportional to the atom's velocity.
        
        Args:
            total_time (float): Total simulation time
            dt (float): Time step
            callback (function): Optional callback function called at each step
            
        Returns:
            tuple: (times, temperatures)
        """
        # Reset atoms
        self.reset_atoms()
        
        n_steps = int(total_time / dt)
        times = np.linspace(0, total_time, n_steps)
        temperatures = np.zeros(n_steps)
        
        # Doppler cooling parameters
        gamma = GAMMA_D2
        detuning = -gamma / 2  # Optimal detuning for Doppler cooling
        saturation = 0.1  # Reduced saturation parameter to prevent excessive heating
        
        for i in range(n_steps):
            # Calculate Doppler cooling force for each atom
            for j in range(self.n_atoms):
                velocity = self.atom_velocities[j]
                
                # Doppler cooling force (simplified model)
                # F = ħk * γ/2 * s0 / (1 + s0 + (2(δ + kv)/γ)²)
                # For red detuning, this creates a velocity-dependent force that opposes motion
                force = -hbar * WAVENUMBER_D2 * gamma/2 * saturation * 4 * detuning * velocity / (
                    gamma * (1 + saturation + (2*detuning/gamma)**2)**2
                )
                
                # Apply force
                self.atom_velocities[j] += force / RB87_MASS * dt
                
                # Add random recoil kicks from spontaneous emission
                # Reduced scattering rate to prevent excessive heating
                scattering_rate = 0.1 * gamma/2 * saturation / (
                    1 + saturation + (2*(detuning + WAVENUMBER_D2*velocity)/gamma)**2
                )
                n_scattering = np.random.poisson(scattering_rate * dt)
                
                if n_scattering > 0:
                    recoil_velocity = hbar * WAVENUMBER_D2 / RB87_MASS
                    for _ in range(n_scattering):
                        direction = np.random.choice([-1, 1])
                        self.atom_velocities[j] += direction * recoil_velocity
                
                # Add velocity damping to ensure cooling
                damping_coefficient = 0.005  # Small damping coefficient
                self.atom_velocities[j] *= (1.0 - damping_coefficient)
                
                # Update kinetic energy
                self.kinetic_energies[j] = 0.5 * RB87_MASS * self.atom_velocities[j]**2
            
            # Calculate temperature
            temperatures[i] = np.mean(self.kinetic_energies) / k_B
            
            if callback is not None:
                callback(i, times[i], temperatures[i])
        
        return times, temperatures


def plot_cooling_comparison(sisyphus_times, sisyphus_temps, doppler_times, doppler_temps):
    """
    Plot a comparison of Sisyphus cooling and Doppler cooling.
    
    Args:
        sisyphus_times (ndarray): Time points for Sisyphus cooling
        sisyphus_temps (ndarray): Temperatures for Sisyphus cooling
        doppler_times (ndarray): Time points for Doppler cooling
        doppler_temps (ndarray): Temperatures for Doppler cooling
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to microkelvin for better readability
    sisyphus_temps_uK = sisyphus_temps / uK_to_K
    doppler_temps_uK = doppler_temps / uK_to_K
    
    # Plot temperature evolution
    plt.plot(sisyphus_times * 1e3, sisyphus_temps_uK, 'b-', label='Sisyphus Cooling')
    plt.plot(doppler_times * 1e3, doppler_temps_uK, 'r-', label='Doppler Cooling')
    
    # Add temperature limits
    plt.axhline(y=RECOIL_TEMPERATURE / uK_to_K, color='k', linestyle='--', 
                label=f'Recoil Limit ({RECOIL_TEMPERATURE/uK_to_K:.1f} μK)')
    plt.axhline(y=DOPPLER_TEMPERATURE / uK_to_K, color='g', linestyle='--', 
                label=f'Doppler Limit ({DOPPLER_TEMPERATURE/uK_to_K:.1f} μK)')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Temperature (μK)')
    plt.title('Comparison of Cooling Methods for Rubidium-87')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('cooling_comparison.png', dpi=300)
    plt.close()


def plot_spatial_distribution(positions, velocities, times, n_snapshots=4):
    """
    Plot the spatial distribution of atoms at different time points.
    
    Args:
        positions (ndarray): Atom positions at each time step
        velocities (ndarray): Atom velocities at each time step
        times (ndarray): Time points
        n_snapshots (int): Number of snapshots to plot
    """
    plt.figure(figsize=(12, 8))
    
    # Select time indices for snapshots
    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)
    
    for i, idx in enumerate(indices):
        plt.subplot(n_snapshots, 1, i+1)
        
        # Plot phase space distribution (position vs. velocity)
        plt.scatter(positions[idx], velocities[idx], s=1, alpha=0.5)
        
        # Calculate temperature at this time
        kinetic_energies = 0.5 * RB87_MASS * velocities[idx]**2
        temperature = np.mean(kinetic_energies) / k_B
        
        plt.title(f'Time: {times[idx]*1e3:.1f} ms, Temperature: {temperature/uK_to_K:.1f} μK')
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('spatial_distribution.png', dpi=300)
    plt.close()


def plot_energy_levels():
    """Plot the energy level diagram for Rubidium-87 D2 transition."""
    plt.figure(figsize=(8, 10))
    
    # Ground state (5²S₁/₂)
    plt.plot([-1, 1], [0, 0], 'k-', linewidth=2)
    plt.text(-1.5, 0, '5²S₁/₂, F=2', fontsize=12)
    
    # Ground state magnetic sublevels
    for m_F in range(-2, 3):
        plt.plot([m_F*0.2-0.1, m_F*0.2+0.1], [0.05, 0.05], 'b-', linewidth=2)
        plt.text(m_F*0.2, 0.1, f'm_F={m_F}', fontsize=10, ha='center')
    
    # Excited state (5²P₃/₂)
    plt.plot([-1, 1], [1, 1], 'k-', linewidth=2)
    plt.text(-1.5, 1, '5²P₃/₂, F=3', fontsize=12)
    
    # Excited state magnetic sublevels
    for m_F in range(-3, 4):
        plt.plot([m_F*0.15-0.05, m_F*0.15+0.05], [1.05, 1.05], 'r-', linewidth=2)
        plt.text(m_F*0.15, 1.1, f'm_F={m_F}', fontsize=10, ha='center')
    
    # Draw transitions
    # σ+ transitions (Δm_F = +1)
    for m_F in range(-2, 3):
        plt.arrow(m_F*0.2, 0.07, (m_F+1)*0.15-m_F*0.2, 0.93, 
                 head_width=0.03, head_length=0.05, fc='g', ec='g', alpha=0.5)
    
    # σ- transitions (Δm_F = -1)
    for m_F in range(-1, 3):
        plt.arrow(m_F*0.2, 0.07, (m_F-1)*0.15-m_F*0.2, 0.93, 
                 head_width=0.03, head_length=0.05, fc='b', ec='b', alpha=0.5)
    
    # π transitions (Δm_F = 0)
    for m_F in range(-2, 3):
        plt.arrow(m_F*0.2, 0.07, m_F*0.15-m_F*0.2, 0.93, 
                 head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.5)
    
    # Add legend
    plt.plot([], [], 'g-', label='σ+ transitions (Δm_F = +1)')
    plt.plot([], [], 'b-', label='σ- transitions (Δm_F = -1)')
    plt.plot([], [], 'r-', label='π transitions (Δm_F = 0)')
    
    plt.legend(loc='upper center')
    plt.title('Rubidium-87 D2 Transition Energy Levels')
    plt.xlim(-2, 2)
    plt.ylim(-0.5, 1.5)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('energy_levels.png', dpi=300)
    plt.close()


def plot_optical_lattice_potential(cooling_model):
    """
    Plot the optical lattice potential for different ground state sublevels.
    
    Args:
        cooling_model (SisyphusCooling): Cooling model instance
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate potentials across the lattice
    positions = np.linspace(0, cooling_model.lattice_period, 100)
    potentials = np.array([cooling_model.optical_lattice_potential(pos) for pos in positions])
    
    # Plot potentials for each ground state sublevel
    for m_F in range(-2, 3):
        idx = m_F + 2
        plt.plot(positions / cooling_model.lattice_period, 
                 potentials[:, idx] / (hbar * GAMMA_D2), 
                 label=f'm_F = {m_F}')
    
    plt.xlabel('Position (λ/2)')
    plt.ylabel('Potential Energy (ħΓ)')
    plt.title('Optical Lattice Potential for Different Ground State Sublevels')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optical_lattice_potential.png', dpi=300)
    plt.close()


def plot_polarization_gradient(cooling_model):
    """
    Plot the polarization gradient along the optical lattice.
    
    Args:
        cooling_model (SisyphusCooling): Cooling model instance
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate polarization components across the lattice
    positions = np.linspace(0, cooling_model.lattice_period, 100)
    sigma_plus = []
    pi = []
    sigma_minus = []
    
    for pos in positions:
        sp, p, sm = cooling_model.polarization_at_position(pos)
        sigma_plus.append(sp)
        pi.append(p)
        sigma_minus.append(sm)
    
    # Plot polarization components
    plt.plot(positions / cooling_model.lattice_period, sigma_plus, 'g-', label='σ+')
    plt.plot(positions / cooling_model.lattice_period, pi, 'r-', label='π')
    plt.plot(positions / cooling_model.lattice_period, sigma_minus, 'b-', label='σ-')
    
    plt.xlabel('Position (λ/2)')
    plt.ylabel('Polarization Component')
    plt.title('Polarization Gradient Along the Optical Lattice')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('polarization_gradient.png', dpi=300)
    plt.close()


def main():
    """Main function to run the Sisyphus cooling simulation."""
    print("Sisyphus Cooling Model for Rubidium-87 Atoms")
    print("--------------------------------------------")
    print(f"Recoil temperature: {RECOIL_TEMPERATURE/uK_to_K:.2f} μK")
    print(f"Doppler temperature: {DOPPLER_TEMPERATURE/uK_to_K:.2f} μK")
    
    # Create cooling model
    cooling_model = SisyphusCooling(n_atoms=100, n_positions=100)
    
    # Plot energy levels
    print("\nGenerating energy level diagram...")
    plot_energy_levels()
    
    # Plot optical lattice potential
    print("Generating optical lattice potential plot...")
    plot_optical_lattice_potential(cooling_model)
    
    # Plot polarization gradient
    print("Generating polarization gradient plot...")
    plot_polarization_gradient(cooling_model)
    
    # Simulate Sisyphus cooling
    print("\nSimulating Sisyphus cooling...")
    cooling_model.reset_atoms(initial_temp=100e-6)  # 100 μK initial temperature
    
    # Define callback to print progress
    def progress_callback(step, time, temp):
        if step % 100 == 0:
            print(f"Step {step}: t = {time*1e3:.2f} ms, T = {temp/uK_to_K:.2f} μK")
    
    # Run simulation
    sisyphus_times, sisyphus_temps, positions, velocities = cooling_model.simulate(
        total_time=5e-3,  # 5 ms
        dt=1e-6,          # 1 μs
        callback=progress_callback
    )
    
    # Simulate Doppler cooling for comparison
    print("\nSimulating Doppler cooling for comparison...")
    doppler_times, doppler_temps = cooling_model.simulate_doppler_cooling(
        total_time=5e-3,  # 5 ms
        dt=1e-6,          # 1 μs
        callback=progress_callback
    )
    
    # Plot cooling comparison
    print("\nGenerating cooling comparison plot...")
    plot_cooling_comparison(sisyphus_times, sisyphus_temps, doppler_times, doppler_temps)
    
    # Plot spatial distribution
    print("Generating spatial distribution plots...")
    plot_spatial_distribution(positions, velocities, sisyphus_times)
    
    print("\nSimulation complete. Results saved as PNG files.")


if __name__ == "__main__":
    main()
