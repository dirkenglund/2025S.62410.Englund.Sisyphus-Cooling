/**
 * Sisyphus Cooling Debug Script
 * 
 * This script adds debugging functionality and fixes to 
 * ensure the simulation runs properly
 */

// Wait for the page to fully load
window.addEventListener('load', function() {
  console.log('Debug script loaded');
  
  // Fix 1: Add direct event listeners to replace the conflicting ones
  const startButton = document.getElementById('start-sim-btn');
  const resetButton = document.getElementById('reset-sim-btn');
  
  if (startButton) {
    console.log('Start button found, adding direct event listener');
    // Remove existing listeners and add new one
    startButton.replaceWith(startButton.cloneNode(true));
    document.getElementById('start-sim-btn').addEventListener('click', function() {
      console.log('Start button clicked');
      runSimulation();
    });
  } else {
    console.error('Start button not found!');
  }
  
  if (resetButton) {
    console.log('Reset button found, adding direct event listener');
    resetButton.replaceWith(resetButton.cloneNode(true));
    document.getElementById('reset-sim-btn').addEventListener('click', function() {
      console.log('Reset button clicked');
      resetSimulation();
    });
  } else {
    console.error('Reset button not found!');
  }
  
  // Fix 2: Show the simulation tab on initial load to make UI more accessible
  const simulationTab = document.getElementById('simulation-tab');
  if (simulationTab) {
    simulationTab.click();
  }
  
  // Fix 3: Add diagnostic info and clear status messages
  const statusElement = document.getElementById('simulation-status');
  if (statusElement) {
    statusElement.textContent = 'Ready to run simulation (Debug mode active)';
  }
  
  console.log('Diagnostics complete. Simulation controls should now work.');
});

// Direct simulation function to bypass any scope issues
function runSimulation() {
  // Show loading indicator
  document.getElementById('start-sim-btn').disabled = true;
  document.getElementById('start-sim-btn').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Simulating...';
  
  // Get simulation parameters
  const detuning = parseFloat(document.getElementById('detuning-slider').value);
  const rabi = parseFloat(document.getElementById('rabi-slider').value);
  const initialTemp = parseFloat(document.getElementById('initial-temp-slider').value);
  
  // Update simulation status
  const statusElement = document.getElementById('simulation-status');
  if (statusElement) {
    statusElement.textContent = `Running simulation with: Detuning=${detuning}Γ, Rabi=${rabi}Γ, Initial Temp=${initialTemp}μK`;
  }
  
  // Update initial values in the results table
  document.getElementById('initial-temperature').textContent = initialTemp + ' μK';
  document.getElementById('initial-energy').textContent = (initialTemp * 1.38e-23 * 1e6).toExponential(2) + ' J';
  document.getElementById('initial-velocity-spread').textContent = (Math.sqrt(initialTemp * 1.38e-23 * 1e6 / (86.909180527 * 1.66053886e-27))).toFixed(3) + ' m/s';
  
  console.log(`Calling API with parameters: initialTemp=${initialTemp}, detuning=${detuning}, rabiFreq=${rabi}`);
  
  // Call the Flask API for simulation data
  const apiUrl = `${window.location.origin}/api/simulate-cooling?initialTemp=${initialTemp}&detuning=${detuning}&rabiFreq=${rabi}`;
  fetch(apiUrl)
    .then(response => {
      console.log('API response received:', response.status);
      if (!response.ok) {
        throw new Error(`Network response error: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Simulation data received:', data);
      
      // Use real physics data from our backend
      const times = data.times.map(t => t * 1000); // Convert s to ms
      const sisyphusTempsMuK = data.sisyphus_temps;
      const dopplerTempsMuK = data.doppler_temps;
      
      console.log(`Temperatures: Start=${sisyphusTempsMuK[0]}μK, End=${sisyphusTempsMuK[sisyphusTempsMuK.length-1]}μK`);
      
      // Update temperature plot
      Plotly.react('temperature-plot', [
        {
          x: times,
          y: sisyphusTempsMuK,
          name: 'Sisyphus Cooling',
          type: 'scatter',
          mode: 'lines',
          line: { color: '#1f77b4', width: 2 }
        },
        {
          x: times,
          y: dopplerTempsMuK,
          name: 'Doppler Cooling',
          type: 'scatter',
          mode: 'lines',
          line: { color: '#ff7f0e', width: 2 }
        }
      ], {
        title: `Temperature vs. Time (Detuning: ${detuning}Γ, Rabi: ${rabi}Γ)`,
        xaxis: { title: 'Time (ms)' },
        yaxis: { title: 'Temperature (μK)', type: 'log' },
        showlegend: true,
        legend: { x: 0, y: 1 }
      });
      
      // Get final temperature
      const finalTemp = sisyphusTempsMuK[sisyphusTempsMuK.length - 1];
      
      // Update velocity distribution
      updateVelocityDistribution(finalTemp);
      
      // Update simulation status
      if (statusElement) {
        statusElement.className = 'alert alert-success';
        statusElement.textContent = `Simulation complete! Final temperature: ${finalTemp.toFixed(2)} μK`;
      }
      
      // Re-enable the button
      document.getElementById('start-sim-btn').disabled = false;
      document.getElementById('start-sim-btn').innerHTML = 'Start Simulation';
      
      console.log('Simulation completed successfully');
    })
    .catch(error => {
      console.error('Error running simulation:', error);
      
      // Show error in status
      if (statusElement) {
        statusElement.className = 'alert alert-danger';
        statusElement.textContent = `Error: ${error.message}`;
      }
      
      // Re-enable the button
      document.getElementById('start-sim-btn').disabled = false;
      document.getElementById('start-sim-btn').innerHTML = 'Start Simulation';
    });
}

// Create a velocity distribution based on final temperature
function updateVelocityDistribution(temperature) {
  console.log(`Updating velocity distribution for temperature ${temperature}μK`);
  
  // Calculate sigma for velocity spread
  const sigma = Math.sqrt(temperature * 1.38e-23 * 1e6 / (86.909180527 * 1.66053886e-27));
  
  // Generate velocity values
  const velocities = [];
  const distribution = [];
  const vMax = 0.5; // Max velocity in m/s
  
  for (let i = 0; i < 50; i++) {
    const v = -vMax + (i * 2 * vMax / 49);
    velocities.push(v);
    distribution.push(Math.exp(-(v * v) / (2 * sigma * sigma)) / (Math.sqrt(2 * Math.PI) * sigma));
  }
  
  // Update the plot
  Plotly.react('velocity-plot', [{
    x: velocities,
    y: distribution,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    line: { color: 'blue', width: 2 }
  }], {
    title: `Velocity Distribution (T = ${temperature.toFixed(2)} μK)`,
    xaxis: { title: 'Velocity (m/s)' },
    yaxis: { title: 'Probability Density' }
  });
  
  // Update results table
  document.getElementById('final-temperature').textContent = temperature.toFixed(2) + ' μK';
  document.getElementById('final-energy').textContent = (temperature * 1.38e-23 * 1e6).toExponential(2) + ' J';
  document.getElementById('final-velocity-spread').textContent = sigma.toFixed(3) + ' m/s';
  
  console.log(`Velocity distribution updated. Sigma = ${sigma}`);
}
