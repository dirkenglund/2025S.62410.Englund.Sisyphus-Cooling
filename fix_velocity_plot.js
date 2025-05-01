/**
 * This script contains enhanced implementations for the Sisyphus cooling simulation
 * focusing on properly updating the velocity distribution plot
 */

// Helper function to create velocity distribution data based on temperature
function createVelocityDistribution(finalTemp) {
  const rb87Mass = 1.4431608971290477e-25; // kg
  const kB = 1.38e-23; // Boltzmann constant
  
  // Calculate velocity spread (sigma) based on Maxwell-Boltzmann distribution
  const tempInK = finalTemp * 1e-6; // Convert μK to K
  const sigma = Math.sqrt(kB * tempInK / rb87Mass);
  
  // Create velocity distribution data points
  const velocities = [];
  const distribution = [];
  
  // Use more points for a smoother distribution
  const numPoints = 50;
  const vMax = 0.5; // Max velocity in m/s
  
  for (let i = 0; i < numPoints; i++) {
    const v = -vMax + (i * 2 * vMax / (numPoints - 1));
    velocities.push(v);
    // Maxwell-Boltzmann distribution for 1D
    distribution.push(Math.exp(-(v * v) / (2 * sigma * sigma)) / (Math.sqrt(2 * Math.PI) * sigma));
  }
  
  return {
    velocities,
    distribution,
    sigma
  };
}

// Enhanced function to initialize plots
function initializeVelocityPlot() {
  const initialTemp = 100; // Initial temperature in μK
  const { velocities, distribution } = createVelocityDistribution(initialTemp);
  
  Plotly.newPlot('velocity-plot', [{
    x: velocities,
    y: distribution,
    type: 'scatter',
    mode: 'lines',
    name: 'Velocity Distribution',
    line: {
      width: 2,
      color: 'rgb(0, 120, 212)'
    },
    fill: 'tozeroy',
    fillcolor: 'rgba(0, 120, 212, 0.2)'
  }], {
    title: `Velocity Distribution (T = ${initialTemp.toFixed(2)} μK)`,
    xaxis: {
      title: 'Velocity (m/s)',
      range: [-0.5, 0.5]
    },
    yaxis: {
      title: 'Probability Density'
    },
    margin: { t: 50, r: 10, b: 50, l: 60 },
    plot_bgcolor: '#f8f9fa',
    paper_bgcolor: '#ffffff'
  });
}

// Enhanced function to update velocity plot
function updateVelocityPlot(finalTemp) {
  console.log('Updating velocity plot with temperature:', finalTemp, 'μK');
  
  const { velocities, distribution, sigma } = createVelocityDistribution(finalTemp);
  
  // Use Plotly.react for a complete redraw
  Plotly.react('velocity-plot', [{
    x: velocities,
    y: distribution,
    type: 'scatter',
    mode: 'lines',
    name: 'Velocity Distribution',
    line: {
      width: 2,
      color: 'rgb(0, 120, 212)'
    },
    fill: 'tozeroy',
    fillcolor: 'rgba(0, 120, 212, 0.2)'
  }], {
    title: `Velocity Distribution (T = ${finalTemp.toFixed(2)} μK)`,
    xaxis: {
      title: 'Velocity (m/s)',
      range: [-0.5, 0.5]
    },
    yaxis: {
      title: 'Probability Density'
    },
    margin: { t: 50, r: 10, b: 50, l: 60 },
    plot_bgcolor: '#f8f9fa',
    paper_bgcolor: '#ffffff'
  });
  
  // Update related DOM elements with the final temperature
  document.getElementById('final-temperature').textContent = finalTemp.toFixed(2) + ' μK';
  document.getElementById('final-energy').textContent = (finalTemp * 1.38e-23 * 1e-6).toExponential(2) + ' J';
  document.getElementById('final-velocity-spread').textContent = sigma.toFixed(3) + ' m/s';
  
  // Also update the atom animation speed based on temperature
  updateAtomSpeed(finalTemp);
}

// Update atom animation speed based on temperature
function updateAtomSpeed(temperature) {
  // Scale animation speed based on temperature (atoms move faster at higher temperatures)
  const baseSpeed = 0.5;
  const speed = baseSpeed * Math.sqrt(temperature / 10); // Rough approximation
  
  // Store the speed value in a data attribute for the animation function to use
  document.getElementById('lattice-visualization').dataset.speed = speed.toString();
  
  console.log('Updated atom animation speed to', speed, 'based on temperature', temperature, 'μK');
}
