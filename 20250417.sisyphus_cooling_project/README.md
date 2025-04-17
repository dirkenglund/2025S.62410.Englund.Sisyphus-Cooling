# Sisyphus Cooling Project

This repository contains an interactive educational simulation of Sisyphus cooling for Rubidium-87 atoms, implemented using QuTiP and Streamlit.

## Installation Instructions for Apple Silicon

### Prerequisites
- Python 3.8+ (preferably Python 3.10)
- pip (package installer for Python)

### Setup Instructions

1. **Clone or download this repository to your local machine**

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   
   For Apple Silicon (M1/M2/M3), some packages may require special installation:
   
   ```bash
   # Install numpy, scipy first (optimized for Apple Silicon)
   pip install --upgrade pip
   pip install numpy scipy
   
   # Install QuTiP (may use Rosetta 2 translation)
   pip install qutip
   
   # Install remaining dependencies
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with QuTiP on Apple Silicon, you may need to install it with specific compiler flags:
   ```bash
   OPENBLAS=$(brew --prefix openblas) pip install qutip
   ```

4. **Run the application**
   ```bash
   cd src
   streamlit run sisyphus_cooling_demo.py
   ```

## Project Structure

- `src/` - Source code for the Sisyphus cooling simulation
  - `sisyphus_cooling_demo.py` - Main application file
  - `world_model.py` - Implementation of the physics concept relationships

- `data/` - Data files including images and simulation results

- `tests/` - Unit tests for the simulation
  - `test_sisyphus_cooling_demo.py` - Test suite for the simulation

- `docs/` - Documentation files

## Features

- Interactive simulation of Sisyphus cooling process
- Visualization of temperature evolution and atom retention
- Comparison with other cooling methods (Doppler, VSCPT, Raman, Evaporative)
- Interactive concept map showing relationships between physics concepts
- Detailed educational content explaining the physics principles

## Apple Silicon Compatibility Notes

This project should run efficiently on Apple Silicon Macs. The numerical computations use NumPy and SciPy, which have optimized versions for Apple Silicon. The visualization components use Plotly and Streamlit, which are compatible with ARM architecture.

If you encounter any performance issues, consider:
1. Ensuring you're using ARM-native Python (not Rosetta 2)
2. Using the latest versions of all dependencies
3. Reducing the number of atoms in the simulation for faster performance
