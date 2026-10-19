# Topological Insulator Band Structure & Strain Modeling

Computational modeling of 2D Kane-Mele topological insulators under uniaxial mechanical strain to evaluate material robustness for defense microelectronics and spintronic applications.

## Overview
This repository contains a tight binding calculation developed in Python using `PythTB`. The project models spin orbit coupling (SOC) and evaluates the evolution of the topological band gap ($\Delta E$) under uniaxial lattice strain from 0% to 10%.

## Key Findings
- **0% Strain Gap:** $\approx 0.520\text{ eV}$ (Inverted gap protected by spin-orbit coupling).
- **Strain Sweep (0–10%):** The topological band gap remains non-zero across the entire deformation range, demonstrating structural robustness against quantum phase transitions.
- **Application:** Identifies candidate topological material architectures capable of retaining low-dissipation edge-state transport in high-stress, harsh naval operating environments.

## Repository Structure
- `topological_project.py`: Generates comparative band structure plots for 0% vs 5% uniaxial strain.
- `strain_sweep_project.py`: Conducts a continuous strain sweep (0–10%) and exports quantitative gap data.
- `topological_gap_vs_strain.csv`: Exported numerical dataset containing strain fractions and corresponding $K$-point band gaps.
- `topological_strain_analysis.png`: High-resolution dual-panel band structure comparison plot.
- `gap_vs_strain_plot.png`: Quantitative plot displaying band gap evolution as a function of strain.

## Dependencies
- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- PythTB (`pip install pythtb`)
