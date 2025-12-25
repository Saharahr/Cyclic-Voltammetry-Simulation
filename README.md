# Cyclic Voltammetry (CV) Simulation

A Python-based numerical simulation of cyclic voltammetry for a two-species redox system using the Crank-Nicolson finite difference method with Butler-Volmer electrode kinetics.

## Overview

This project simulates the electrochemical technique of **Cyclic Voltammetry (CV)**, which is widely used in electrochemistry to study redox reactions, determine formal potentials, and characterize electrode kinetics.

The simulation solves:
1. **Fick's Second Law** for diffusion of species in solution
2. **Butler-Volmer equation** for electrode kinetics

Using:
- **Crank-Nicolson method** (implicit, unconditionally stable, 2nd-order accurate)
- **Thomas algorithm** for efficient tridiagonal matrix solving
- **Picard iteration** for nonlinear coupling between current and concentration

---

## Physical System

The simulation models a simple one-electron redox reaction:

```
O + e⁻  ⇌  R
```

Where:
- **O** = Oxidized species (electron acceptor)
- **R** = Reduced species (electron donor)
- **e⁻** = Electron

### Sign Convention (IUPAC/Ossila)

| Process | Current Sign |
|---------|--------------|
| Oxidation (R → O + e⁻) | Positive (anodic) |
| Reduction (O + e⁻ → R) | Negative (cathodic) |

---

## Features

- ✅ Two-species diffusion (O and R) with independent diffusion coefficients
- ✅ Butler-Volmer electrode kinetics with adjustable rate constant
- ✅ Multiple CV cycles to observe approach to steady state
- ✅ Automatic peak detection and analysis
- ✅ Comparison with Randles-Sevcik theoretical predictions
- ✅ Fully commented code for educational purposes
- ✅ Publication-quality plotting

---

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib

### Setup

```bash
# Clone or download the project
git clone https://github.com/yourusername/cv-simulation.git
cd cv-simulation

# Install dependencies
pip install numpy matplotlib

# Optional: Install SciPy for faster matrix solving
pip install scipy
```
---

## Usage

### Basic Usage

```bash
python cv_reversible.py
```

### Custom Parameters

```python
from cv_reversible.py import simulate_cv_stable, plot_results

# Run simulation with custom parameters
result = simulate_cv_stable(
    E0=0.0,           # Formal potential [V]
    E_start=-0.25,    # Starting potential [V]
    E_switch=0.25,    # Switching potential [V]
    v=0.2,            # Scan rate [V/s]
    k0=0.2,           # Rate constant [cm/s]
    C_total=1e-6,     # Concentration [mol/cm³]
    n_cycles=3,       # Number of cycles
)

# Plot results
plot_results(result, save_path="cv_stable.png")
```

### Output

The simulation outputs:
1. **Console**: Peak analysis for each cycle
2. **Plots**: CV curve and potential waveform saved to `./plots/`

---

## Results

### Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| E⁰ | 0.0 V | Formal potential |
| E_start | -0.25 V | Starting potential |
| E_switch | +0.25 V | Switching potential |
| v | 0.2 V/s | Scan rate |
| n | 1 | Electrons transferred |
| α | 0.5 | Transfer coefficient |
| k⁰ | 0.2 cm/s | Standard rate constant |
| D_O, D_R | 7×10⁻⁶ cm²/s | Diffusion coefficients |
| C | 1 mM | Concentration |
| A | 0.071 cm² | Electrode area |

### Peak Analysis

| Cycle | E_pa (V) | i_pa (µA) | E_pc (V) | i_pc (µA) | ΔE_p (mV) | \|i_pa/i_pc\| |
|-------|----------|-----------|----------|-----------|-----------|---------------|
| 1 | +0.0290 | +22.47 | -0.0297 | -16.17 | 58.7 | 1.389 |
| 2 | +0.0295 | +20.68 | -0.0296 | -16.98 | 59.1 | 1.218 |
| 3 | +0.0296 | +20.19 | -0.0296 | -17.32 | 59.1 | 1.166 |

### Theoretical Comparison

| Metric | Simulated | Theory | Agreement |
|--------|-----------|--------|-----------|
| ΔE_p | 59.1 mV | 59.0 mV | ✅ Excellent |
| i_pa (Cycle 1) | 22.47 µA | 22.57 µA (Randles-Sevcik) | ✅ 99.6% |
| \|i_pa/i_pc\| | 1.17 → 1.0 | 1.0 (steady state) | ✅ Approaching |

### Key Observations

1. **Peak Separation (ΔE_p ≈ 59 mV)**: Confirms reversible electron transfer kinetics, matching the theoretical value of 59/n mV at 25°C.

2. **Peak Current Ratio**: 
   - Cycle 1: |i_pa/i_pc| = 1.39 (asymmetric due to initial conditions)
   - Cycle 3: |i_pa/i_pc| = 1.17 (approaching unity)
   - With more cycles, this ratio approaches 1.0 as expected for a reversible system.

3. **Randles-Sevcik Validation**: The simulated peak current (22.47 µA) matches the theoretical prediction (22.57 µA) within 0.5%, confirming correct implementation of diffusion physics.

---

## Theory

### Governing Equations

#### 1. Diffusion (Fick's Second Law)

$$\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}$$

Where:
- C = concentration [mol/cm³]
- D = diffusion coefficient [cm²/s]
- x = distance from electrode [cm]
- t = time [s]

#### 2. Electrode Kinetics (Butler-Volmer)

$$j = nFk^0 \left[ C_O(0) e^{-\alpha \beta \eta} - C_R(0) e^{(1-\alpha) \beta \eta} \right]$$

Where:
- j = current density [A/cm²]
- k⁰ = standard rate constant [cm/s]
- η = E - E⁰ = overpotential [V]
- α = transfer coefficient
- β = nF/RT ≈ 38.9 V⁻¹ at 25°C

#### 3. Randles-Sevcik Equation (Peak Current)

$$i_p = 0.4463 \cdot nFAC^* \sqrt{\frac{nFvD}{RT}}$$

This equation predicts the peak current for a reversible CV and is used to validate the simulation.

### Numerical Methods

| Method | Purpose | Complexity |
|--------|---------|------------|
| Crank-Nicolson   | Solve diffusion equation | O(Nx) per step |
| Thomas Algorithm | Solve tridiagonal system | O(Nx) |
| Picard Iteration | Handle nonlinear coupling | 10-50 iterations |

### Boundary Conditions

| Location | Type | Condition |
|----------|------|-----------|
| x = 0 (electrode) | Flux BC | J = -D ∂C/∂x = j/(nF) |
| x = L (bulk) | Neumann | ∂C/∂x = 0 |

---

## Parameters

### Electrochemical Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Formal potential | E⁰ | 0.0 V | -2 to +2 V | Equilibrium potential |
| Scan rate | v | 0.2 V/s | 0.001-10 V/s | Potential sweep rate |
| Transfer coefficient | α | 0.5 | 0.3-0.7 | Symmetry of barrier |
| Rate constant | k⁰ | 0.2 cm/s | 10⁻⁷-1 cm/s | Electron transfer rate |

### Numerical Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Grid points | Nx | 301 | Spatial resolution |
| Time step | dt | 2×10⁻⁴ s | Temporal resolution |
| Relaxation | relax | 0.08 | Picard iteration damping |
| Tolerance | tol | 10⁻¹⁰ | Convergence criterion |

### Stability Guidelines

```
λ = D·dt/dx² < 1    (for accuracy)
Nx > 10·δ/L         (resolve diffusion layer)
relax ≈ 0.05-0.10   (prevent oscillation)
```

---

## Performance

| Configuration | Time | Accuracy |
|---------------|------|----------|
| Nx=601, dt=2e-4, 3 cycles | ~10 min | Excellent |
| Nx=301, dt=2e-4, 3 cycles | ~3 min | Very Good |
| Nx=101, dt=1e-3, 1 cycle | ~30 sec | Good |

---

## References

1. Bard, A.J. and Faulkner, L.R. *Electrochemical Methods: Fundamentals and Applications*, 2nd ed., Wiley, 2001.

2. Compton, R.G. and Banks, C.E. *Understanding Voltammetry*, 2nd ed., Imperial College Press, 2011.

3. Nicholson, R.S. and Shain, I. "Theory of Stationary Electrode Polarography", *Anal. Chem.*, 1964, 36, 706-723.

4. Crank, J. and Nicolson, P. "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type", *Proc. Camb. Phil. Soc.*, 1947, 43, 50-67.

5. Elgrishi et al., A Practical Beginner’s Guide to Cyclic Voltammetry, Journal of Chemical Education, 2017.
---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
---
## Contact

For questions or feedback, please open an issue on GitHub.