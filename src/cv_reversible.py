"""
Two-species 1D Cyclic Voltammetry (CV) Simulation
============================================================================

PHYSICAL SYSTEM:
================
    Redox couple:   O + n e-  <-->  R
    
    At the electrode surface, oxidized species (O) can gain electrons to become
    reduced species (R), or R can lose electrons to become O.

WHAT THIS CODE SOLVES:
======================
    1. Fick's Second Law (diffusion): ∂C/∂t = D ∂²C/∂x²
       - Describes how species concentrations change due to diffusion
       - Species move from high concentration to low concentration
       
    2. Butler-Volmer equation (electrode kinetics):
       j = nFk₀[Cₒ·exp(-αβη) - Cᵣ·exp((1-α)βη)]
       - Describes how fast electron transfer occurs at electrode surface
       - η = E - E⁰ is the overpotential (driving force for reaction)

NUMERICAL METHODS USED:
=======================
    1. Crank-Nicolson method: Solves diffusion equation (stable, 2nd order accurate)
    2. Thomas algorithm: Efficiently solves the tridiagonal matrix system
    3. Picard iteration: Handles the nonlinear coupling between current and concentration

SIGN CONVENTION (Ossila-style):
===============================
    - Oxidation (R → O + e⁻): POSITIVE current (anodic)
    - Reduction (O + e⁻ → R): NEGATIVE current (cathodic)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================
# These are universal constants

F = 96485.33212       # Faraday constant [C/mol] - charge of 1 mole of electrons
R_gas = 8.314462618   # Universal gas constant [J/(mol·K)] - relates energy to temperature


# =============================================================================
# UTILITY FUNCTIONS

def safe_exp(x: float, clamp: float = 50.0) -> float:
    """
    Compute exponential with protection against overflow/underflow.
    
    WHY NEEDED:
        In Butler-Volmer equation, we compute exp(±αβη) where β ≈ 38.9 V⁻¹
        If η = 2V, then αβη ≈ 39, and exp(39) ≈ 10^17 (huge!)
        If η = -2V, then exp(-39) ≈ 10^-17 (tiny!)
        
        Without clamping, we get numerical overflow/underflow errors.
    
    HOW IT WORKS:
        Limits the exponent to [-clamp, +clamp] before computing exp()
        exp(50) ≈ 5×10^21 is large but still representable
        exp(-50) ≈ 2×10^-22 is small but still representable
    
    Parameters:
        x: The exponent value
        clamp: Maximum absolute value allowed (default 50)
    
    Returns:
        exp(x) with x clamped to [-clamp, +clamp]
    """
    return float(np.exp(np.clip(x, -clamp, clamp)))


def triangular_potential_multicycle(
    t: np.ndarray, 
    E_start: float, 
    E_switch: float, 
    v: float,
    n_cycles: int = 1
) -> np.ndarray:
    """
    Generate triangular potential waveform for cyclic voltammetry.
    
    WHAT THIS DOES:
        Creates the E(t) signal that looks like this for one cycle:
        
        E_switch ──────►  /\
                         /  \
                        /    \
        E_start  ──────►      \/
                        
                       |──────|──────|
                         t_half  t_half
                       |─── t_cycle ──|
    
    THE PHYSICS:
        - Potential changes linearly with time at rate v (scan rate)
        - Forward scan: E goes from E_start toward E_switch
        - Reverse scan: E returns from E_switch to E_start
        - Multiple cycles repeat this pattern
    
    Parameters:
        t: Array of time points [seconds]
        E_start: Starting potential [V]
        E_switch: Switching potential (vertex) [V]
        v: Scan rate [V/s] - how fast potential changes
        n_cycles: Number of complete cycles
    
    Returns:
        E: Array of potential values at each time point [V]
    
    Example:
        E_start = -0.25V, E_switch = +0.25V, v = 0.2 V/s
        t_half = |0.25 - (-0.25)| / 0.2 = 2.5 seconds per half-cycle
        t_cycle = 5.0 seconds per full cycle
    """
    # Calculate timing parameters
    dE = E_switch - E_start           # Total potential change [V]
    t_half = abs(dE) / v              # Time for half cycle (forward OR reverse) [s]
    t_cycle = 2.0 * t_half            # Time for complete cycle [s]
    s = 1.0 if dE >= 0 else -1.0      # Sign: +1 if scanning positive, -1 if scanning negative
    
    # Build potential array
    E = np.empty_like(t, dtype=float)
    for k, tk in enumerate(t):
        # Use modulo to find position within current cycle
        # This automatically handles multiple cycles
        t_in_cycle = tk % t_cycle
        
        if t_in_cycle <= t_half:
            # FORWARD SCAN: Moving from E_start toward E_switch
            # E increases (or decreases) linearly with time
            E[k] = E_start + s * v * t_in_cycle
        else:
            # REVERSE SCAN: Moving from E_switch back to E_start
            # E decreases (or increases) linearly with time
            E[k] = E_switch - s * v * (t_in_cycle - t_half)
    
    return E


def clamp_scalar(x: float, lo: float, hi: float) -> float:
    """
    Clamp a value to be within [lo, hi] range.
    
    WHY NEEDED:
        - Prevents current from becoming unrealistically large
        - Helps stabilize the Picard iteration
        - Acts as a safety limit
    
    Parameters:
        x: Value to clamp
        lo: Minimum allowed value
        hi: Maximum allowed value
    
    Returns:
        x if lo ≤ x ≤ hi, otherwise lo or hi
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def finite_scalar(x: float, default: float = 0.0) -> float:
    """
    Return x if it's a valid finite number, otherwise return default.
    
    WHY NEEDED:
        Numerical errors can produce NaN (Not a Number) or Inf (Infinity).
        This function catches those and replaces them with a safe default.
    
    Parameters:
        x: Value to check
        default: Value to return if x is NaN or Inf
    
    Returns:
        x if finite, else default
    """
    return x if np.isfinite(x) else default


# =============================================================================
# TRIDIAGONAL MATRIX SOLVER (THOMAS ALGORITHM)
# =============================================================================

def thomas_solve_inplace(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal system of equations using the Thomas algorithm.
    
    WHAT IS A TRIDIAGONAL SYSTEM?
    =============================
        A system where the matrix has non-zero elements only on three diagonals:
        
        | b₀  c₀   0   0   0  |   | x₀ |   | d₀ |
        | a₀  b₁  c₁   0   0  |   | x₁ |   | d₁ |
        |  0  a₁  b₂  c₂   0  | × | x₂ | = | d₂ |
        |  0   0  a₂  b₃  c₃  |   | x₃ |   | d₃ |
        |  0   0   0  a₃  b₄  |   | x₄ |   | d₄ |
        
        - b: main diagonal (center)
        - a: lower diagonal (below main) - has n-1 elements
        - c: upper diagonal (above main) - has n-1 elements
        - d: right-hand side vector
    
    WHY THIS STRUCTURE?
    ===================
        The Crank-Nicolson discretization of the diffusion equation:
        
        -λ/2 · Cᵢ₋₁ⁿ⁺¹ + (1+λ) · Cᵢⁿ⁺¹ - λ/2 · Cᵢ₊₁ⁿ⁺¹ = RHS
        
        Each equation involves only 3 neighboring points, giving tridiagonal structure.
    
    WHY THOMAS ALGORITHM?
    =====================
        - General matrix solve: O(n³) operations
        - Thomas algorithm: O(n) operations - MUCH faster!
        - Perfect for 1D diffusion problems
    
    HOW IT WORKS (2 phases):
    ========================
        Phase 1 - Forward Elimination:
            Eliminate the lower diagonal by subtracting rows
            Transform into upper triangular form
            
        Phase 2 - Back Substitution:
            Solve from bottom to top
            Each unknown depends only on the one below it
    
    Parameters:
        a: Lower diagonal coefficients (length n-1)
        b: Main diagonal coefficients (length n)
        c: Upper diagonal coefficients (length n-1)
        d: Right-hand side vector (length n) - MODIFIED IN PLACE to contain solution
    
    Returns:
        d: The solution vector x (overwrites input d)
    
    Raises:
        FloatingPointError: If matrix is singular or near-singular
    """
    n = b.size
    eps = 1e-30  # Small number to detect division by zero

    # =========================================================================
    # PHASE 1: FORWARD ELIMINATION
    # =========================================================================
    # Goal: Eliminate lower diagonal (a), transform to upper triangular
    #
    # For each row i (starting from row 1):
    #   1. Compute multiplier: w = a[i-1] / b[i-1]
    #   2. Subtract w × (row i-1) from row i
    #   3. This zeros out a[i-1] and modifies b[i] and d[i]
    
    for i in range(1, n):
        denom = b[i - 1]
        
        # Check for division by zero (singular matrix)
        if (not np.isfinite(denom)) or (abs(denom) < eps):
            raise FloatingPointError(f"Thomas breakdown: b[{i-1}]={denom}")
        
        w = a[i - 1] / denom        # Multiplier for elimination
        b[i] -= w * c[i - 1]        # Update diagonal element
        d[i] -= w * d[i - 1]        # Update RHS

    # =========================================================================
    # PHASE 2: BACK SUBSTITUTION
    # =========================================================================
    # Now we have upper triangular form:
    #   | b₀  c₀   0  |   | x₀ |   | d₀ |
    #   |  0  b₁' c₁  | × | x₁ | = | d₁'|
    #   |  0   0  b₂' |   | x₂ |   | d₂'|
    #
    # Solve from bottom up:
    #   x₂ = d₂' / b₂'
    #   x₁ = (d₁' - c₁·x₂) / b₁'
    #   x₀ = (d₀ - c₀·x₁) / b₀
    
    # Solve last equation first
    if (not np.isfinite(b[-1])) or (abs(b[-1]) < eps):
        raise FloatingPointError(f"Thomas breakdown: b[-1]={b[-1]}")
    d[-1] /= b[-1]

    # Solve remaining equations from bottom to top
    for i in range(n - 2, -1, -1):
        denom = b[i]
        if (not np.isfinite(denom)) or (abs(denom) < eps):
            raise FloatingPointError(f"Thomas breakdown: b[{i}]={denom}")
        d[i] = (d[i] - c[i] * d[i + 1]) / denom

    return d  # Solution is now stored in d


# =============================================================================
# CRANK-NICOLSON MATRIX BUILDER
# =============================================================================

def build_cn_lhs_neumann_right(Nx: int, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the LEFT-HAND SIDE (LHS) matrix for Crank-Nicolson scheme.
    
    WHAT IS CRANK-NICOLSON?
    =======================
        A numerical method to solve the diffusion equation:
        
            ∂C/∂t = D ∂²C/∂x²
        
        It averages between explicit (old time) and implicit (new time):
        
            (Cⁿ⁺¹ - Cⁿ)/Δt = D/2 × [∂²Cⁿ⁺¹/∂x² + ∂²Cⁿ/∂x²]
        
        Rearranging gives:  A · Cⁿ⁺¹ = b
        
        This function builds the matrix A (LHS).
    
    THE DISCRETIZED EQUATIONS:
    ==========================
        For interior points (i = 1, 2, ..., Nx-2):
            -λ/2 · Cᵢ₋₁ⁿ⁺¹ + (1+λ) · Cᵢⁿ⁺¹ - λ/2 · Cᵢ₊₁ⁿ⁺¹ = RHS
        
        Where λ = D·Δt/Δx² (the "diffusion number")
        
    BOUNDARY CONDITIONS:
    ====================
        At x=0 (electrode, i=0): 
            Flux boundary condition (handled in RHS function)
            (1+λ)·C₀ⁿ⁺¹ - λ·C₁ⁿ⁺¹ = RHS₀
            
        At x=L (bulk, i=Nx-1):
            Neumann BC (zero gradient): Cₙₓ₋₁ = Cₙₓ₋₂
            This means: -1·Cₙₓ₋₂ⁿ⁺¹ + 1·Cₙₓ₋₁ⁿ⁺¹ = 0
    
    THE MATRIX STRUCTURE:
    =====================
        Row 0 (electrode):     | 1+λ   -λ    0    0   ...  0  |
        Row 1 (interior):      | -λ/2  1+λ  -λ/2  0   ...  0  |
        Row 2 (interior):      |  0   -λ/2  1+λ  -λ/2 ...  0  |
        ...
        Row Nx-1 (bulk BC):    |  0    0    ...  -1    1      |
    
    Parameters:
        Nx: Number of spatial grid points
        lam: λ = D·Δt/Δx² (diffusion number, dimensionless)
    
    Returns:
        a: Lower diagonal (length Nx-1)
        b: Main diagonal (length Nx)
        c: Upper diagonal (length Nx-1)
    """
    n = Nx
    
    # Initialize diagonal arrays
    a = np.zeros(n - 1, dtype=float)  # Lower diagonal
    b = np.zeros(n, dtype=float)       # Main diagonal
    c = np.zeros(n - 1, dtype=float)   # Upper diagonal

    # -------------------------------------------------------------------------
    # ROW 0: Electrode surface boundary (x = 0)
    # -------------------------------------------------------------------------
    # Flux BC is incorporated via the RHS, so LHS just has:
    # (1+λ)·C₀ - λ·C₁ = RHS₀
    b[0] = 1.0 + lam    # Main diagonal: coefficient of C₀
    c[0] = -lam         # Upper diagonal: coefficient of C₁

    # -------------------------------------------------------------------------
    # ROWS 1 to Nx-2: Interior points
    # -------------------------------------------------------------------------
    # Standard Crank-Nicolson discretization:
    # -λ/2·Cᵢ₋₁ + (1+λ)·Cᵢ - λ/2·Cᵢ₊₁ = RHS
    for r in range(1, n - 1):
        a[r - 1] = -0.5 * lam   # Lower diagonal: coefficient of Cᵢ₋₁
        b[r] = 1.0 + lam        # Main diagonal: coefficient of Cᵢ
        if r <= n - 2:
            c[r] = -0.5 * lam if r < n - 1 else 0.0  # Upper diagonal: coefficient of Cᵢ₊₁

    # -------------------------------------------------------------------------
    # ROW Nx-1: Bulk boundary (x = L)
    # -------------------------------------------------------------------------
    # Neumann BC: ∂C/∂x = 0 at x = L
    # Discretized as: Cₙₓ₋₁ = Cₙₓ₋₂
    # Rearranged: -Cₙₓ₋₂ + Cₙₓ₋₁ = 0
    a[n - 2] = -1.0   # Coefficient of Cₙₓ₋₂
    b[n - 1] = 1.0    # Coefficient of Cₙₓ₋₁

    return a, b, c


def cn_rhs_with_flux_neumann_right(
    C_old: np.ndarray,
    lam: float,
    dt: float,
    dx: float,
    J_new: float,
    J_prev: float,
) -> np.ndarray:
    """
    Build the RIGHT-HAND SIDE (RHS) vector for Crank-Nicolson scheme.
    
    WHAT THIS COMPUTES:
    ===================
        The RHS of:  A · Cⁿ⁺¹ = RHS
        
        RHS contains:
        - Contributions from old time step concentrations (known)
        - Flux boundary condition at electrode surface
    
    THE FLUX BOUNDARY CONDITION (Key concept!):
    ===========================================
        At the electrode (x=0), species are consumed/produced by the reaction.
        
        Fick's first law: J = -D · ∂C/∂x|ₓ₌₀
        
        Where J is the molar flux [mol/(cm²·s)]:
        - J > 0: Species flowing INTO the electrode (being consumed)
        - J < 0: Species flowing OUT of the electrode (being produced)
        
        Using a ghost node approach and Crank-Nicolson averaging:
        
        RHS₀ = (1-λ)·C₀ⁿ + λ·C₁ⁿ + (Δt/Δx)·(Jⁿ⁺¹ + Jⁿ)
                └──── diffusion ────┘   └─── flux term ───┘
        
        The flux term adds/removes material at the surface based on
        the electrochemical reaction rate.
    
    Parameters:
        C_old: Concentration array at previous time step (length Nx)
        lam: λ = D·Δt/Δx² (diffusion number)
        dt: Time step [s]
        dx: Spatial step [cm]
        J_new: Molar flux at new time step [mol/(cm²·s)]
        J_prev: Molar flux at previous time step [mol/(cm²·s)]
    
    Returns:
        rhs: Right-hand side vector (length Nx)
    """
    Nx = C_old.size
    rhs = np.empty(Nx, dtype=float)

    # -------------------------------------------------------------------------
    # ROW 0: Electrode surface with flux BC
    # -------------------------------------------------------------------------
    # RHS = (diffusion from old time step) + (flux contribution)
    # 
    # The flux term (dt/dx)·(J_new + J_prev) represents:
    # - Crank-Nicolson averaging of flux over old and new time
    # - Converts flux [mol/(cm²·s)] to concentration change [mol/cm³]
    #
    # Physical meaning:
    # - If J > 0: Reaction consumes species → concentration decreases
    # - If J < 0: Reaction produces species → concentration increases
    rhs[0] = (1.0 - lam) * C_old[0] + lam * C_old[1] + (dt / dx) * (J_new + J_prev)

    # -------------------------------------------------------------------------
    # ROWS 1 to Nx-2: Interior points (standard diffusion)
    # -------------------------------------------------------------------------
    # No flux here, just diffusion from neighboring points
    # RHS = λ/2·Cᵢ₋₁ⁿ + (1-λ)·Cᵢⁿ + λ/2·Cᵢ₊₁ⁿ
    for r in range(1, Nx - 1):
        rhs[r] = (1.0 - lam) * C_old[r] + 0.5 * lam * (C_old[r - 1] + C_old[r + 1])

    # -------------------------------------------------------------------------
    # ROW Nx-1: Bulk boundary (Neumann BC: zero gradient)
    # -------------------------------------------------------------------------
    # Equation is: Cₙₓ₋₁ - Cₙₓ₋₂ = 0, so RHS = 0
    rhs[Nx - 1] = 0.0
    
    return rhs


# =============================================================================
# MAIN CV SIMULATION FUNCTION
# =============================================================================

def simulate_cv_multicycle(
    *,
    E0=0.0,           # Formal potential [V] - where reaction is at equilibrium
    E_start=-0.25,    # Starting potential [V]
    E_switch=0.25,    # Switching (vertex) potential [V]
    v=0.2,            # Scan rate [V/s]
    n_electrons=1,    # Number of electrons transferred
    T=298.15,         # Temperature [K] (298.15 K = 25°C)
    alpha=0.5,        # Transfer coefficient (symmetry factor)
    k0=0.2,           # Standard rate constant [cm/s]
    DO=7e-6,          # Diffusion coefficient of O [cm²/s]
    DR=7e-6,          # Diffusion coefficient of R [cm²/s]
    C_total=1e-6,     # Total concentration [mol/cm³]
    A=0.071,          # Electrode area [cm²]
    L=0.10,           # Domain length [cm]
    Nx=301,           # Number of spatial grid points
    dt=5e-4,          # Time step [s]
    n_cycles=3,       # Number of CV cycles to simulate
    picard_max=50,    # Maximum Picard iterations per time step
    relax0=0.08,      # Initial relaxation factor for Picard iteration
    exp_clamp=50.0,   # Clamp value for exponentials
    tol_j=1e-10,      # Convergence tolerance for current density
    tol_c0=1e-12,     # Convergence tolerance for surface concentration
    J_cap_factor=50.0,  # Factor for capping maximum flux
) -> dict:
    """
    Simulate cyclic voltammetry for a two-species redox system.
    
    PHYSICAL SYSTEM:
    ================
        O + n e⁻ ⇌ R
        
        - O: Oxidized species (electron acceptor)
        - R: Reduced species (electron donor)
        - n: Number of electrons transferred
    
    WHAT THIS FUNCTION DOES:
    ========================
        1. Sets up spatial grid and initial conditions
        2. Time-steps through the potential sweep
        3. At each time step:
           a. Compute potential E(t) from triangular waveform
           b. Use Picard iteration to solve coupled problem:
              - Butler-Volmer gives current from surface concentrations
              - Diffusion equation evolves concentrations given flux
           c. Store current for output
    
    THE BUTLER-VOLMER EQUATION:
    ===========================
        j = nFk₀[Cₒ(0)·exp(-αβη) - Cᵣ(0)·exp((1-α)βη)]
        
        Where:
        - j: Current density [A/cm²]
        - η = E - E⁰: Overpotential [V]
        - β = nF/RT ≈ 38.9 V⁻¹ at 25°C
        - α: Transfer coefficient (typically 0.5)
        - k₀: Standard rate constant [cm/s]
        - Cₒ(0), Cᵣ(0): Surface concentrations [mol/cm³]
        
        Physical meaning:
        - First term: Reduction rate (O + e⁻ → R)
        - Second term: Oxidation rate (R → O + e⁻)
        - Net current is the difference
    
    PICARD ITERATION (Why needed?):
    ===============================
        The problem is NONLINEAR because:
        - Current j depends on surface concentrations Cₒ(0), Cᵣ(0)
        - Surface concentrations depend on flux J = j/(nF)
        - Flux affects how diffusion evolves concentrations
        - It's circular! j ↔ C(0) ↔ J ↔ diffusion ↔ C(0)
        
        Picard iteration resolves this by:
        1. Guess a current j
        2. Compute concentrations from diffusion with that flux
        3. Compute new current from Butler-Volmer with those concentrations
        4. Relax: j_new = relax·j_BV + (1-relax)·j_old
        5. Repeat until converged
        
        The relaxation factor (0 < relax < 1) prevents oscillation.
    
    Parameters:
        See parameter definitions above
    
    Returns:
        dict containing:
        - 'E': Potential array [V]
        - 'i': Current array [A]
        - 't': Time array [s]
        - 'cycle_data': List of dicts, one per cycle
        - 'CO_final': Final O concentration profile [mol/cm³]
        - 'CR_final': Final R concentration profile [mol/cm³]
    """
    
    # =========================================================================
    # SETUP: Compute derived parameters
    # =========================================================================
    
    # β = nF/RT: Appears in Butler-Volmer exponentials
    # At 25°C: β ≈ 38.9 V⁻¹ for n=1
    # Physical meaning: How sensitive reaction rate is to potential
    beta = n_electrons * F / (R_gas * T)
    
    # Spatial discretization
    dx = L / (Nx - 1)  # Grid spacing [cm]
    
    # Diffusion numbers (λ): Key stability parameter
    # λ = D·Δt/Δx²
    # Rule of thumb: λ < 1 for stability (CN is stable for any λ, but accuracy suffers if too large)
    lamO = DO * dt / dx**2  # For oxidized species
    lamR = DR * dt / dx**2  # For reduced species

    # =========================================================================
    # SETUP: Time and potential arrays
    # =========================================================================
    
    # Calculate cycle timing
    t_half = abs(E_switch - E_start) / v  # Time for half cycle [s]
    t_cycle = 2.0 * t_half                 # Time for full cycle [s]
    t_total = n_cycles * t_cycle           # Total simulation time [s]
    
    # Create time array
    t = np.arange(0.0, t_total + dt, dt)
    
    # Generate triangular potential waveform E(t)
    E = triangular_potential_multicycle(t, E_start, E_switch, v, n_cycles)

    # =========================================================================
    # SETUP: Initial conditions
    # =========================================================================
    
    # Start with only reduced species (R) present
    # This is typical: you start with your analyte in reduced form
    CO = np.full(Nx, 0.0, dtype=float)        # [O] = 0 everywhere
    CR = np.full(Nx, C_total, dtype=float)    # [R] = C_total everywhere

    # =========================================================================
    # SETUP: Build Crank-Nicolson LHS matrices
    # =========================================================================
    # These don't change during simulation, so build once
    
    aLO, bLO, cLO = build_cn_lhs_neumann_right(Nx, lamO)  # For O diffusion
    aLR, bLR, cLR = build_cn_lhs_neumann_right(Nx, lamR)  # For R diffusion

    # =========================================================================
    # SETUP: Initialize storage and variables
    # =========================================================================
    
    i = np.zeros_like(t, dtype=float)  # Current array [A]
    j = 0.0  # Current density [A/cm²], starts at zero
    
    # Fluxes from previous time step (for Crank-Nicolson averaging)
    JO_prev = 0.0  # Flux of O at previous time step [mol/(cm²·s)]
    JR_prev = 0.0  # Flux of R at previous time step [mol/(cm²·s)]

    # =========================================================================
    # SETUP: Safety limits for numerical stability
    # =========================================================================
    
    # Maximum expected flux (for capping unrealistic values)
    J_scale = max(DR * C_total / dx, 1e-30)
    J_cap = J_cap_factor * J_scale
    j_cap = n_electrons * F * J_cap  # Convert to current density

    # =========================================================================
    # MAIN TIME LOOP
    # =========================================================================
    
    for k in range(t.size):
        # Current potential and overpotential
        eta = E[k] - E0  # Overpotential: driving force for reaction
        
        # Save old values for Picard iteration
        CO_old = CO.copy()
        CR_old = CR.copy()
        j_old = j
        
        # Surface concentrations for convergence check
        cO0_old = CO[0]
        cR0_old = CR[0]
        
        # Reset relaxation factor for this time step
        relax = relax0

        # =====================================================================
        # PICARD ITERATION: Solve nonlinear coupled problem
        # =====================================================================
        
        for it in range(picard_max):
            # -----------------------------------------------------------------
            # Step 1: Get current surface concentrations
            # -----------------------------------------------------------------
            CO0 = max(finite_scalar(float(CO[0]), 0.0), 0.0)  # [O] at electrode
            CR0 = max(finite_scalar(float(CR[0]), 0.0), 0.0)  # [R] at electrode

            # -----------------------------------------------------------------
            # Step 2: Compute Butler-Volmer current density
            # -----------------------------------------------------------------
            # j = nFk₀[Cₒ·exp(-αβη) - Cᵣ·exp((1-α)βη)]
            #
            # exp(-αβη): Forward (reduction) rate factor
            #   - η > 0 (positive potential): This term decreases → less reduction
            #   - η < 0 (negative potential): This term increases → more reduction
            #
            # exp((1-α)βη): Reverse (oxidation) rate factor
            #   - η > 0: This term increases → more oxidation
            #   - η < 0: This term decreases → less oxidation
            
            ef = safe_exp(-alpha * beta * eta, clamp=exp_clamp)        # Reduction factor
            er = safe_exp((1.0 - alpha) * beta * eta, clamp=exp_clamp) # Oxidation factor
            
            # Butler-Volmer current density
            j_bv = n_electrons * F * k0 * (CO0 * ef - CR0 * er)
            j_bv = finite_scalar(j_bv, 0.0)  # Handle any NaN/Inf

            # -----------------------------------------------------------------
            # Step 3: Relaxation to ensure convergence
            # -----------------------------------------------------------------
            # Instead of jumping directly to j_bv, we move gradually:
            # j_new = relax·j_bv + (1-relax)·j_old
            #
            # If relax = 1: No relaxation, might oscillate
            # If relax = 0.1: Slow but stable convergence
            
            j_new = relax * j_bv + (1.0 - relax) * j_old
            j_new = finite_scalar(j_new, 0.0)
            j_new = clamp_scalar(j_new, -j_cap, +j_cap)  # Safety clamp

            # -----------------------------------------------------------------
            # Step 4: Convert current density to molar flux
            # -----------------------------------------------------------------
            # j = nF·J  →  J = j/(nF)
            # 
            # Flux direction convention:
            # - j > 0: Net reduction (O being consumed at surface)
            # - JO = -J: O flux is negative (O flowing toward electrode to be reduced)
            # - JR = +J: R flux is positive (R flowing away from electrode after being produced)
            
            J = j_new / (n_electrons * F)  # Molar flux [mol/(cm²·s)]
            
            JO_new = -J  # O is consumed when j > 0 (reduction)
            JR_new = +J  # R is produced when j > 0 (reduction)

            # -----------------------------------------------------------------
            # Step 5: Solve diffusion equations with new fluxes
            # -----------------------------------------------------------------
            # Build RHS vectors with flux boundary conditions
            rhsO = cn_rhs_with_flux_neumann_right(CO_old, lamO, dt, dx, JO_new, JO_prev)
            rhsR = cn_rhs_with_flux_neumann_right(CR_old, lamR, dt, dx, JR_new, JR_prev)

            # Solve tridiagonal systems: A·C_new = RHS
            try:
                # Note: We copy the LHS arrays because thomas_solve_inplace modifies them
                solO = thomas_solve_inplace(aLO.copy(), bLO.copy(), cLO.copy(), rhsO)
                solR = thomas_solve_inplace(aLR.copy(), bLR.copy(), cLR.copy(), rhsR)
            except FloatingPointError:
                # If solver fails, reduce relaxation and retry
                relax *= 0.5
                CO = CO_old.copy()
                CR = CR_old.copy()
                j_old *= 0.5
                if relax < 1e-6:
                    break  # Give up if relaxation is too small
                continue

            # Enforce non-negative concentrations (physically required)
            CO_new = np.clip(solO, 0.0, np.inf)
            CR_new = np.clip(solR, 0.0, np.inf)

            # -----------------------------------------------------------------
            # Step 6: Check convergence
            # -----------------------------------------------------------------
            dj = abs(j_new - j_old)                                    # Change in current
            dc0 = max(abs(CO_new[0] - cO0_old), abs(CR_new[0] - cR0_old))  # Change in surface conc

            # Update for next iteration
            CO, CR = CO_new, CR_new
            j_old = j_new
            cO0_old, cR0_old = CO_new[0], CR_new[0]

            # Converged if both current and concentration changes are small
            if dj < tol_j and dc0 < tol_c0:
                break
        
        # =====================================================================
        # END PICARD ITERATION: Store results
        # =====================================================================
        
        # Final values for this time step
        j = j_old
        J = j / (n_electrons * F)
        JO_prev = -J  # Save for next time step's Crank-Nicolson averaging
        JR_prev = +J

        # Convert current density to current
        # Sign convention: i = -j × A
        # - j > 0 means reduction (O + e⁻ → R), which is cathodic
        # - But Ossila convention: cathodic = negative current
        # - So we negate to match convention
        i[k] = -j * A

    # =========================================================================
    # POST-PROCESSING: Split into individual cycles
    # =========================================================================
    
    points_per_cycle = int(t_cycle / dt)
    cycle_data = []
    for cyc in range(n_cycles):
        start_idx = cyc * points_per_cycle
        end_idx = min((cyc + 1) * points_per_cycle, len(t))
        cycle_data.append({
            'E': E[start_idx:end_idx],
            'i': i[start_idx:end_idx],
            't': t[start_idx:end_idx] - t[start_idx],  # Reset time to 0 for each cycle
            'cycle_num': cyc + 1
        })

    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    
    return {
        'E': E,            # Full potential array [V]
        'i': i,            # Full current array [A]
        't': t,            # Full time array [s]
        'cycle_data': cycle_data,  # Per-cycle data
        'CO_final': CO,    # Final O concentration profile [mol/cm³]
        'CR_final': CR,    # Final R concentration profile [mol/cm³]
        'params': {
            'E_start': E_start,
            'E_switch': E_switch,
            'E0': E0,
            'v': v,
            'n_cycles': n_cycles,
            't_cycle': t_cycle,
            'A': A,
        }
    }


# =============================================================================
# PEAK ANALYSIS FUNCTION
# =============================================================================

def analyze_cycle_peaks(E: np.ndarray, i: np.ndarray, E_start: float, E_switch: float) -> dict:
    """
    Analyze CV peaks for a single cycle.
    
    WHAT THIS FINDS:
    ================
        - Anodic peak (ipa, Epa): Maximum positive current (oxidation)
        - Cathodic peak (ipc, Epc): Maximum negative current (reduction)
        - Peak separation ΔEp = |Epa - Epc|
        - Peak ratio |ipa/ipc|
    
    WHY THESE MATTER:
    =================
        For a reversible system:
        - ΔEp = 59/n mV at 25°C (diagnostic for reversibility)
        - |ipa/ipc| = 1.0 (indicates chemical reversibility)
        
        Deviations indicate:
        - Larger ΔEp: Slower kinetics (quasi-reversible or irreversible)
        - |ipa/ipc| ≠ 1: Chemical reactions, adsorption, or asymmetric diffusion
    
    Parameters:
        E: Potential array for one cycle [V]
        i: Current array for one cycle [A]
        E_start: Starting potential [V]
        E_switch: Switching potential [V]
    
    Returns:
        dict with: E_pc, i_pc, E_pa, i_pa, delta_Ep, ratio, i_min, i_max
    """
    # Find the switching point (where potential reverses)
    if E_switch > E_start:
        k_switch = int(np.argmax(E))  # Switch at maximum E
    else:
        k_switch = int(np.argmin(E))  # Switch at minimum E
    
    k_switch = max(1, min(k_switch, len(E) - 2))

    # Split into forward and reverse scans
    E_fwd = E[:k_switch + 1]
    i_fwd = i[:k_switch + 1]
    E_rev = E[k_switch:]
    i_rev = i[k_switch:]

    # Find peaks
    # Forward scan (oxidative): Find maximum current (anodic peak)
    idx_pa = int(np.argmax(i_fwd))
    # Reverse scan (reductive): Find minimum current (cathodic peak)
    idx_pc = int(np.argmin(i_rev))

    # Extract peak values
    E_pc, i_pc = float(E_rev[idx_pc]), float(i_rev[idx_pc])
    E_pa, i_pa = float(E_fwd[idx_pa]), float(i_fwd[idx_pa])

    return {
        "E_pc": E_pc,      # Cathodic peak potential [V]
        "i_pc": i_pc,      # Cathodic peak current [A]
        "E_pa": E_pa,      # Anodic peak potential [V]
        "i_pa": i_pa,      # Anodic peak current [A]
        "delta_Ep": abs(E_pa - E_pc),  # Peak separation [V]
        "ratio": abs(i_pa / i_pc) if i_pc != 0 else np.nan,  # Peak current ratio
        "i_min": float(np.min(i)),
        "i_max": float(np.max(i)),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_potential_waveform(t: np.ndarray, E: np.ndarray, params: dict, save_path: str = None):
    """
    Plot the potential waveform E(t) - the triangular wave applied to electrode.
    
    This shows the INPUT signal to the electrochemical cell:
    
        Potential (V)
             ^
        E_switch ──►  /\        /\
                     /  \      /  \
                    /    \    /    \
        E_start ──►/      \  /      \
                   |       \/        \
                   +──────────────────────► Time (s)
                   0     t_cycle   2*t_cycle
    
    Parameters:
        t: Time array [s]
        E: Potential array [V]
        params: Dictionary with E_start, E_switch, v, n_cycles, etc.
        save_path: If provided, save figure to this path
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot the waveform with green color (like the reference image)
    ax.plot(t, E, lw=2.5, color='#2E7D32')
    
    # Add horizontal dashed lines at E_start and E_switch for reference
    ax.axhline(y=params['E_start'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=params['E_switch'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add labels for the potential limits
    ax.text(t[-1] * 0.02, params['E_start'] - 0.02, f"E_start = {params['E_start']} V", 
            va='top', fontsize=10, color='gray')
    ax.text(t[-1] * 0.02, params['E_switch'] + 0.02, f"E_switch = {params['E_switch']} V", 
            va='bottom', fontsize=10, color='gray')
    
    # Labels and title
    ax.set_xlabel("Time (sec)", fontsize=12)
    ax.set_ylabel("Potential (V)", fontsize=12)
    ax.set_title("Potential Waveform", fontsize=14, fontweight='bold')
    
    # Add scan rate and cycle info as annotation
    info_text = f"Scan rate: {params['v']} V/s\nCycles: {params['n_cycles']}"
    ax.annotate(
        info_text,
        xy=(0.98, 0.98), xycoords='axes fraction',
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Potential waveform saved to: {save_path}")
    
    return fig, ax


def plot_cyclic_voltammogram(E: np.ndarray, i: np.ndarray, params: dict, save_path: str = None):
    """
    Plot the cyclic voltammogram (Current vs Potential).
    
    This shows the OUTPUT of the electrochemical measurement:
    
    Parameters:
        E: Potential array [V]
        i: Current array [A]
        params: Dictionary with E0, etc.
        save_path: If provided, save figure to this path
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert current to µA for readability
    i_uA = i * 1e6
    
    # Plot the CV curve
    ax.plot(E, i_uA, lw=2, color='#1a1a1a')
    
    # Add zero lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=params.get('E0', 0), color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Labels
    ax.set_xlabel("Potential (V)", fontsize=12)
    ax.set_ylabel("Current (µA)", fontsize=12)
    ax.set_title("Cyclic Voltammogram", fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"CV plot saved to: {save_path}")
    
    return fig, ax


def plot_cv_and_waveform(result: dict, save_path: str = None):
    """
    Plot both CV and potential waveform side by side
    
    Parameters:
        result: Dictionary returned by simulate_cv_multicycle()
        save_path: If provided, save figure to this path
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    params = result['params']
    
    # =========================================================================
    # LEFT PANEL: Cyclic Voltammogram (Current vs Potential)
    # =========================================================================
    ax1 = axes[0]
    
    # Use the last cycle (steady state)
    last_cycle = result['cycle_data'][-1]
    E_plot = last_cycle['E']
    i_plot = last_cycle['i'] * 1e6  # Convert A to µA
    
    # Plot CV curve
    ax1.plot(E_plot, i_plot, lw=2, color='#1a1a1a')
    
    # Analyze and mark peaks
    stats = analyze_cycle_peaks(
        last_cycle['E'], last_cycle['i'],
        params['E_start'], params['E_switch']
    )
    
    # Mark anodic peak (red dot)
    ax1.plot(stats['E_pa'], stats['i_pa'] * 1e6, 'o', ms=8, color='#c00000',
             label=f"Anodic: {stats['i_pa']*1e6:.1f} µA")
    # Mark cathodic peak (red dot)
    ax1.plot(stats['E_pc'], stats['i_pc'] * 1e6, 'o', ms=8, color='#c00000',
             label=f"Cathodic: {stats['i_pc']*1e6:.1f} µA")
    
    # Zero reference lines
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=params['E0'], color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Labels
    ax1.set_xlabel("Potential (V)", fontsize=12)
    ax1.set_ylabel("Current (µA)", fontsize=12)
    ax1.set_title("Cyclic Voltammogram", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # RIGHT PANEL: Potential Waveform (Potential vs Time)
    # =========================================================================
    ax2 = axes[1]
    
    # Plot the triangular waveform
    ax2.plot(result['t'], result['E'], lw=2.5, color='#2E7D32')
    
    # Reference lines at E_start and E_switch
    ax2.axhline(y=params['E_start'], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=params['E_switch'], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels
    ax2.set_xlabel("Time (sec)", fontsize=12)
    ax2.set_ylabel("Potential (V)", fontsize=12)
    ax2.set_title("Potential Waveform", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined figure saved to: {save_path}")
    
    return fig, axes


def plot_all_cycles(result: dict, save_path: str = None):
    """
    Plot all CV cycles overlaid on one graph.
    
    Useful for seeing how the system evolves toward steady state:
    - Cycle 1: Asymmetric (|ipa/ipc| > 1)
    - Cycle 2: More symmetric
    - Cycle 3+: Approaching steady state (|ipa/ipc| ≈ 1)
    
    Parameters:
        result: Dictionary returned by simulate_cv_multicycle()
        save_path: If provided, save figure to this path
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color gradient: lighter for early cycles, darker for later
    n_cycles = len(result['cycle_data'])
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_cycles))
    
    # Plot each cycle
    for idx, cycle in enumerate(result['cycle_data']):
        label = f"Cycle {cycle['cycle_num']}"
        ax.plot(cycle['E'], cycle['i'] * 1e6, lw=2, color=colors[idx], label=label)
    
    # Reference lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=result['params']['E0'], color='k', linestyle='--', 
               linewidth=0.5, alpha=0.5, label=f"E° = {result['params']['E0']} V")
    
    # Labels
    ax.set_xlabel("Potential (V)", fontsize=12)
    ax.set_ylabel("Current (µA)", fontsize=12)
    ax.set_title("Cyclic Voltammograms — All Cycles", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"All cycles plot saved to: {save_path}")
    
    return fig, ax


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run the CV simulation and generate all plots.
    
    WORKFLOW:
    =========
        1. Define simulation parameters
        2. Run simulation
        3. Analyze peaks for each cycle
        4. Generate plots:
           - CV and waveform side by side
           - All cycles overlaid
           - Just the potential waveform
    """
    
    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================
    
    params = dict(
        # ----- Electrochemical parameters -----
        E0=0.0,            # Formal potential [V] - equilibrium potential of redox couple
        E_start=-0.25,     # Start potential [V] - begin at negative for oxidative-first scan
        E_switch=0.25,     # Switching potential [V] - vertex of triangular wave
        v=0.2,             # Scan rate [V/s] - how fast we sweep potential
        n_electrons=1,     # Number of electrons in reaction: O + ne⁻ ⇌ R
        T=298.15,          # Temperature [K] = 25°C (room temperature)
        alpha=0.5,         # Transfer coefficient - symmetry of activation barrier
        k0=0.2,            # Standard rate constant [cm/s] - electron transfer speed
        
        # ----- Species properties -----
        DO=7e-6,           # Diffusion coefficient of O [cm²/s]
        DR=7e-6,           # Diffusion coefficient of R [cm²/s] (equal = symmetric diffusion)
        C_total=1e-6,      # Total concentration [mol/cm³] = 1 mM
        
        # ----- Electrode -----
        A=0.071,           # Electrode area [cm²]
        
        # ----- Numerical parameters -----
        L=0.10,            # Domain size [cm] - must be >> diffusion layer thickness
        Nx=601,            # Spatial grid points (more = more accurate, slower)
        dt=2e-4,           # Time step [s] (smaller = more accurate, slower)
        n_cycles=3,        # Number of CV cycles to simulate
        
        # ----- Picard iteration (nonlinear solver) -----
        relax0=0.05,       # Relaxation factor (0.05-0.1 is usually stable)
        picard_max=120,    # Maximum iterations per time step
        J_cap_factor=50.0, # Safety limit for flux
        exp_clamp=50.0,    # Safety limit for exponentials
    )

    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    
    print("="*60)
    print("CYCLIC VOLTAMMETRY SIMULATION")
    print("="*60)
    print(f"\nParameters:")
    print(f"  E_start  = {params['E_start']} V")
    print(f"  E_switch = {params['E_switch']} V")
    print(f"  E0       = {params['E0']} V")
    print(f"  Scan rate = {params['v']} V/s")
    print(f"  Cycles   = {params['n_cycles']}")
    print(f"  k0       = {params['k0']} cm/s")
    print(f"\nRunning simulation...")
    
    result = simulate_cv_multicycle(**params)
    
    print("Done!")
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "-"*60)
    print("PEAK ANALYSIS")
    print("-"*60)
    
    for cycle in result['cycle_data']:
        stats = analyze_cycle_peaks(
            cycle['E'], cycle['i'],
            params['E_start'], params['E_switch']
        )
        print(f"\nCycle {cycle['cycle_num']}:")
        print(f"  Anodic peak:   E_pa = {stats['E_pa']:+.4f} V,  i_pa = {stats['i_pa']*1e6:+.2f} µA")
        print(f"  Cathodic peak: E_pc = {stats['E_pc']:+.4f} V,  i_pc = {stats['i_pc']*1e6:+.2f} µA")
        print(f"  ΔE_p = {stats['delta_Ep']*1000:.1f} mV  (theory: 59 mV for reversible)")
        print(f"  |i_pa/i_pc| = {stats['ratio']:.4f}  (theory: 1.0 for reversible)")
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    
    print("\n" + "-"*60)
    print("GENERATING PLOTS")
    print("-"*60)
    
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: CV and waveform side by side (like the reference image)
    print("\n1. CV + Waveform (side by side)...")
    plot_cv_and_waveform(result, save_path="plots/cv_and_waveform.png")
    
    # Plot 2: All cycles overlaid
    print("2. All cycles overlaid...")
    plot_all_cycles(result, save_path="plots/cv_all_cycles.png")
    
    # Plot 3: Just the potential waveform
    print("3. Potential waveform only...")
    plot_potential_waveform(result['t'], result['E'], result['params'], 
                           save_path="plots/potential_waveform.png")
    
    print("\n" + "="*60)
    print("All plots saved to ./plots/")
    print("="*60)
    
    # Show plots
    plt.show()

# RUN MAIN

if __name__ == "__main__":
    main()