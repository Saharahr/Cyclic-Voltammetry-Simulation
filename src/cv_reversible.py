"""
    This code simulates Cyclic Voltammetry (CV) for a simple redox reaction:
    
        O + n e⁻  ⇌  R
        
    Where:
    - O = Oxidized species (electron acceptor)
    - R = Reduced species (electron donor)  
    - n = Number of electrons transferred (typically 1)
    
    Example: Fe³⁺ + e⁻ ⇌ Fe²⁺

WHAT IS CYCLIC VOLTAMMETRY?
===========================
    CV is an electrochemical technique where:
    1. The electrode potential is swept linearly from E_start to E_switch
    2. Then swept back from E_switch to E_start
    3. Current is measured as a function of potential
    
    The resulting current-potential curve reveals:
    - Peak positions → thermodynamics (formal potential E⁰)
    - Peak separation → kinetics (how fast electron transfer is)
    - Peak heights → concentration and diffusion coefficient

GOVERNING EQUATIONS:
====================
    1. Fick's Second Law (diffusion in solution):
       ∂C/∂t = D ∂²C/∂x²
       
       - C = concentration [mol/cm³]
       - D = diffusion coefficient [cm²/s]
       - x = distance from electrode [cm]
       - t = time [s]
       
    2. Butler-Volmer Equation (electrode kinetics):
       j = nFk₀[Cₒ(0)·exp(-αβη) - Cᵣ(0)·exp((1-α)βη)]
       
       - j = current density [A/cm²]
       - k₀ = standard rate constant [cm/s]
       - η = E - E⁰ = overpotential [V]
       - α = transfer coefficient (typically 0.5)
       - β = nF/RT ≈ 38.9 V⁻¹ at 25°C

NUMERICAL METHODS:
==================
    1. Crank-Nicolson: Solves diffusion equation (implicit, stable, 2nd order)
    2. Thomas Algorithm: Efficiently solves tridiagonal matrix systems O(n)
    3. Picard Iteration: Handles nonlinear coupling between current and concentration

SIGN CONVENTION:
===============================
    - Oxidation (R → O + e⁻): POSITIVE current (anodic)
    - Reduction (O + e⁻ → R): NEGATIVE current (cathodic)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================
# These are universal constants from CODATA - never modify these values!

F = 96485.33212       # Faraday constant [C/mol]
                      # = charge of 1 mole of electrons
                      # = N_A × e (Avogadro's number × electron charge)

R_gas = 8.314462618   # Universal gas constant [J/(mol·K)]
                      # Relates thermal energy to temperature
                      # Appears in Nernst equation and Butler-Volmer


# =============================================================================
# POTENTIAL WAVEFORM FUNCTION
# =============================================================================

def triangular_potential(t, E_start, E_switch, v):
    """
    Generate the triangular potential waveform for cyclic voltammetry.
    
    WHAT THIS CREATES:
    ==================
        The potential E(t) that is applied to the electrode over time.
        For CV, this is a triangular wave:

    THE PHYSICS:
    ============
        - Scan rate v [V/s] determines how fast potential changes
        - dE/dt = ±v (positive on forward scan, negative on reverse)
        - t_half = |E_switch - E_start| / v = time for one half-cycle
        - t_cycle = 2 × t_half = time for complete cycle
    
    WHY VECTORIZED:
    ===============
        Using np.where instead of a Python for-loop makes this
        ~100x faster for large arrays.
    
    Parameters:
    -----------
        t : np.ndarray
            Array of time points [seconds]
        E_start : float
            Starting potential [V] - where the sweep begins
        E_switch : float  
            Switching potential [V] - vertex of the triangle
        v : float
            Scan rate [V/s] - how fast potential changes
    
    Returns:
    --------
        E : np.ndarray
            Potential at each time point [V]
    
    Example:
    --------
        E_start = -0.25 V, E_switch = +0.25 V, v = 0.2 V/s
        |E_switch - E_start| = 0.5 V
        t_half = 0.5 / 0.2 = 2.5 seconds
        t_cycle = 5.0 seconds
    """
    # Calculate the potential range and direction
    dE = E_switch - E_start           # Total potential change [V]
    t_half = abs(dE) / v              # Time for half cycle [s]
    t_cycle = 2.0 * t_half            # Time for full cycle [s]
    
    # Direction sign: +1 if scanning positive, -1 if scanning negative
    s = 1.0 if dE >= 0 else -1.0
    
    # Use modulo to handle multiple cycles automatically
    # t % t_cycle gives the time position within the current cycle
    t_in_cycle = t % t_cycle
    
    # Boolean mask: True for forward scan, False for reverse scan
    forward_mask = t_in_cycle <= t_half
    
    # Vectorized potential calculation using np.where
    # np.where(condition, value_if_true, value_if_false)
    E = np.where(
        forward_mask,
        # FORWARD SCAN: E increases from E_start toward E_switch
        E_start + s * v * t_in_cycle,
        # REVERSE SCAN: E decreases from E_switch back to E_start  
        E_switch - s * v * (t_in_cycle - t_half)
    )
    
    return E


# =============================================================================
# THOMAS ALGORITHM (TRIDIAGONAL MATRIX SOLVER)
# =============================================================================

def thomas_solve(a, b, c, d):
    """
    Solve a tridiagonal system of linear equations using the Thomas algorithm.
    
    WHAT IS A TRIDIAGONAL SYSTEM?
    =============================
        A system of equations where the matrix has non-zero elements
        only on three diagonals (lower, main, upper):
        
        | b₀  c₀   0   0   0  |   | x₀ |   | d₀ |
        | a₀  b₁  c₁   0   0  |   | x₁ |   | d₁ |
        |  0  a₁  b₂  c₂   0  | × | x₂ | = | d₂ |
        |  0   0  a₂  b₃  c₃  |   | x₃ |   | d₃ |
        |  0   0   0  a₃  b₄  |   | x₄ |   | d₄ |
        
        This structure arises from discretizing the 1D diffusion equation.
    
    WHY THOMAS ALGORITHM?
    =====================
        - General matrix solve (Gaussian elimination): O(n³) operations
        - Thomas algorithm: O(n) operations - MUCH faster!
        - Perfect for 1D problems where only neighboring points interact
    
    HOW IT WORKS:
    =============
        PHASE 1 - Forward Elimination:
            Eliminate the lower diagonal by row operations.
            Transform the system into upper triangular form.
            
            For each row i (starting from 1):
                w = a[i-1] / b[i-1]        # Multiplier
                b[i] = b[i] - w × c[i-1]   # Update diagonal
                d[i] = d[i] - w × d[i-1]   # Update RHS
        
        PHASE 2 - Back Substitution:
            Solve from bottom to top.
            
            x[n-1] = d[n-1] / b[n-1]       # Last equation
            For i from n-2 down to 0:
                x[i] = (d[i] - c[i] × x[i+1]) / b[i]
    
    Parameters:
    -----------
        a : np.ndarray
            Lower diagonal coefficients (length n-1)
            a[i] is the coefficient of x[i] in equation i+1
        b : np.ndarray
            Main diagonal coefficients (length n)
            b[i] is the coefficient of x[i] in equation i
        c : np.ndarray
            Upper diagonal coefficients (length n-1)
            c[i] is the coefficient of x[i+1] in equation i
        d : np.ndarray
            Right-hand side vector (length n)
    
    Returns:
    --------
        x : np.ndarray
            Solution vector (length n)
    
    Note:
    -----
        This function makes copies of b and d to avoid modifying
        the original arrays (important for reusing matrices).
    """
    n = len(b)  # Number of equations/unknowns
    
    # Make copies to avoid modifying original arrays
    # (We reuse the same LHS matrices for O and R species)
    b_work = b.copy()
    d_work = d.copy()
    
    # =========================================================================
    # PHASE 1: FORWARD ELIMINATION
    # =========================================================================
    # Goal: Transform to upper triangular form by eliminating lower diagonal
    #
    # Original:                    After elimination:
    # | b₀  c₀   0  |             | b₀  c₀   0  |
    # | a₀  b₁  c₁  |  ────────►  |  0  b₁' c₁  |
    # |  0  a₁  b₂  |             |  0   0  b₂' |
    
    for i in range(1, n):
        # Compute the multiplier for row elimination
        # w = coefficient below diagonal / diagonal element
        w = a[i-1] / b_work[i-1]
        
        # Update the diagonal element
        # b[i] = b[i] - w × c[i-1]
        b_work[i] -= w * c[i-1]
        
        # Update the right-hand side
        # d[i] = d[i] - w × d[i-1]
        d_work[i] -= w * d_work[i-1]
    
    # =========================================================================
    # PHASE 2: BACK SUBSTITUTION
    # =========================================================================
    # Now we have upper triangular form:
    #
    # | b₀  c₀   0  |   | x₀ |   | d₀ |
    # |  0  b₁' c₁  | × | x₁ | = | d₁'|
    # |  0   0  b₂' |   | x₂ |   | d₂'|
    #
    # Solve from bottom to top:
    #   x₂ = d₂' / b₂'
    #   x₁ = (d₁' - c₁×x₂) / b₁'
    #   x₀ = (d₀ - c₀×x₁) / b₀
    
    # Solve the last equation first
    d_work[-1] /= b_work[-1]
    
    # Solve remaining equations from bottom to top
    for i in range(n-2, -1, -1):
        d_work[i] = (d_work[i] - c[i] * d_work[i+1]) / b_work[i]
    
    # The solution is now stored in d_work
    return d_work


# =============================================================================
# CRANK-NICOLSON MATRIX BUILDER
# =============================================================================

def build_cn_matrices(Nx, lam):
    """
    Build the LEFT-HAND SIDE (LHS) tridiagonal matrix for Crank-Nicolson scheme.
    
    WHAT IS CRANK-NICOLSON?
    =======================
        A numerical method to solve the diffusion equation:
        
            ∂C/∂t = D ∂²C/∂x²
        
        It averages between explicit (old time) and implicit (new time):
        
            (Cⁿ⁺¹ᵢ - Cⁿᵢ)/Δt = D/2 × [∂²Cⁿ⁺¹/∂x² + ∂²Cⁿ/∂x²]
        
        Advantages:
        - Unconditionally stable (no restriction on time step for stability)
        - Second-order accurate in both time and space
        - Implicit, so requires solving a linear system each time step
    
    DISCRETIZATION:
    ===============
        Using central differences for the second derivative:
        
        ∂²C/∂x² ≈ (Cᵢ₋₁ - 2Cᵢ + Cᵢ₊₁) / Δx²
        
        The Crank-Nicolson equation for interior points becomes:
        
        -λ/2·Cⁿ⁺¹ᵢ₋₁ + (1+λ)·Cⁿ⁺¹ᵢ - λ/2·Cⁿ⁺¹ᵢ₊₁ = 
         λ/2·Cⁿᵢ₋₁ + (1-λ)·Cⁿᵢ + λ/2·Cⁿᵢ₊₁
        
        Where λ = D·Δt/Δx² is the "diffusion number"
    
    BOUNDARY CONDITIONS:
    ====================
        Row 0 (electrode surface, x=0):
            Flux boundary condition is incorporated via the RHS function.
            LHS coefficients: (1+λ)·C₀ - λ·C₁ = RHS₀
            
        Row Nx-1 (bulk solution, x=L):
            Neumann BC: ∂C/∂x = 0 (zero flux at bulk boundary)
            Discretized: Cₙₓ₋₁ = Cₙₓ₋₂
            Rearranged: -Cₙₓ₋₂ + Cₙₓ₋₁ = 0
    
    MATRIX STRUCTURE:
    =================
        Row 0 (electrode):   | 1+λ   -λ    0    0  ... |
        Row 1 (interior):    | -λ/2  1+λ  -λ/2  0  ... |
        Row 2 (interior):    |  0   -λ/2  1+λ  -λ/2 ...|
        ...
        Row Nx-1 (bulk BC):  |  0    0   ...  -1    1  |
    
    Parameters:
    -----------
        Nx : int
            Number of spatial grid points
        lam : float
            Diffusion number λ = D·Δt/Δx² (dimensionless)
            Should be < 1 for good accuracy (though CN is stable for any λ)
    
    Returns:
    --------
        a : np.ndarray
            Lower diagonal coefficients (length Nx-1)
        b : np.ndarray
            Main diagonal coefficients (length Nx)
        c : np.ndarray
            Upper diagonal coefficients (length Nx-1)
    """
    # Initialize arrays for the three diagonals
    a = np.zeros(Nx - 1)   # Lower diagonal (below main)
    b = np.zeros(Nx)        # Main diagonal
    c = np.zeros(Nx - 1)   # Upper diagonal (above main)
    
    # -------------------------------------------------------------------------
    # ROW 0: Electrode surface boundary condition
    # -------------------------------------------------------------------------
    # At x=0, we have a flux boundary condition (species consumed/produced)
    # The flux term is added to the RHS in build_rhs()
    # LHS coefficients for equation: (1+λ)C₀ⁿ⁺¹ - λC₁ⁿ⁺¹ = RHS₀
    b[0] = 1.0 + lam    # Coefficient of C₀ (main diagonal)
    c[0] = -lam         # Coefficient of C₁ (upper diagonal)
    
    # -------------------------------------------------------------------------
    # ROWS 1 to Nx-2: Interior points (standard Crank-Nicolson)
    # -------------------------------------------------------------------------
    # Equation: -λ/2·Cᵢ₋₁ + (1+λ)·Cᵢ - λ/2·Cᵢ₊₁ = RHS
    # Using vectorized assignment for efficiency
    a[0:Nx-2] = -0.5 * lam    # Coefficient of Cᵢ₋₁
    b[1:Nx-1] = 1.0 + lam     # Coefficient of Cᵢ
    c[1:Nx-1] = -0.5 * lam    # Coefficient of Cᵢ₊₁
    
    # -------------------------------------------------------------------------
    # ROW Nx-1: Bulk boundary condition (Neumann: zero gradient)
    # -------------------------------------------------------------------------
    # Physical meaning: Far from electrode, concentration gradient is zero
    # Mathematical form: ∂C/∂x|ₓ₌ₗ = 0
    # Discretized: (Cₙₓ₋₁ - Cₙₓ₋₂)/Δx = 0  →  Cₙₓ₋₁ = Cₙₓ₋₂
    # Rearranged: -Cₙₓ₋₂ + Cₙₓ₋₁ = 0
    a[Nx-2] = -1.0    # Coefficient of Cₙₓ₋₂
    b[Nx-1] = 1.0     # Coefficient of Cₙₓ₋₁
    
    return a, b, c


# =============================================================================
# RIGHT-HAND SIDE (RHS) BUILDER WITH FLUX BOUNDARY CONDITION
# =============================================================================

def build_rhs(C_old, lam, dt, dx, J_new, J_prev):
    """
    Build the RIGHT-HAND SIDE (RHS) vector for the Crank-Nicolson system.
    
    THE LINEAR SYSTEM:
    ==================
        We solve: A · Cⁿ⁺¹ = RHS
        
        Where:
        - A is the LHS matrix (built by build_cn_matrices)
        - Cⁿ⁺¹ is the unknown concentration at new time step
        - RHS contains known values from old time step + boundary conditions
    
    FLUX BOUNDARY CONDITION AT ELECTRODE:
    =====================================
        At x=0, species are consumed or produced by the electrochemical reaction.
        
        Fick's First Law relates flux to concentration gradient:
            J = -D · ∂C/∂x|ₓ₌₀
        
        Where J is the molar flux [mol/(cm²·s)]:
        - J > 0: Species flowing INTO solution (being produced)
        - J < 0: Species flowing OUT of solution (being consumed)
        
        Using a "ghost node" approach to handle this flux BC:
        
            C₋₁ = C₀ + J·Δx/D
            
        Substituting into the Crank-Nicolson equation and simplifying,
        the flux contribution appears as an additional term in RHS₀:
        
            RHS₀ = (1-λ)·C₀ⁿ + λ·C₁ⁿ + (Δt/Δx)·(Jⁿ⁺¹ + Jⁿ)
            
        The (Δt/Δx)·(J_new + J_prev) term is the Crank-Nicolson average
        of the flux at old and new time steps.
    
    INTERIOR POINTS:
    ================
        For i = 1, 2, ..., Nx-2:
        
            RHSᵢ = λ/2·Cⁿᵢ₋₁ + (1-λ)·Cⁿᵢ + λ/2·Cⁿᵢ₊₁
        
        This is the explicit part of Crank-Nicolson (known from old time step).
    
    BULK BOUNDARY:
    ==============
        For i = Nx-1 (Neumann BC: Cₙₓ₋₁ = Cₙₓ₋₂):
        
            RHSₙₓ₋₁ = 0
        
        Combined with LHS row [-1, 1], this gives Cₙₓ₋₁ = Cₙₓ₋₂.
    
    Parameters:
    -----------
        C_old : np.ndarray
            Concentration array at previous time step [mol/cm³]
        lam : float
            Diffusion number λ = D·Δt/Δx²
        dt : float
            Time step [s]
        dx : float
            Spatial grid spacing [cm]
        J_new : float
            Molar flux at new time step [mol/(cm²·s)]
        J_prev : float
            Molar flux at previous time step [mol/(cm²·s)]
    
    Returns:
    --------
        rhs : np.ndarray
            Right-hand side vector (length Nx)
    """
    Nx = len(C_old)
    rhs = np.empty(Nx)  # Pre-allocate (faster than np.zeros for large arrays)
    
    # -------------------------------------------------------------------------
    # ROW 0: Electrode surface with flux BC
    # -------------------------------------------------------------------------
    # RHS₀ = (diffusion terms from old time step) + (flux contribution)
    #
    # Flux term explanation:
    #   (dt/dx) converts flux [mol/(cm²·s)] to concentration change [mol/cm³]
    #   (J_new + J_prev) is Crank-Nicolson averaging over old and new time
    #
    # Physical meaning:
    #   - If J > 0: Species is being added → concentration increases
    #   - If J < 0: Species is being removed → concentration decreases
    rhs[0] = (1.0 - lam) * C_old[0] + lam * C_old[1] + (dt / dx) * (J_new + J_prev)
    
    # -------------------------------------------------------------------------
    # ROWS 1 to Nx-2: Interior points (VECTORIZED for speed!)
    # -------------------------------------------------------------------------
    # Standard Crank-Nicolson RHS: weighted average of neighbors
    # RHSᵢ = λ/2·Cᵢ₋₁ + (1-λ)·Cᵢ + λ/2·Cᵢ₊₁
    #
    # This is ~10x faster than a Python for-loop!
    rhs[1:Nx-1] = ((1.0 - lam) * C_old[1:Nx-1] + 
                   0.5 * lam * (C_old[0:Nx-2] + C_old[2:Nx]))
    
    # -------------------------------------------------------------------------
    # ROW Nx-1: Bulk boundary (Neumann BC)
    # -------------------------------------------------------------------------
    # Equation is: -Cₙₓ₋₂ + Cₙₓ₋₁ = 0, so RHS = 0
    rhs[Nx-1] = 0.0
    
    return rhs


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def simulate_cv_stable(
    *,  # Force keyword arguments (prevents positional argument errors)
    # -------------------------------------------------------------------------
    # ELECTROCHEMICAL PARAMETERS
    # -------------------------------------------------------------------------
    E0=0.0,           # Formal potential [V]
                      # The equilibrium potential of the redox couple
                      # At E = E0, forward and reverse rates are equal (for equal conc.)
    
    E_start=-0.25,    # Starting potential [V]
                      # Where the potential sweep begins
                      # Typically 100-300 mV negative of E0 for oxidative-first scan
    
    E_switch=0.25,    # Switching potential [V] (vertex)
                      # Where the sweep direction reverses
                      # Typically 100-300 mV positive of E0
    
    v=0.2,            # Scan rate [V/s]
                      # How fast the potential changes
                      # Typical range: 0.01 - 10 V/s
                      # Faster scan → larger peaks but more kinetic limitations
    
    n_electrons=1,    # Number of electrons transferred
                      # For O + n·e⁻ ⇌ R
                      # Affects peak separation: ΔEp = 59/n mV for reversible
    
    T=298.15,         # Temperature [K]
                      # 298.15 K = 25°C (standard conditions)
                      # Affects β = nF/RT and diffusion
    
    alpha=0.5,        # Transfer coefficient (symmetry factor)
                      # Fraction of applied potential that accelerates reduction
                      # α = 0.5 means symmetric energy barrier
                      # Typical range: 0.3 - 0.7
    
    k0=0.2,           # Standard rate constant [cm/s]
                      # How fast electron transfer occurs at E = E0
                      # k0 > 0.1: Fast/reversible kinetics
                      # k0 = 0.001-0.1: Quasi-reversible
                      # k0 < 0.001: Slow/irreversible
    
    # -------------------------------------------------------------------------
    # SPECIES PROPERTIES
    # -------------------------------------------------------------------------
    DO=7e-6,          # Diffusion coefficient of O [cm²/s]
                      # How fast oxidized species moves through solution
                      # Typical values: 10⁻⁶ to 10⁻⁵ cm²/s
    
    DR=7e-6,          # Diffusion coefficient of R [cm²/s]
                      # How fast reduced species moves through solution
                      # Often assumed equal to DO (symmetric diffusion)
    
    C_total=1e-6,     # Total concentration [mol/cm³]
                      # = 1 mM (millimolar)
                      # Initial condition: all R, no O
    
    # -------------------------------------------------------------------------
    # ELECTRODE PROPERTIES
    # -------------------------------------------------------------------------
    A=0.071,          # Electrode area [cm²]
                      # Current i = j × A (current density × area)
    
    L=0.10,           # Domain length [cm]
                      # Distance from electrode to bulk boundary
                      # Must be >> diffusion layer thickness δ ≈ √(D·t)
    
    # -------------------------------------------------------------------------
    # NUMERICAL PARAMETERS
    # -------------------------------------------------------------------------
    Nx=301,           # Number of spatial grid points
                      # More points → better accuracy but slower
                      # Rule: Need ~10-20 points in diffusion layer
                      # δ/dx ≈ √(D·t_total) / (L/Nx) should be > 10
    
    dt=2e-4,          # Time step [s]
                      # Smaller → more accurate but slower
                      # Check: λ = D·dt/dx² should be < 1 for accuracy
    
    n_cycles=3,       # Number of CV cycles
                      # Cycle 1: |ipa/ipc| > 1 (asymmetric)
                      # Cycles 2-3: Approaches |ipa/ipc| ≈ 1 (steady state)
    
    picard_max=50,    # Maximum Picard iterations per time step
                      # Usually converges in 10-30 iterations
                      # Set higher for safety
    
    relax=0.08,       # Relaxation factor for Picard iteration
                      # j_new = relax × j_BV + (1-relax) × j_old
                      # 
                      # KEY STABILITY PARAMETER!
                      # - relax = 1.0: No relaxation, may oscillate
                      # - relax = 0.1: Conservative, stable
                      # - relax = 0.05-0.10: Recommended range
    
    tol=1e-10,        # Convergence tolerance for Picard iteration
                      # Stop when |j_new - j_old| < tol
):
    """
    Simulate cyclic voltammetry for a two-species redox system.
    
    WHAT THIS FUNCTION DOES:
    ========================
        1. Set up spatial grid and initial conditions
        2. Build the LHS matrices for Crank-Nicolson (done once)
        3. Time-step through the potential sweep:
           a. Calculate E(t) from triangular waveform
           b. Picard iteration to solve coupled problem:
              - Butler-Volmer → current from surface concentrations
              - Diffusion → new concentrations from flux
           c. Store current for output
        4. Return results organized by cycle
    
    THE BUTLER-VOLMER EQUATION:
    ===========================
        j = nFk₀[Cₒ(0)·exp(-αβη) - Cᵣ(0)·exp((1-α)βη)]
        
        Terms:
        - First term: Reduction rate (O + e⁻ → R)
          * exp(-αβη) increases when η < 0 (negative potential)
          * Proportional to Cₒ(0) (need O at surface to reduce)
          
        - Second term: Oxidation rate (R → O + e⁻)
          * exp((1-α)βη) increases when η > 0 (positive potential)
          * Proportional to Cᵣ(0) (need R at surface to oxidize)
        
        Net current j is the difference between these rates.
    
    PICARD ITERATION (Why needed?):
    ===============================
        The problem is NONLINEAR and COUPLED:
        
        j depends on ──► C(0) depends on ──► J depends on ──► j
        (Butler-Volmer)   (diffusion)        (j = nFJ)
        
        This circular dependency is resolved iteratively:
        1. Guess current j
        2. Calculate flux J = j/(nF)
        3. Solve diffusion equation → new C(0)
        4. Calculate new j from Butler-Volmer with new C(0)
        5. Relax: j = relax×j_new + (1-relax)×j_old
        6. Repeat until |j_new - j_old| < tolerance
    
    Returns:
    --------
        dict containing:
            'E': Full potential array [V]
            'i': Full current array [A]
            't': Full time array [s]
            'cycle_data': List of dicts for each cycle
            'CO_final': Final O concentration profile [mol/cm³]
            'CR_final': Final R concentration profile [mol/cm³]
            'params': Dictionary of parameters used
    """
    
    # =========================================================================
    # DERIVED PARAMETERS
    # =========================================================================
    
    # β = nF/RT: Exponential sensitivity to potential
    # At 25°C with n=1: β ≈ 38.9 V⁻¹
    # Meaning: 25.7 mV change in potential → e-fold change in rate
    beta = n_electrons * F / (R_gas * T)
    
    # Spatial grid spacing
    dx = L / (Nx - 1)  # [cm]
    
    # Diffusion numbers (λ = D·Δt/Δx²)
    # These determine the "spread" of diffusion per time step
    # λ < 1 is recommended for good accuracy
    lamO = DO * dt / dx**2  # For oxidized species
    lamR = DR * dt / dx**2  # For reduced species
    
    # Print stability check
    print(f"Stability check: λ_O = {lamO:.4f}, λ_R = {lamR:.4f} (should be < 1)")
    
    # =========================================================================
    # TIME AND POTENTIAL ARRAYS
    # =========================================================================
    
    # Calculate cycle timing
    t_half = abs(E_switch - E_start) / v  # Time for half cycle [s]
    t_cycle = 2.0 * t_half                 # Time for complete cycle [s]
    t_total = n_cycles * t_cycle           # Total simulation time [s]
    
    # Create time array (from 0 to t_total with step dt)
    t = np.arange(0.0, t_total + dt, dt)
    
    # Generate triangular potential waveform
    E = triangular_potential(t, E_start, E_switch, v)
    
    n_steps = len(t)
    print(f"Time steps: {n_steps:,}")
    print(f"Grid points: {Nx}")
    
    # =========================================================================
    # INITIAL CONDITIONS
    # =========================================================================
    
    # Start with only reduced species (R) present
    # This is the typical experimental condition:
    # - Add your reduced analyte to solution
    # - Scan positive to oxidize it first
    CO = np.zeros(Nx)                  # [O] = 0 everywhere initially
    CR = np.full(Nx, C_total)          # [R] = C_total everywhere initially
    
    # =========================================================================
    # BUILD LHS MATRICES (done once, reused every time step)
    # =========================================================================
    
    aO, bO, cO = build_cn_matrices(Nx, lamO)  # For O diffusion equation
    aR, bR, cR = build_cn_matrices(Nx, lamR)  # For R diffusion equation
    
    # =========================================================================
    # INITIALIZE OUTPUT AND STATE VARIABLES
    # =========================================================================
    
    i_out = np.zeros(n_steps)  # Current array [A]
    
    j = 0.0          # Current density [A/cm²]
    JO_prev = 0.0    # O flux at previous time step [mol/(cm²·s)]
    JR_prev = 0.0    # R flux at previous time step [mol/(cm²·s)]
    
    # Flux cap for stability (prevent unrealistic values)
    J_cap = 50.0 * DR * C_total / dx
    j_cap = n_electrons * F * J_cap
    
    # Progress tracking
    print_interval = n_steps // 10  # Print every 10%
    t_start = perf_counter()
    
    # =========================================================================
    # MAIN TIME LOOP
    # =========================================================================
    
    for k in range(n_steps):
        
        # ---------------------------------------------------------------------
        # Progress indicator (every 10%)
        # ---------------------------------------------------------------------
        if k > 0 and k % print_interval == 0:
            elapsed = perf_counter() - t_start
            pct = 100 * k / n_steps
            remaining = elapsed / k * (n_steps - k)
            print(f"  {pct:.0f}% complete, ~{remaining:.0f}s remaining...")
        
        # ---------------------------------------------------------------------
        # Calculate overpotential
        # ---------------------------------------------------------------------
        # η = E - E⁰ is the driving force for the reaction
        # η > 0: Favors oxidation (R → O + e⁻)
        # η < 0: Favors reduction (O + e⁻ → R)
        eta = E[k] - E0
        
        # ---------------------------------------------------------------------
        # Save old state for Picard iteration
        # ---------------------------------------------------------------------
        CO_old = CO.copy()
        CR_old = CR.copy()
        j_old = j
        
        # ---------------------------------------------------------------------
        # Pre-compute exponential factors (constant for this time step)
        # ---------------------------------------------------------------------
        # These appear in Butler-Volmer equation:
        #   j = nFk₀[Cₒ·ef - Cᵣ·er]
        #
        # ef = exp(-αβη): Reduction rate factor
        #   - η > 0 → ef small (reduction suppressed at positive potential)
        #   - η < 0 → ef large (reduction enhanced at negative potential)
        #
        # er = exp((1-α)βη): Oxidation rate factor
        #   - η > 0 → er large (oxidation enhanced at positive potential)
        #   - η < 0 → er small (oxidation suppressed at negative potential)
        #
        # Clipping prevents overflow: exp(50) ≈ 5×10²¹ (still finite)
        ef = np.exp(np.clip(-alpha * beta * eta, -50, 50))
        er = np.exp(np.clip((1.0 - alpha) * beta * eta, -50, 50))
        
        # =====================================================================
        # PICARD ITERATION
        # =====================================================================
        # Iteratively solve the coupled nonlinear problem
        
        for it in range(picard_max):
            
            # -----------------------------------------------------------------
            # Step 1: Get surface concentrations (ensure non-negative)
            # -----------------------------------------------------------------
            CO0 = max(CO[0], 0.0)  # [O] at electrode surface
            CR0 = max(CR[0], 0.0)  # [R] at electrode surface
            
            # -----------------------------------------------------------------
            # Step 2: Calculate Butler-Volmer current density
            # -----------------------------------------------------------------
            # j = nFk₀[Cₒ(0)·exp(-αβη) - Cᵣ(0)·exp((1-α)βη)]
            #       └── reduction term ──┘   └── oxidation term ──┘
            #
            # Sign convention:
            # - j > 0: Net reduction (O being consumed)
            # - j < 0: Net oxidation (R being consumed)
            j_bv = n_electrons * F * k0 * (CO0 * ef - CR0 * er)
            
            # -----------------------------------------------------------------
            # Step 3: Relaxation (KEY FOR STABILITY!)
            # -----------------------------------------------------------------
            # Instead of jumping directly to j_bv, we blend with old value:
            #   j_new = relax × j_bv + (1 - relax) × j_old
            #
            # This prevents oscillation when the system is stiff
            # (i.e., when small changes in C cause large changes in j)
            j_new = relax * j_bv + (1.0 - relax) * j_old
            
            # Clamp to prevent runaway (safety measure)
            j_new = np.clip(j_new, -j_cap, j_cap)
            
            # -----------------------------------------------------------------
            # Step 4: Convert current density to molar fluxes
            # -----------------------------------------------------------------
            # j = nF × J, where J is molar flux [mol/(cm²·s)]
            J = j_new / (n_electrons * F)
            
            # Flux direction convention (for O + ne⁻ → R):
            # - When j > 0 (reduction): O is consumed, R is produced
            #   * JO < 0 (O flows toward electrode to be consumed)
            #   * JR > 0 (R flows away from electrode after production)
            # - When j < 0 (oxidation): R is consumed, O is produced
            #   * JO > 0 (O flows away after production)
            #   * JR < 0 (R flows toward electrode to be consumed)
            JO_new = -J  # O flux (negative of net reaction flux)
            JR_new = +J  # R flux (positive of net reaction flux)
            
            # -----------------------------------------------------------------
            # Step 5: Build RHS and solve diffusion equations
            # -----------------------------------------------------------------
            # Build RHS vectors with flux boundary conditions
            rhsO = build_rhs(CO_old, lamO, dt, dx, JO_new, JO_prev)
            rhsR = build_rhs(CR_old, lamR, dt, dx, JR_new, JR_prev)
            
            # Solve tridiagonal systems: A × C_new = RHS
            CO_new = thomas_solve(aO, bO, cO, rhsO)
            CR_new = thomas_solve(aR, bR, cR, rhsR)
            
            # Enforce non-negative concentrations (physical requirement)
            np.maximum(CO_new, 0.0, out=CO_new)
            np.maximum(CR_new, 0.0, out=CR_new)
            
            # -----------------------------------------------------------------
            # Step 6: Check convergence
            # -----------------------------------------------------------------
            dj = abs(j_new - j_old)  # Change in current density
            
            # Update for next iteration
            CO = CO_new
            CR = CR_new
            j_old = j_new
            
            # Converged if current change is below tolerance
            if dj < tol:
                break
        
        # =====================================================================
        # END PICARD ITERATION: Store results
        # =====================================================================
        
        # Save final values
        j = j_new
        
        # Update flux history for next time step's Crank-Nicolson averaging
        JO_prev = -j / (n_electrons * F)
        JR_prev = +j / (n_electrons * F)
        
        # Calculate current from current density
        # Sign convention: i = -j × A
        # - This converts from our internal convention (j > 0 for reduction)
        # - To Ossila convention (reduction = negative current)
        i_out[k] = -j * A
    
    # =========================================================================
    # TIMING REPORT
    # =========================================================================
    elapsed = perf_counter() - t_start
    print(f"Completed in {elapsed:.1f} seconds")
    
    # =========================================================================
    # SPLIT RESULTS INTO INDIVIDUAL CYCLES
    # =========================================================================
    
    points_per_cycle = int(t_cycle / dt)
    cycle_data = []
    
    for cyc in range(n_cycles):
        start_idx = cyc * points_per_cycle
        end_idx = min((cyc + 1) * points_per_cycle, len(t))
        
        cycle_data.append({
            'E': E[start_idx:end_idx],           # Potential for this cycle
            'i': i_out[start_idx:end_idx],       # Current for this cycle
            't': t[start_idx:end_idx] - t[start_idx],  # Time (reset to 0)
            'cycle_num': cyc + 1                 # Cycle number (1-indexed)
        })
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    
    return {
        'E': E,                  # Full potential array [V]
        'i': i_out,              # Full current array [A]
        't': t,                  # Full time array [s]
        'cycle_data': cycle_data,  # Per-cycle data
        'CO_final': CO,          # Final O concentration profile [mol/cm³]
        'CR_final': CR,          # Final R concentration profile [mol/cm³]
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

def analyze_peaks(E, i, E_start, E_switch):
    """
    Analyze the peaks in a cyclic voltammogram.
    
    WHAT THIS FINDS:
    ================
        - Anodic peak (ipa, Epa): Maximum positive current (oxidation)
        - Cathodic peak (ipc, Epc): Maximum negative current (reduction)
        - Peak separation: ΔEp = |Epa - Epc|
        - Peak ratio: |ipa/ipc|
    
    DIAGNOSTIC CRITERIA:
    ====================
        For a REVERSIBLE system (fast kinetics):
        - ΔEp = 59/n mV at 25°C (where n = electrons transferred)
        - |ipa/ipc| = 1.0 (at steady state)
        
        For QUASI-REVERSIBLE system:
        - 59/n mV < ΔEp < 200 mV
        - |ipa/ipc| ≈ 1.0
        
        For IRREVERSIBLE system:
        - ΔEp > 200 mV (or only one peak visible)
        
        |ipa/ipc| ≠ 1 indicates:
        - First cycle: Normal (asymmetric starting conditions)
        - Steady state: Chemical reactions, adsorption, unequal diffusion
    
    Parameters:
    -----------
        E : np.ndarray
            Potential array for one cycle [V]
        i : np.ndarray
            Current array for one cycle [A]
        E_start : float
            Starting potential [V]
        E_switch : float
            Switching potential [V]
    
    Returns:
    --------
        dict with:
            'E_pa': Anodic peak potential [V]
            'i_pa': Anodic peak current [A]
            'E_pc': Cathodic peak potential [V]
            'i_pc': Cathodic peak current [A]
            'delta_Ep': Peak separation [V]
            'ratio': |ipa/ipc| (dimensionless)
    """
    # Find the switching point (where potential reaches its extreme)
    if E_switch > E_start:
        k_switch = np.argmax(E)  # Switching at maximum E
    else:
        k_switch = np.argmin(E)  # Switching at minimum E
    
    # Ensure k_switch is within valid range
    k_switch = max(1, min(k_switch, len(E) - 2))
    
    # Anodic peak: Maximum current in forward scan
    # (Forward scan is from start to switch)
    idx_pa = np.argmax(i[:k_switch+1])
    
    # Cathodic peak: Minimum current in reverse scan
    # (Reverse scan is from switch to end)
    idx_pc = np.argmin(i[k_switch:]) + k_switch
    
    # Extract peak values
    E_pa, i_pa = E[idx_pa], i[idx_pa]
    E_pc, i_pc = E[idx_pc], i[idx_pc]
    
    return {
        'E_pa': E_pa,                    # Anodic peak potential [V]
        'i_pa': i_pa,                    # Anodic peak current [A]
        'E_pc': E_pc,                    # Cathodic peak potential [V]
        'i_pc': i_pc,                    # Cathodic peak current [A]
        'delta_Ep': abs(E_pa - E_pc),    # Peak separation [V]
        'ratio': abs(i_pa / i_pc) if i_pc != 0 else np.nan  # Peak ratio
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_results(result, save_path=None):
    """
    Plot CV and potential waveform side by side.
    
    Parameters:
    -----------
        result : dict
            Output from simulate_cv_stable()
        save_path : str, optional
            If provided, save figure to this path
    
    Returns:
    --------
        fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    params = result['params']
    
    # =========================================================================
    # LEFT PANEL: Cyclic Voltammogram (Current vs Potential)
    # =========================================================================
    ax1 = axes[0]
    
    # Use the last cycle (closest to steady state)
    last = result['cycle_data'][-1]
    
    # Plot CV curve (current in µA for readability)
    ax1.plot(last['E'], last['i'] * 1e6, 'k-', lw=1.5)
    
    # Analyze and mark peaks
    stats = analyze_peaks(last['E'], last['i'], params['E_start'], params['E_switch'])
    
    # Mark anodic peak (red circle)
    ax1.plot(stats['E_pa'], stats['i_pa'] * 1e6, 'ro', ms=8, 
             label=f"Anodic: {stats['i_pa']*1e6:.1f} µA")
    
    # Mark cathodic peak (blue circle)
    ax1.plot(stats['E_pc'], stats['i_pc'] * 1e6, 'bo', ms=8, 
             label=f"Cathodic: {stats['i_pc']*1e6:.1f} µA")
    
    # Reference lines
    ax1.axhline(0, color='gray', lw=0.5)               # Zero current line
    ax1.axvline(params['E0'], color='gray', lw=0.5, ls='--')  # E0 reference
    
    # Labels and formatting
    ax1.set_xlabel('Potential (V)', fontsize=12)
    ax1.set_ylabel('Current (µA)', fontsize=12)
    ax1.set_title('Cyclic Voltammogram', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add peak analysis annotation
    ax1.annotate(
        f'ΔEp = {stats["delta_Ep"]*1000:.1f} mV\n|ipa/ipc| = {stats["ratio"]:.3f}',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # =========================================================================
    # RIGHT PANEL: Potential Waveform (Potential vs Time)
    # =========================================================================
    ax2 = axes[1]
    
    # Plot triangular waveform
    ax2.plot(result['t'], result['E'], 'g-', lw=2)
    
    # Reference lines at E_start and E_switch
    ax2.axhline(params['E_start'], color='gray', ls='--', lw=1, alpha=0.5)
    ax2.axhline(params['E_switch'], color='gray', ls='--', lw=1, alpha=0.5)
    
    # Labels and formatting
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Potential (V)', fontsize=12)
    ax2.set_title('Potential Waveform', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    return fig, axes


def plot_all_cycles(result, save_path=None):
    """
    Plot all CV cycles overlaid on one graph.
    
    Useful for visualizing the approach to steady state:
    - Cycle 1: Asymmetric (|ipa/ipc| > 1)
    - Cycle 2: More symmetric
    - Cycle 3+: Near steady state (|ipa/ipc| ≈ 1)
    
    Parameters:
    -----------
        result : dict
            Output from simulate_cv_stable()
        save_path : str, optional
            If provided, save figure to this path
    
    Returns:
    --------
        fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color gradient: lighter for early cycles, darker for later
    n_cycles = len(result['cycle_data'])
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_cycles))
    
    # Plot each cycle
    for idx, cycle in enumerate(result['cycle_data']):
        ax.plot(cycle['E'], cycle['i'] * 1e6, lw=1.5, color=colors[idx], 
                label=f"Cycle {cycle['cycle_num']}")
    
    # Reference line at zero current
    ax.axhline(0, color='gray', lw=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Potential (V)', fontsize=12)
    ax.set_ylabel('Current (µA)', fontsize=12)
    ax.set_title('All CV Cycles', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig, ax


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run CV simulation and generate output.
    
    WORKFLOW:
    =========
        1. Define simulation parameters
        2. Run simulation
        3. Analyze peaks for each cycle
        4. Compare to theoretical predictions
        5. Generate and save plots
    """
    
    print("="*60)
    print("STABLE CV SIMULATION")
    print("="*60)
    
    # =========================================================================
    # DEFINE PARAMETERS
    # =========================================================================
    
    params = dict(
        # Electrochemical parameters
        E0=0.0,            # Formal potential [V]
        E_start=-0.25,     # Start potential [V]
        E_switch=0.25,     # Switching potential [V]
        v=0.2,             # Scan rate [V/s]
        n_electrons=1,     # Electrons transferred
        T=298.15,          # Temperature [K]
        alpha=0.5,         # Transfer coefficient
        k0=0.2,            # Rate constant [cm/s]
        
        # Species properties
        DO=7e-6,           # Diffusion coefficient of O [cm²/s]
        DR=7e-6,           # Diffusion coefficient of R [cm²/s]
        C_total=1e-6,      # Concentration [mol/cm³]
        
        # Electrode
        A=0.071,           # Electrode area [cm²]
        L=0.10,            # Domain length [cm]
        
        # Numerical parameters (STABLE values)
        Nx=301,            # Grid points
        dt=2e-4,           # Time step [s]
        n_cycles=3,        # Number of cycles
        picard_max=50,     # Max Picard iterations
        relax=0.08,        # Relaxation factor (KEY FOR STABILITY!)
        tol=1e-10,         # Convergence tolerance
    )
    
    print(f"\nParameters:")
    print(f"  Nx = {params['Nx']}, dt = {params['dt']}")
    print(f"  relax = {params['relax']} (conservative for stability)")
    print(f"  n_cycles = {params['n_cycles']}")
    
    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    
    print("\nRunning simulation...")
    result = simulate_cv_stable(**params)
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    
    for cycle in result['cycle_data']:
        stats = analyze_peaks(cycle['E'], cycle['i'],
                             params['E_start'], params['E_switch'])
        print(f"\nCycle {cycle['cycle_num']}:")
        print(f"  Anodic peak:   E_pa = {stats['E_pa']:+.4f} V,  i_pa = {stats['i_pa']*1e6:+.2f} µA")
        print(f"  Cathodic peak: E_pc = {stats['E_pc']:+.4f} V,  i_pc = {stats['i_pc']*1e6:+.2f} µA")
        print(f"  ΔEp = {stats['delta_Ep']*1000:.1f} mV")
        print(f"  |ipa/ipc| = {stats['ratio']:.4f}")
    
    # =========================================================================
    # THEORETICAL COMPARISON
    # =========================================================================
    
    print("\n" + "-"*60)
    print("THEORETICAL COMPARISON:")
    print("-"*60)
    print(f"  Expected ΔEp (reversible): 59.0 mV")
    print(f"  Expected |ipa/ipc| (steady state): 1.0")
    
    # Randles-Sevcik equation for peak current
    # i_p = 0.4463 × nFAC × √(nFvD/RT)
    n, D, C, v, A = 1, 7e-6, 1e-6, 0.2, 0.071
    i_theory = 0.4463 * n * F * A * C * np.sqrt(n * F * v * D / (R_gas * 298.15))
    print(f"\n  Randles-Sevcik theoretical i_p: {i_theory*1e6:.2f} µA")
    
    # Compare to simulation
    last_stats = analyze_peaks(result['cycle_data'][-1]['E'], 
                               result['cycle_data'][-1]['i'],
                               params['E_start'], params['E_switch'])
    print(f"  Simulated i_pa (last cycle):   {last_stats['i_pa']*1e6:.2f} µA")
    print(f"  Agreement: {100*last_stats['i_pa']/i_theory:.1f}%")
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: CV and waveform side by side
    plot_results(result, "plots/cv_stable.png")
    
    # Plot 2: All cycles overlaid
    plot_all_cycles(result, "plots/cv_all_cycles.png")
    
    print("\n" + "="*60)
    print("Plots saved to ./plots/")
    print("="*60)
    
    plt.show()
    
    return result


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()