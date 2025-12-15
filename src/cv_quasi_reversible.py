import numpy as np
import matplotlib.pyplot as plt
import os

F = 96485.33212
R = 8.314462618

def safe_exp(x: float, clamp: float = 60.0) -> float:
    return float(np.exp(np.clip(x, -clamp, clamp)))

def triangular_potential(t: np.ndarray, E_start: float, E_switch: float, v: float) -> np.ndarray:
    dE = E_switch - E_start
    t1 = abs(dE) / v
    s = 1.0 if dE >= 0 else -1.0
    E = np.empty_like(t, dtype=float)
    for k, tk in enumerate(t):
        if tk <= t1:
            E[k] = E_start + s * v * tk
        else:
            E[k] = E_switch - s * v * (tk - t1)
    return E

def thomas_solve_inplace(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    d[-1] /= b[-1]
    for i in range(n - 2, -1, -1):
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i]
    return d

def simulate_cv_bv_flux(
    *,
    E0=0.0,
    E_start=0.25,
    E_switch=-0.25,
    v=0.2,
    n=1,
    T=298.15,
    alpha=0.5,
    k0=0.2,
    D=7e-6,
    C_bulk=1e-6,
    A=0.071,
    L=0.05,
    Nx=301,
    dt=2e-4,          # smaller dt to remove jitter
    picard_max=40,    # more iterations to converge stiff BV
    relax=0.15,       # stronger damping
    tol_j=1e-10,
    tol_c0=1e-12,
):
    print(">>> RUNNING: simulate_cv_bv_flux (flux BC + BV, noise-controlled)")

    beta = n * F / (R * T)
    dx = L / (Nx - 1)
    lam = D * dt / dx**2

    t1 = abs(E_switch - E_start) / v
    t = np.arange(0.0, 2.0 * t1 + dt, dt)
    E = triangular_potential(t, E_start, E_switch, v)

    CO = np.full(Nx, C_bulk, dtype=float)
    i = np.zeros_like(t)
    j = 0.0

    M = Nx - 1  # unknown nodes 0..Nx-2

    # Prebuild CN LHS (constant)
    aL = np.zeros(M - 1)
    bL = np.zeros(M)
    cL = np.zeros(M - 1)

    bL[0] = 1.0 + lam
    cL[0] = -lam

    for r in range(1, M - 1):
        aL[r - 1] = -0.5 * lam
        bL[r] = 1.0 + lam
        cL[r] = -0.5 * lam

    aL[M - 2] = -0.5 * lam
    bL[M - 1] = 1.0 + lam

    for k in range(len(t)):
        eta = E[k] - E0
        CO_old = CO.copy()
        j_old = j
        c0_old = CO[0]

        # adaptive damping if needed
        local_relax = relax

        converged = False
        for it in range(picard_max):
            CO0 = float(np.clip(CO[0], 0.0, C_bulk))
            CR0 = float(max(C_bulk - CO0, 0.0))

            j_bv = n * F * k0 * (
                CO0 * safe_exp(-alpha * beta * eta)
                - CR0 * safe_exp((1.0 - alpha) * beta * eta)
            )

            j_new = local_relax * j_bv + (1.0 - local_relax) * j_old

            g = dx * (j_new / (n * F * D))

            rhs = np.empty(M, dtype=float)
            rhs[0] = (1.0 - lam) * CO_old[0] + lam * CO_old[1] - lam * g

            for r in range(1, M - 1):
                rhs[r] = (1.0 - lam) * CO_old[r] + 0.5 * lam * (CO_old[r - 1] + CO_old[r + 1])

            rhs[M - 1] = (1.0 - lam) * CO_old[M - 1] + 0.5 * lam * (CO_old[M - 2] + C_bulk)

            a = aL.copy()
            b = bL.copy()
            c = cL.copy()
            sol = thomas_solve_inplace(a, b, c, rhs)

            CO_new = CO_old.copy()
            CO_new[:M] = sol
            CO_new[-1] = C_bulk
            CO_new = np.clip(CO_new, 0.0, C_bulk)

            # convergence check on BOTH flux and surface concentration
            dj = abs(j_new - j_old)
            dc0 = abs(CO_new[0] - c0_old)

            CO = CO_new
            j_old = j_new
            c0_old = CO_new[0]

            if dj < tol_j and dc0 < tol_c0:
                converged = True
                break

        # if not converged, do one more heavily damped update (prevents spiking)
        if not converged:
            local_relax = 0.05
            CO = CO_old.copy()
            j_old = j
            CO0 = float(np.clip(CO[0], 0.0, C_bulk))
            CR0 = float(max(C_bulk - CO0, 0.0))
            j_bv = n * F * k0 * (
                CO0 * safe_exp(-alpha * beta * eta)
                - CR0 * safe_exp((1.0 - alpha) * beta * eta)
            )
            j_new = local_relax * j_bv + (1.0 - local_relax) * j_old
            g = dx * (j_new / (n * F * D))

            rhs = np.empty(M, dtype=float)
            rhs[0] = (1.0 - lam) * CO_old[0] + lam * CO_old[1] - lam * g
            for r in range(1, M - 1):
                rhs[r] = (1.0 - lam) * CO_old[r] + 0.5 * lam * (CO_old[r - 1] + CO_old[r + 1])
            rhs[M - 1] = (1.0 - lam) * CO_old[M - 1] + 0.5 * lam * (CO_old[M - 2] + C_bulk)

            a = aL.copy()
            b = bL.copy()
            c = cL.copy()
            sol = thomas_solve_inplace(a, b, c, rhs)
            CO_new = CO_old.copy()
            CO_new[:M] = sol
            CO_new[-1] = C_bulk
            CO = np.clip(CO_new, 0.0, C_bulk)
            j = j_new
        else:
            j = j_old

        i[k] = j * A

    return E, i, t

def main():
    E0 = 0.0
    E_start = 0.25
    E_switch = -0.25
    v = 0.2
    k0 = 0.2

    E, i, t = simulate_cv_bv_flux(
        E0=E0, E_start=E_start, E_switch=E_switch,
        v=v, k0=k0,
        D=7e-6, C_bulk=1e-6, A=0.071,
        L=0.05, Nx=301, dt=2e-4
    )

    k_switch = int(np.argmin(np.abs(E - E_switch)))
    k_switch = max(1, min(k_switch, len(E) - 2))

    E_fwd = E[:k_switch + 1]
    i_fwd = i[:k_switch + 1]
    E_rev = E[k_switch:]
    i_rev = i[k_switch:]

    idx_pc = int(np.argmax(i_fwd))
    E_pc = E_fwd[idx_pc]
    i_pc = i_fwd[idx_pc]

    skip = max(10, int(0.05 * len(i_rev)))
    idx_pa = int(np.argmin(i_rev[skip:])) + skip
    E_pa = E_rev[idx_pa]
    i_pa = i_rev[idx_pa]

    print("\nCathodic peak (forward, +):")
    print(f"  E_pc = {E_pc:.4f} V")
    print(f"  i_pc = {i_pc*1e6:.2f} µA")
    print("Anodic peak (reverse, -):")
    print(f"  E_pa = {E_pa:.4f} V")
    print(f"  i_pa = {i_pa*1e6:.2f} µA")
    print("Peak separation:")
    print(f"  ΔE_p = {abs(E_pa - E_pc)*1000:.1f} mV")
    ratio = abs(i_pa / i_pc) if i_pc != 0 else np.nan
    print(f"Peak ratio |ipa/ipc| = {ratio:.3f}")
    print("Current range (global):")
    print(f"  i_min = {np.min(i)*1e6:.2f} µA, i_max = {np.max(i)*1e6:.2f} µA")

    # --- ensure plots directory exists ---
    os.makedirs("plots", exist_ok=True)

    # --- plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(E, i, lw=2)
    # Mark peaks
    plt.plot(E_pc, i_pc, "o", ms=8)
    plt.plot(E_pa, i_pa, "o", ms=8)
    plt.xlabel("Potential (V)")
    plt.ylabel("Current (A)")
    plt.title("Simulated Cyclic Voltammogram (Diffusion + Butler–Volmer)")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    # --- save plot ---
    plot_path = "plots/cv_duck_shape.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()