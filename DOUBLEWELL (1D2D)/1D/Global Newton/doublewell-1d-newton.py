"""Solve the 1D double-well MFG with Global Newton and viscosity continuation."""

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ==========================================
# SECTION 1: PARAMETERS AND GRID DISCRETIZATION
# ==========================================

# --- Physics Parameters ---
coupling_str = 1.0       # kappa: Power-law coupling strength
tilt = 0.3               # Linear tilt parameter for the double-well potential asymmetry
well_depth = 2.0         # Depth of the wells in the V(x) potential
alpha = 2                # Power-law exponent: f(m) = kappa * m^alpha
initial_std = 0.6        # Standard deviation of the initial Gaussian density

# --- Spatial and Temporal Grid ---
xmin, xmax = -2.5, 2.5   # Spatial domain boundaries
T = 1.0                  # Total time horizon
Nx = 500                 # Number of spatial intervals
Nt = 100                 # Number of temporal intervals

# --- Newton Iteration Parameters ---
NEWTON_MAX_ITER = 20     # Maximum Global Newton iterations per viscosity level
NEWTON_TOL = 1e-9        # Convergence tolerance for the monolithic residual

# --- Continuation Parameters ---
NU_START = 0.5           # Starting viscosity
NU_TARGET = 0.1          # Target viscosity
TAU = 0.1               # Reduce the viscosity by a factor of ten after each successful solve.
GAMMA = 0.5              # Backtracking factor
CONT_TOL = 1e-4          # Minimum allowable step size

# --- Derived Quantities ---
L  = xmax - xmin         # Length of the spatial domain
Dx = L / Nx              # Spatial step size
dt = T / Nt              # Temporal step size
x  = xmin + np.arange(Nx) * Dx  # Grid nodes on the periodic interval [xmin, xmax)
num_x = x.size

# --- Saved File Names ---
history_filename = "doublewell-1d-newton-history.txt"
plot_snapshot_filename = "doublewell-1d-newton-plot.png"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

V = well_depth * (x**2 - 1.0)**2 + tilt * x

m0 = np.exp(-x**2 / (2 * initial_std**2))
m0 /= (np.sum(m0) * Dx)
uT = V.copy()

def compute_f(m_dist):
    return V + coupling_str * (np.maximum(m_dist, 0)**alpha)

def discrete_Hamiltonian(dp, dm, f_val):
    return 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2 - f_val

def dH_dp(p):
    return p

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I = sp.eye(num_x)

# Forward difference with periodic wrap-around at the right boundary.
data_p = np.array([-np.ones(num_x), np.ones(num_x)])
D_plus = sp.spdiags(data_p, [0, 1], num_x, num_x).tolil()
D_plus[num_x-1, num_x-1] = -1.0
D_plus[num_x-1, 0] = 1.0
D_plus = D_plus.tocsr() / Dx

# Backward difference with periodic wrap-around at the left boundary.
data_m = np.array([-np.ones(num_x), np.ones(num_x)])
D_minus = sp.spdiags(data_m, [-1, 0], num_x, num_x).tolil()
D_minus[0, 0] = 1.0
D_minus[0, num_x-1] = -1.0
D_minus = D_minus.tocsr() / Dx

# Periodic Laplacian.  On the torus D_plus^T = -D_minus, so this is the standard
# three-point stencil; its row and column sums vanish, so its transpose is also
# the conservative FP diffusion operator.
Laplacian = -(D_plus.transpose() @ D_plus).tocsr()

# ==========================================
# SECTION 4: GLOBAL NEWTON SOLVER
# ==========================================

def global_newton_step(nu_val, W_guess, log_file=None):
    """Apply monolithic Global Newton at one fixed viscosity value."""
    if W_guess is None:
        U = np.tile(uT, (Nt + 1, 1))
        M = np.tile(m0, (Nt + 1, 1))
    else:
        U = W_guess[:(Nt+1)*num_x].reshape((Nt+1, num_x))
        M = W_guess[(Nt+1)*num_x:].reshape((Nt+1, num_x))

    for it in range(NEWTON_MAX_ITER):
        J_UU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_UM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]

        F_U = np.zeros((Nt+1, num_x))
        F_M = np.zeros((Nt+1, num_x))

        # Terminal condition for U
        J_UU[Nt][Nt] = I.tocsr()
        F_U[Nt] = U[Nt] - uT
        J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        # Initial condition for M
        J_MM[0][0] = I.tocsr()
        F_M[0] = M[0] - m0
        J_MU[0][0] = J_MU[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        for n in range(Nt):
            un, un_next = U[n], U[n+1]
            mn, mn_next = M[n], M[n+1]
            dp, dm = D_plus @ un, D_minus @ un
            p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
            f_val = compute_f(mn_next)
            J_H = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()

            # --- Backward HJB Equation ---
            F_U[n] = (un - un_next) / dt - nu_val * Laplacian @ un + discrete_Hamiltonian(dp, dm, f_val)

            dH_dm = -coupling_str * alpha * (np.maximum(mn_next, 0)**(alpha-1))
            J_UU[n][n]   = ((1.0/dt) * I - nu_val * Laplacian + J_H).tocsr()
            J_UU[n][n+1] = (-(1.0/dt) * I).tocsr()
            J_UM[n][n]   = sp.csr_matrix((num_x, num_x))
            J_UM[n][n+1] = sp.diags(dH_dm).tocsr()

            # --- Forward FP Equation ---
            F_M[n+1] = (mn_next - mn) / dt - nu_val * Laplacian @ mn_next + J_H.transpose() @ mn_next

            J_MM[n+1][n+1] = ((1.0/dt) * I - nu_val * Laplacian + J_H.transpose()).tocsr()
            J_MM[n+1][n]   = (-(1.0/dt) * I).tocsr()
            J_MU[n+1][n]   = (D_plus.transpose() @ sp.diags(mn_next * (dp < 0)) @ D_plus + D_minus.transpose() @ sp.diags(mn_next * (dm > 0)) @ D_minus).tocsr()

        J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)], [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()
        F = np.concatenate([F_U.flatten(), F_M.flatten()])
        res = np.linalg.norm(F, np.inf)

        if np.isnan(res) or np.isinf(res):
            return np.concatenate([U.flatten(), M.flatten()]), False, res
        if res < NEWTON_TOL:
            iter_log = f"      Iter {it:2d} | res = {res:.4e} (Converged)"
            print(iter_log)
            if log_file: log_file.write(iter_log + "\n")
            return np.concatenate([U.flatten(), M.flatten()]), True, res

        # --- TIMER FOR LINEAR SOLVE ---
        t_solve_start = time.time()
        try:
            dW = spla.spsolve(J, F)
            t_solve_elapsed = time.time() - t_solve_start

            iter_log = f"      Iter {it:2d} | res = {res:.4e} | Solve Time: {t_solve_elapsed:.4f}s"
            print(iter_log)
            if log_file: log_file.write(iter_log + "\n")

            U = (U.flatten() - dW[:(Nt+1)*num_x]).reshape((Nt+1, num_x))
            M = (M.flatten() - dW[(Nt+1)*num_x:]).reshape((Nt+1, num_x))
        except Exception as exc:
            failure_msg = f"      Linear solve failed: {exc}"
            print(failure_msg)
            if log_file:
                log_file.write(failure_msg + "\n")
            return np.concatenate([U.flatten(), M.flatten()]), False, res

    return np.concatenate([U.flatten(), M.flatten()]), False, res

# ==========================================
# SECTION 5: VISCOSITY CONTINUATION MAIN LOOP
# ==========================================

start_time_all = time.time()
with open(history_filename, "w", buffering=1) as f:
    header = (
        f"{'='*118}\n"
        f"   DOUBLE WELL MFG EXAMPLE (Global Newton)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {NU_TARGET}, alpha = {alpha}, kappa = {coupling_str}\n"
        f"Solver Parameters:\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  Continuation: NU_START={NU_START}, TAU={TAU}, GAMMA={GAMMA}, CONT_TOL={CONT_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  tilt = {tilt}, well_depth = {well_depth}, initial_std = {initial_std}, boundary = periodic\n"
        f"{'-'*118}\n"
    )
    print(header, end=""); f.write(header)

    nu_curr = NU_START
    print(f"First Solve: nu = {nu_curr:.4f}"); f.write(f"First Solve: nu = {nu_curr:.4f}\n")

    t_nu_start = time.time()
    W, converged, res = global_newton_step(nu_curr, None, f)
    t_nu_elapsed = time.time() - t_nu_start

    if not converged: sys.exit()
    conv_msg = f"    -> Converged! (res={res:.4e}) Total for nu={nu_curr:.4f}: {t_nu_elapsed:.4f}s\n"
    print(conv_msg); f.write(conv_msg + "\n")

    while nu_curr > NU_TARGET:
        nu_next = max(NU_TARGET, TAU * nu_curr)
        success = False
        while not success:
            print(f"  Attempting nu = {nu_next:.4f}..."); f.write(f"  Attempting nu = {nu_next:.4f}...\n")

            t_nu_start = time.time()
            W_trial, conv, r = global_newton_step(nu_next, W, f)
            t_nu_elapsed = time.time() - t_nu_start

            if conv:
                W, nu_curr, success = W_trial, nu_next, True
                conv_msg = f"    -> Converged! (res={r:.4e}) Total for nu={nu_next:.4f}: {t_nu_elapsed:.4f}s\n"
                print(conv_msg); f.write(conv_msg + "\n")
            else:
                fail_msg = f"    -> Failed (res={r:.4e}) in {t_nu_elapsed:.4f}s. Backtracking...\n"
                print(fail_msg); f.write(fail_msg)
                nu_next = nu_curr - GAMMA * (nu_curr - nu_next)
                if abs(nu_curr - nu_next) < CONT_TOL: sys.exit()

    total_time = time.time() - start_time_all
    time_msg = f"{'-'*118}\nTotal Execution Time: {total_time:.4f} seconds.\n"
    print(time_msg); f.write(time_msg)

# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

U_flow = W[:(Nt+1)*num_x].reshape((Nt+1, num_x))
M_flow = W[(Nt+1)*num_x:].reshape((Nt+1, num_x))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
target_times = np.linspace(0, T, 5)
colors = plt.cm.jet(np.linspace(0, 1, len(target_times)))

im = axes[0].imshow(M_flow, aspect='auto', extent=[xmin, xmax, T, 0], cmap='viridis')
axes[0].set_title('Density Evolution (m)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
fig.colorbar(im, ax=axes[0])

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[1].plot(x, M_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val:.2f}')
axes[1].set_title(f'Density Snapshots (T={T})')
axes[1].set_xlabel('x')
axes[1].grid(True)
axes[1].legend(fontsize=9)

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[2].plot(x, U_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val:.2f}')
axes[2].set_title(f'Value Function Snapshots (T={T})')
axes[2].set_xlabel('x')
axes[2].grid(True)
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(plot_snapshot_filename)
plt.close()
