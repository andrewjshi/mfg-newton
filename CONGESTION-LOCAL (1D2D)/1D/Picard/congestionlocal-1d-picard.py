"""Solve the 1D local-congestion MFG with the Picard method."""

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
nu   = 0.01              # Viscosity coefficient
beta = 1.5               # Congestion exponent
zeta = 1.0               # Coupling strength

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0    # Spatial domain boundaries
T  = 1.0                 # Total time horizon
Nx = 200                 # Number of spatial intervals
Nt = 100                 # Number of temporal intervals

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 300    # Maximum number of outer HJB-FP coupling iterations
PICARD_TOL = 1e-6        # Convergence tolerance for the outer Picard loop
DAMPING = 0.8            # Damping parameter (0 < DAMPING <= 1). 1 corresponds to no damping.

NEWTON_MAX_ITER = 20     # Maximum Newton iterations for the backward HJB solves
NEWTON_TOL = 1e-9        # Convergence tolerance for the backward Newton solver

# --- Derived Quantities ---
L  = xmax - xmin         # Length of the spatial domain
Dx = L / Nx              # Spatial step size
dt = T / Nt              # Temporal step size
x  = np.linspace(xmin, xmax, Nx, endpoint=False) # Computational grid points (Periodic)
num_x = x.size
norm_const = np.sqrt(Dx * dt)

# --- Saved File Names ---
history_filename = "congestionlocal-1d-picard-history.txt"
plot_snapshot_filename = "congestionlocal-1d-picard-plot.png"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# Initial condition m0: Square Wave on [0.375, 0.625]
m0 = np.zeros(num_x)
mask = (x >= 0.375) & (x <= 0.625)
m0[mask] = 4.0
m0 = m0 / (np.sum(m0) * Dx)

# Terminal cost: double quadratic well at x=0.3 and x=0.7
uT = 15.0 * np.minimum((x - 0.3)**2, (x - 0.7)**2)

def compute_f(m_dist):
    """Coupling/running cost f(x, m) = zeta * m."""
    return zeta * m_dist

def discrete_Hamiltonian(dp, dm, m_curr, f_val):
    """H = |p|^2 / (2*(1+4m)^beta) - zeta*m"""
    m_safe     = np.maximum(m_curr, 1e-10)
    cong_coeff = 1.0 / (2.0 * (1.0 + 4.0 * m_safe)**beta)
    p_sq       = np.minimum(dp, 0)**2 + np.maximum(dm, 0)**2
    return cong_coeff * p_sq - f_val

def dH_dp(p, m_curr):
    """D_p H = p / (1+4m)^beta"""
    m_safe     = np.maximum(m_curr, 1e-10)
    cong_coeff = 1.0 / (2.0 * (1.0 + 4.0 * m_safe)**beta)
    return 2.0 * cong_coeff * p

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I = sp.eye(num_x)

# Forward Difference Matrix (D+) - Periodic
data_p = np.array([-np.ones(num_x), np.ones(num_x)])
D_plus = sp.spdiags(data_p, [0, 1], num_x, num_x).tolil()
D_plus[num_x-1, num_x-1] = -1.0
D_plus[num_x-1, 0]       =  1.0
D_plus = D_plus.tocsr() / Dx

# Backward Difference Matrix (D-) - Periodic
data_m = np.array([-np.ones(num_x), np.ones(num_x)])
D_minus = sp.spdiags(data_m, [-1, 0], num_x, num_x).tolil()
D_minus[0, 0]       =  1.0
D_minus[0, num_x-1] = -1.0
D_minus = D_minus.tocsr() / Dx

# Laplacian Matrix
Laplacian = D_plus @ D_minus

# ==========================================
# SECTION 4: HJB AND FP SOLVERS
# ==========================================

def solve_hjb_backward(M_flow):
    U      = np.zeros((Nt + 1, num_x))
    U[Nt]  = uT

    A_diff = (I - dt * nu * Laplacian).tocsr()

    total_newton = 0
    for n in range(Nt - 1, -1, -1):
        u_next = U[n+1]
        u_curr = u_next.copy()
        m_curr = M_flow[n+1]
        f_val  = compute_f(m_curr)

        for _ in range(NEWTON_MAX_ITER):
            total_newton += 1
            dp, dm  = D_plus @ u_curr, D_minus @ u_curr

            H_total = discrete_Hamiltonian(dp, dm, m_curr, f_val)
            F       = A_diff @ u_curr + dt * H_total - u_next

            if np.linalg.norm(F, np.inf) < NEWTON_TOL:
                break

            p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
            dH_dU = (sp.diags(dH_dp(p_min, m_curr)) @ D_plus +
                     sp.diags(dH_dp(p_max, m_curr)) @ D_minus)
            J = A_diff + dt * dH_dU
            u_curr -= spla.spsolve(J, F)
        U[n] = u_curr
    return U, total_newton

def solve_fp_forward(U_flow, M_old_flow):
    M    = np.zeros((Nt + 1, num_x))
    M[0] = m0
    for n in range(Nt):
        m_ref  = M_old_flow[n+1]
        dp, dm = D_plus @ U_flow[n], D_minus @ U_flow[n]
        J_H = (sp.diags(dH_dp(np.minimum(dp, 0), m_ref)) @ D_plus +
               sp.diags(dH_dp(np.maximum(dm, 0), m_ref)) @ D_minus).tocsr()
        A      = (I - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
        M[n+1] = spla.spsolve(A, M[n])
    return M

# ==========================================
# SECTION 5: DAMPED PICARD ITERATION
# ==========================================

start_time_all = time.time()

with open(history_filename, "w", buffering=1) as f_log:
    header = (
        f"{'='*118}\n"
        f"   LOCAL CONGESTION MFG EXAMPLE (Picard)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {nu}, beta = {beta}, zeta = {zeta}\n"
        f"Solver Parameters:\n"
        f"  PICARD_MAX_ITER = {PICARD_MAX_ITER}, PICARD_TOL = {PICARD_TOL}\n"
        f"  DAMPING = {DAMPING}\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  initial_support = [0.375, 0.625], initial_height = 4.0, boundary = periodic\n"
        f"  terminal_wells = [0.3, 0.7], terminal_scale = 15.0, congestion_scale = 4.0\n"
        f"{'-'*118}\n"
        f"{'Iter':<5} | {'Abs Err U':<12} | {'Rel Err U':<12} | {'Abs Err M':<12} | {'Rel Err M':<12} | {'Newton It':<10} | {'Time (s)':<10}\n"
        f"{'-'*118}\n"
    )
    print(header, end=''); f_log.write(header)

    # Initialise M_flow by diffusing m0 forward
    M_flow      = np.tile(m0, (Nt+1, 1))
    A_diff_only = (I - dt * nu * Laplacian).tocsr()
    for n in range(Nt):
        M_flow[n+1] = spla.spsolve(A_diff_only, M_flow[n])

    U_flow     = np.zeros((Nt + 1, num_x))

    for k in range(1, PICARD_MAX_ITER + 1):
        t0 = time.time()

        U_candidate, n_iters = solve_hjb_backward(M_flow)
        U_flow_new  = DAMPING * U_candidate  + (1 - DAMPING) * U_flow
        M_candidate = solve_fp_forward(U_flow_new, M_flow)
        M_flow_new  = DAMPING * M_candidate  + (1 - DAMPING) * M_flow

        delta_u_norm = np.linalg.norm((U_flow_new - U_flow).ravel())
        delta_m_norm = np.linalg.norm((M_flow_new - M_flow).ravel())
        previous_u_norm = np.linalg.norm(U_flow.ravel())
        previous_m_norm = np.linalg.norm(M_flow.ravel())
        abs_err_u = delta_u_norm * norm_const
        rel_err_u = delta_u_norm / previous_u_norm if previous_u_norm > 1e-12 else (0.0 if delta_u_norm <= 1e-12 else 1.0)
        abs_err_m = delta_m_norm * norm_const
        rel_err_m = delta_m_norm / previous_m_norm if previous_m_norm > 1e-12 else (0.0 if delta_m_norm <= 1e-12 else 1.0)

        iter_time = time.time() - t0
        log_str = (f"{k:<5} | {abs_err_u:.4e}   | {rel_err_u:.4e}   | "
                   f"{abs_err_m:.4e}   | {rel_err_m:.4e}   | {n_iters:<10} | {iter_time:.4f}")
        print(log_str); f_log.write(log_str + "\n")

        U_flow, M_flow = U_flow_new, M_flow_new
        if rel_err_u < PICARD_TOL and rel_err_m < PICARD_TOL:
            conv_msg = f"{'-'*118}\nCONVERGED at nu={nu} in {k} iterations.\n"
            print(conv_msg)
            f_log.write(conv_msg)
            break

    total_time = time.time() - start_time_all
    time_msg = f"Total Execution Time: {total_time:.4f} seconds.\n"
    print(time_msg)
    f_log.write(time_msg)

# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

target_times = [0.0, T/5, 2*T/5, 3*T/5, 4*T/5, T]
colors = plt.cm.jet(np.linspace(0, 1, len(target_times)))

axes[0].imshow(M_flow, aspect='auto', extent=[xmin, xmax, T, 0], cmap='viridis')
axes[0].set_title('Density Evolution $(m)$')
axes[0].set_xlabel('x'); axes[0].set_ylabel('t')

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[1].plot(x, M_flow[n_idx], color=colors[i], label=f't={t_val:.2f}')
axes[1].set_title('Density $(m)$ Snapshots')
axes[1].set_xlabel('x'); axes[1].set_ylabel('m(t,x)')
axes[1].legend(fontsize=8)

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[2].plot(x, U_flow[n_idx], color=colors[i], label=f't={t_val:.2f}')
axes[2].set_title('Value $(u)$ Snapshots')
axes[2].set_xlabel('x'); axes[2].set_ylabel('u(t,x)')

plt.tight_layout()
plt.savefig(plot_snapshot_filename, dpi=150)
plt.show()
print(f"Saved: {plot_snapshot_filename}")
