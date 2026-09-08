"""Solve the 1D discounted MFG with the Picard method."""

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
rho = 0.5                # Discount factor
mu = 0                   # Analytical center
nu = 0.5                 # Viscosity coefficient

# --- Spatial and Temporal Grid ---
width = 2.0
xmin, xmax = mu - width, mu + width # Spatial domain boundaries
T  = 10.0                # Total time horizon
Nx = 200                 # Number of spatial intervals
Nt = 200                 # Number of temporal intervals

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 100    # Maximum number of outer HJB-FP coupling iterations
PICARD_TOL = 1e-6        # Convergence tolerance for the outer Picard loop
DAMPING = 0.8            # Damping parameter (0 < DAMPING <= 1). 1 corresponds to no damping.

NEWTON_MAX_ITER = 20     # Maximum Newton iterations for the backward HJB solves
NEWTON_TOL = 1e-9        # Convergence tolerance for the backward Newton solver

# --- Derived Quantities ---
L  = xmax - xmin         # Length of the spatial domain
Dx = L / Nx              # Spatial step size
dt = T / Nt              # Temporal step size
x  = np.linspace(xmin, xmax, Nx + 1) # Computational grid points (Dirichlet)
num_x = x.size
norm_const = np.sqrt(Dx * dt)

# --- Saved File Names ---
history_filename = "discounted-1d-picard-history.txt"
plot_snapshot_filename = "discounted-1d-picard-plot.png"
value_data_filename = "discounted-1d-picard-value-function.dat"
density_data_filename = "discounted-1d-picard-density.dat"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN) & EXACT SOLUTION
# ==========================================

# 1. Quadratic Coefficient (Curvature)
eta = (1.0 - rho * nu) / (2.0 * nu)

# 2. Gaussian Variance (Consistency Condition)
s_sq = nu / (2.0 * eta)
m_peak = 1.0 / np.sqrt(2.0 * np.pi * s_sq)

# 3. Value Function Offset (omega)
omega = (2.0 * nu * eta - np.log(m_peak)) / rho

def get_exact_m(x_vals):
    return m_peak * np.exp(-(x_vals - mu)**2 / (2.0 * s_sq))

def get_exact_u(x_vals):
    return eta * (x_vals - mu)**2 + omega

m_star = get_exact_m(x)
u_star = get_exact_u(x)
m_bnd_val = m_star[0]
u_bnd_val = u_star[0]

# Initial/Terminal Conditions
m0 = m_star.copy()
uT = u_star.copy()

def compute_f(m_dist):
    m_safe = np.maximum(m_dist, 1e-12)
    return -np.log(m_safe)

def discrete_Hamiltonian(dp, dm, f_val):
    H_p = 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2
    return H_p - f_val

def dH_dp(p):
    return p

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I = sp.eye(num_x)

# Forward Difference Matrix (D+) - Dirichlet
data_p = np.array([-np.ones(num_x), np.ones(num_x)])
D_plus = sp.spdiags(data_p, [0, 1], num_x, num_x).tolil()
D_plus[num_x-1, :] = 0
D_plus = D_plus.tocsr() / Dx

# Backward Difference Matrix (D-) - Dirichlet
data_m = np.array([-np.ones(num_x), np.ones(num_x)])
D_minus = sp.spdiags(data_m, [-1, 0], num_x, num_x).tolil()
D_minus[0, :] = 0
D_minus = D_minus.tocsr() / Dx

# Laplacian Matrix
Laplacian = D_plus @ D_minus

# ==========================================
# SECTION 4: HJB AND FP SOLVERS
# ==========================================

def solve_hjb_backward(M_flow):
    U = np.zeros((Nt + 1, num_x))
    U[Nt] = uT

    A_diff = (1.0 + rho * dt) * I - dt * nu * Laplacian
    A_diff = A_diff.tolil()
    A_diff[0, :], A_diff[0, 0] = 0.0, 1.0
    A_diff[-1, :], A_diff[-1, -1] = 0.0, 1.0
    A_diff = A_diff.tocsr()

    total_newton = 0
    for n in range(Nt - 1, -1, -1):
        u_next = U[n+1]
        u_curr = u_next.copy()
        f_val = compute_f(M_flow[n+1])

        for _ in range(NEWTON_MAX_ITER):
            total_newton += 1
            dp, dm = D_plus @ u_curr, D_minus @ u_curr

            H_total = discrete_Hamiltonian(dp, dm, f_val)
            F = A_diff @ u_curr + dt * H_total - u_next
            F[0] = u_curr[0] - u_bnd_val
            F[-1] = u_curr[-1] - u_bnd_val

            if np.linalg.norm(F, np.inf) < NEWTON_TOL: break

            p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
            dH_dU = sp.diags(dH_dp(p_min)) @ D_plus + sp.diags(dH_dp(p_max)) @ D_minus

            J = A_diff + dt * dH_dU
            J = J.tolil()
            J[0, :], J[0, 0] = 0.0, 1.0
            J[-1, :], J[-1, -1] = 0.0, 1.0

            u_curr -= spla.spsolve(J.tocsr(), F)
        U[n] = u_curr
    return U, total_newton

def solve_fp_forward(U_flow):
    M = np.zeros((Nt + 1, num_x))
    M[0] = m0

    for n in range(Nt):
        m_curr, u_curr = M[n], U_flow[n]

        dp, dm = D_plus @ u_curr, D_minus @ u_curr
        J_G = (sp.diags(dH_dp(np.minimum(dp, 0))) @ D_plus
               + sp.diags(dH_dp(np.maximum(dm, 0))) @ D_minus).tolil()
        # The HJB boundary rows are Dirichlet equations, so their Hamiltonian
        # rows must be removed before taking the discrete adjoint.
        J_G[0, :], J_G[-1, :] = 0.0, 0.0
        J_G = J_G.tocsr()
        Adv_Op = J_G.transpose()

        A = (I - dt * nu * Laplacian + dt * Adv_Op).tolil()
        A[0, :], A[0, 0] = 0.0, 1.0
        A[-1, :], A[-1, -1] = 0.0, 1.0

        rhs = m_curr.copy()
        rhs[0], rhs[-1] = m_bnd_val, m_bnd_val

        M[n+1] = spla.spsolve(A.tocsr(), rhs)
        M[n+1] = np.maximum(M[n+1], 1e-15)

    return M

# ==========================================
# SECTION 5: DAMPED PICARD ITERATION
# ==========================================

start_time_total = time.time()
with open(history_filename, "w", buffering=1) as f:
    header = (
        f"{'='*118}\n"
        f"   DISCOUNTED MFG EXAMPLE (Picard)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {nu}, rho = {rho}, mu = {mu}\n"
        f"Solver Parameters:\n"
        f"  PICARD_MAX_ITER = {PICARD_MAX_ITER}, PICARD_TOL = {PICARD_TOL}\n"
        f"  DAMPING = {DAMPING}\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  H = |p|^2/2 + log(m) (running cost -log(m)), boundary = exact stationary Dirichlet\n"
        f"  initial_m = exact stationary profile, terminal_u = exact stationary profile\n"
        f"  eta = {eta:.12g}, variance = {s_sq:.12g}, m_peak = {m_peak:.12g}, omega = {omega:.12g}\n"
        f"  m_boundary = {m_bnd_val:.12g}, u_boundary = {u_bnd_val:.12g}\n"
        f"  HJB_flux = one-sided Godunov, FP_transport = discrete adjoint of HJB transport\n"
        f"{'-'*118}\n"
        f"{'Iter':<5} | {'Abs Err U':<12} | {'Rel Err U':<12} | {'Abs Err M':<12} | {'Rel Err M':<12} | {'Newton It':<10} | Time (s)\n"
        f"{'-'*118}\n"
    )
    print(header, end='')
    f.write(header)

    M_flow = np.zeros((Nt + 1, num_x))
    M_flow[0] = m0.copy()
    A_diff_only = (I - dt * nu * Laplacian).tolil()
    A_diff_only[0, :], A_diff_only[0, 0] = 0.0, 1.0
    A_diff_only[-1, :], A_diff_only[-1, -1] = 0.0, 1.0
    for n in range(Nt):
        rhs = M_flow[n].copy(); rhs[0], rhs[-1] = m_bnd_val, m_bnd_val
        M_flow[n+1] = spla.spsolve(A_diff_only.tocsr(), rhs)
        M_flow[n+1] = np.maximum(M_flow[n+1], 1e-15)

    U_flow = np.zeros((Nt + 1, num_x))

    for k in range(1, PICARD_MAX_ITER + 1):
        step_start = time.time()

        U_cand, n_it = solve_hjb_backward(M_flow)
        U_flow_new = DAMPING * U_cand + (1 - DAMPING) * U_flow
        M_cand = solve_fp_forward(U_flow_new)
        M_flow_new = DAMPING * M_cand + (1 - DAMPING) * M_flow

        delta_u_norm = np.linalg.norm((U_flow_new - U_flow).ravel())
        delta_m_norm = np.linalg.norm((M_flow_new - M_flow).ravel())
        previous_u_norm = np.linalg.norm(U_flow.ravel())
        previous_m_norm = np.linalg.norm(M_flow.ravel())
        abs_err_u = delta_u_norm * norm_const
        rel_err_u = delta_u_norm / previous_u_norm if previous_u_norm > 1e-12 else (0.0 if delta_u_norm <= 1e-12 else 1.0)
        abs_err_m = delta_m_norm * norm_const
        rel_err_m = delta_m_norm / previous_m_norm if previous_m_norm > 1e-12 else (0.0 if delta_m_norm <= 1e-12 else 1.0)

        step_time = time.time() - step_start

        log_str = f"{k:<5} | {abs_err_u:.4e}   | {rel_err_u:.4e}   | {abs_err_m:.4e}   | {rel_err_m:.4e}   | {n_it:<10} | {step_time:.4f}"
        print(log_str)
        f.write(log_str + "\n")

        U_flow, M_flow = U_flow_new, M_flow_new
        if rel_err_u < PICARD_TOL and rel_err_m < PICARD_TOL:
            conv_msg = f"{'-'*118}\nCONVERGED at nu={nu} in {k} iterations.\n"
            print(conv_msg)
            f.write(conv_msg)
            break

    total_time = time.time() - start_time_total
    time_msg = f"Total Execution Time: {total_time:.4f} seconds.\n"
    print(time_msg)
    f.write(time_msg)

# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

idx_mid = Nt // 2
fig2 = plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(x, u_star, 'k--', linewidth=2, label='Stationary analytical $u^*$')
plt.plot(x, U_flow[idx_mid], 'r-', linewidth=2, label=f'Numerical $u(T/2)$')
plt.title(f'Value Function Comparison (T={T})')
plt.xlabel('x'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(x, m_star, 'k--', linewidth=2, label='Stationary analytical $m^*$')
plt.plot(x, M_flow[idx_mid], 'b-', linewidth=2, label=f'Numerical $m(T/2)$')
plt.title(f'Density Comparison (T={T})')
plt.xlabel('x'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(plot_snapshot_filename); plt.close(fig2)

# Save the 1D arrays to data files for TikZ/LaTeX plotting
np.savetxt(value_data_filename, np.column_stack((x, u_star, U_flow[idx_mid])), header='x u_star u_num', comments='', fmt='%.6e')
np.savetxt(density_data_filename, np.column_stack((x, m_star, M_flow[idx_mid])), header='x m_star m_num', comments='', fmt='%.6e')
