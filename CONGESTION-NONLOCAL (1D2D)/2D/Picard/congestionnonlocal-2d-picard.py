"""Solve the 2D nonlocal-congestion MFG with the Picard method."""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ==========================================
# SECTION 1: PARAMETERS AND GRID DISCRETIZATION
# ==========================================

# --- Physics Parameters ---
nu           = 0.01              # Viscosity — same as 1D nonlocal
beta         = 1.5               # Congestion exponent — same as 1D nonlocal
zeta         = 1.0               # Coupling strength — same as 1D nonlocal
sigma_kernel = 0.2               # Gaussian kernel std — same as 1D nonlocal
                                 # Wide enough to smooth the initial square-blob
                                 # so nonlocal agents feel less congestion and
                                 # begin moving earlier than local agents.

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0    # Spatial domain boundaries
ymin, ymax = 0.0, 1.0    # Spatial domain boundaries
T  = 1.0                 # Total time horizon
Nx = 200                 # Number of spatial intervals
Ny = 200                 # Number of spatial intervals
Nt = 20                  # Number of temporal intervals

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 150    # Maximum number of outer HJB-FP coupling iterations
PICARD_TOL = 1e-6        # Convergence tolerance for the outer Picard loop
DAMPING = 0.2            # Damping parameter (0 < DAMPING <= 1). 1 corresponds to no damping.
                         # Conservative damping stabilizes the nonlocal coupling.

NEWTON_MAX_ITER = 20     # Maximum Newton iterations for the backward HJB solves
NEWTON_TOL = 1e-9        # Convergence tolerance for the backward Newton solver

# --- Derived Quantities ---
dt    = T / Nt           # Temporal step size
Dx    = (xmax - xmin) / Nx # Spatial step size (x)
Dy    = (ymax - ymin) / Ny # Spatial step size (y)
num_x = Nx * Ny
norm_const = np.sqrt(Dx * Dy * dt)

x = np.linspace(xmin, xmax, Nx, endpoint=False) # Computational grid points (Periodic)
y = np.linspace(ymin, ymax, Ny, endpoint=False) # Computational grid points (Periodic)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Saved File Names ---
HERE = Path(__file__).resolve().parent   # outputs land beside this script
history_filename = str(HERE / "congestionnonlocal-2d-picard-history.txt")
plot_contour_filename = str(HERE / "congestionnonlocal-2d-picard-contour.png")
plot_surface_filename = str(HERE / "congestionnonlocal-2d-picard-surface.png")
# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# Initial condition m0: Square blob on [0.375, 0.625]^2 — same as local 2D
m0_2d = np.zeros((Nx, Ny))
mask  = (X >= 0.375) & (X <= 0.625) & (Y >= 0.375) & (Y <= 0.625)
m0_2d[mask] = 4.0
m0_2d = m0_2d / (np.sum(m0_2d) * Dx * Dy)
m0    = m0_2d.flatten()

# Terminal cost: double quadratic well at (0.3, 0.5) and (0.7, 0.5)
uT_2d = 15.0 * np.minimum((X - 0.3)**2 + (Y - 0.5)**2,
                           (X - 0.7)**2 + (Y - 0.5)**2)
uT = uT_2d.flatten()

# --- 2D Gaussian convolution kernel, periodically extended in both directions ---
dist_x = np.minimum(np.abs(X - xmin), (xmax - xmin) - np.abs(X - xmin))
dist_y = np.minimum(np.abs(Y - ymin), (ymax - ymin) - np.abs(Y - ymin))
K_2d   = np.exp(-0.5 * (dist_x**2 + dist_y**2) / sigma_kernel**2)
K_2d  /= (np.sum(K_2d) * Dx * Dy)
K_fft2 = fft2(K_2d)

def compute_m_conv(m_flat):
    """Nonlocal density: m_bar = K * m via 2D circular convolution."""
    m_2d      = m_flat.reshape((Nx, Ny))
    m_conv_2d = np.real(ifft2(K_fft2 * fft2(m_2d))) * Dx * Dy
    return np.maximum(m_conv_2d, 1e-12).flatten()

def discrete_Hamiltonian(dpx, dmx, dpy, dmy, m_conv):
    """H = (|p_x|^2 + |p_y|^2) / (2*(1+4*m_bar)^beta) - zeta*m_bar"""
    mobility = (1.0 + 4.0 * m_conv)**beta
    px_sq    = np.minimum(dpx, 0)**2 + np.maximum(dmx, 0)**2
    py_sq    = np.minimum(dpy, 0)**2 + np.maximum(dmy, 0)**2
    return (px_sq + py_sq) / (2.0 * mobility) - zeta * m_conv

def dH_dp(p, m_conv):
    """D_p H = p / (1+4*m_bar)^beta"""
    mobility = (1.0 + 4.0 * m_conv)**beta
    return p / mobility

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I_1Dx = sp.eye(Nx)
I_1Dy = sp.eye(Ny)
I     = sp.eye(num_x)

# 1D Forward/Backward Differences — periodic, x-direction
data_px = np.array([-np.ones(Nx), np.ones(Nx)])
D_plus_1Dx = sp.spdiags(data_px, [0, 1], Nx, Nx).tolil()
D_plus_1Dx[Nx-1, Nx-1] = -1.0
D_plus_1Dx[Nx-1, 0]    =  1.0
D_plus_1Dx = D_plus_1Dx.tocsr() / Dx

D_minus_1Dx = sp.spdiags(data_px, [-1, 0], Nx, Nx).tolil()
D_minus_1Dx[0, 0]    =  1.0
D_minus_1Dx[0, Nx-1] = -1.0
D_minus_1Dx = D_minus_1Dx.tocsr() / Dx

# 1D Forward/Backward Differences — periodic, y-direction
data_py = np.array([-np.ones(Ny), np.ones(Ny)])
D_plus_1Dy = sp.spdiags(data_py, [0, 1], Ny, Ny).tolil()
D_plus_1Dy[Ny-1, Ny-1] = -1.0
D_plus_1Dy[Ny-1, 0]    =  1.0
D_plus_1Dy = D_plus_1Dy.tocsr() / Dy

D_minus_1Dy = sp.spdiags(data_py, [-1, 0], Ny, Ny).tolil()
D_minus_1Dy[0, 0]    =  1.0
D_minus_1Dy[0, Ny-1] = -1.0
D_minus_1Dy = D_minus_1Dy.tocsr() / Dy

# 2D Kronecker Lift
D_plus_x  = sp.kron(D_plus_1Dx,  I_1Dy).tocsr()
D_minus_x = sp.kron(D_minus_1Dx, I_1Dy).tocsr()
D_plus_y  = sp.kron(I_1Dx, D_plus_1Dy ).tocsr()
D_minus_y = sp.kron(I_1Dx, D_minus_1Dy).tocsr()

Laplacian = D_plus_x @ D_minus_x + D_plus_y @ D_minus_y

# ==========================================
# SECTION 4: HJB AND FP SOLVERS
# ==========================================

def direct_solve(A, b):
    """Sparse direct solve using SuperLU through SciPy's ``spsolve``."""
    return spla.spsolve(A.tocsc(), b)

def solve_hjb_backward(M_flow):
    U      = np.zeros((Nt + 1, num_x))
    U[Nt]  = uT
    A_diff = (I - dt * nu * Laplacian).tocsr()
    total_newton = 0

    for n in range(Nt - 1, -1, -1):
        u_next   = U[n+1]
        u_curr   = u_next.copy()
        m_conv   = compute_m_conv(M_flow[n+1])
        mobility = (1.0 + 4.0 * m_conv)**beta

        for _ in range(NEWTON_MAX_ITER):
            total_newton += 1
            dpx, dmx = D_plus_x  @ u_curr, D_minus_x @ u_curr
            dpy, dmy = D_plus_y  @ u_curr, D_minus_y @ u_curr

            F = A_diff @ u_curr + dt * discrete_Hamiltonian(dpx, dmx, dpy, dmy, m_conv) - u_next
            if np.linalg.norm(F, np.inf) < NEWTON_TOL:
                break

            J = A_diff + dt * (
                sp.diags(np.minimum(dpx, 0) / mobility) @ D_plus_x  +
                sp.diags(np.maximum(dmx, 0) / mobility) @ D_minus_x +
                sp.diags(np.minimum(dpy, 0) / mobility) @ D_plus_y  +
                sp.diags(np.maximum(dmy, 0) / mobility) @ D_minus_y
            )
            u_curr -= direct_solve(J, F)
        U[n] = u_curr
    return U, total_newton

def solve_fp_forward(U_flow, M_old_flow):
    M    = np.zeros((Nt + 1, num_x))
    M[0] = m0
    for n in range(Nt):
        m_conv   = compute_m_conv(M_old_flow[n+1])
        dpx, dmx = D_plus_x @ U_flow[n], D_minus_x @ U_flow[n]
        dpy, dmy = D_plus_y @ U_flow[n], D_minus_y @ U_flow[n]
        mobility = (1.0 + 4.0 * m_conv)**beta
        J_H = (sp.diags(np.minimum(dpx, 0) / mobility) @ D_plus_x +
               sp.diags(np.maximum(dmx, 0) / mobility) @ D_minus_x +
               sp.diags(np.minimum(dpy, 0) / mobility) @ D_plus_y +
               sp.diags(np.maximum(dmy, 0) / mobility) @ D_minus_y).tocsr()
        A        = (I - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
        M[n+1]   = direct_solve(A, M[n])
    return M

# ==========================================
# SECTION 5: DAMPED PICARD ITERATION
# ==========================================

start_time_all = time.perf_counter()

with open(history_filename, "w", buffering=1) as f_log:
    header = (
        f"{'='*118}\n"
        f"   NONLOCAL CONGESTION MFG EXAMPLE 2D (Picard)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
        f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
        f"  nu   = {nu}, beta = {beta}, zeta = {zeta}, sigma_kernel = {sigma_kernel}\n"
        f"Solver Parameters:\n"
        f"  PICARD_MAX_ITER = {PICARD_MAX_ITER}, PICARD_TOL = {PICARD_TOL}\n"
        f"  DAMPING = {DAMPING}\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  LINEAR_SOLVER = direct sparse (spsolve/SuperLU)\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  initial_support = [0.375, 0.625]^2, initial_height = 4.0, boundary = periodic\n"
        f"  terminal_wells = [(0.3, 0.5), (0.7, 0.5)], terminal_scale = 15.0, congestion_scale = 4.0\n"
        f"  kernel = periodic Gaussian, sigma_kernel = {sigma_kernel}\n"
        f"{'-'*118}\n"
        f"{'Iter':<5} | {'Abs Err U':<12} | {'Rel Err U':<12} | {'Abs Err M':<12} | {'Rel Err M':<12} | {'Newton It':<10} | {'Time (s)':<10}\n"
        f"{'-'*118}\n"
    )
    print(header, end='')
    f_log.write(header)
    M_flow      = np.tile(m0, (Nt+1, 1))
    A_diff_only = (I - dt * nu * Laplacian).tocsr()
    for n in range(Nt):
        M_flow[n+1] = direct_solve(A_diff_only, M_flow[n])

    U_flow = np.zeros((Nt + 1, num_x))

    for k in range(1, PICARD_MAX_ITER + 1):
        t0 = time.perf_counter()
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

        iter_time = time.perf_counter() - t0
        log_str = (f"{k:<5} | {abs_err_u:.4e}   | {rel_err_u:.4e}   | "
                   f"{abs_err_m:.4e}   | {rel_err_m:.4e}   | {n_iters:<10} | {iter_time:.4f}")
        print(log_str)
        f_log.write(log_str + "\n")
        U_flow, M_flow = U_flow_new, M_flow_new
        if rel_err_u < PICARD_TOL and rel_err_m < PICARD_TOL:
            conv_msg = f"{'-'*118}\nCONVERGED in {k} iterations.\n"
            print(conv_msg)
            f_log.write(conv_msg)
            break
    else:
        fail_msg = (f"{'-'*118}\n"
                    f"FAILED TO CONVERGE WITHIN PICARD_MAX_ITER = {PICARD_MAX_ITER}.\n")
        print(fail_msg)
        f_log.write(fail_msg)


    total_time = time.perf_counter() - start_time_all
    time_msg = f"Total Execution Time: {total_time:.4f} seconds.\n"
    print(time_msg)
    f_log.write(time_msg)
# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
time_labels  = [0.0, T/4, T/2, 3*T/4, T]

fig_2d, axes_2d = plt.subplots(2, 5, figsize=(25, 9))
for i, (idx, t_val) in enumerate(zip(time_indices, time_labels)):
    M_snap = M_flow[idx].reshape((Nx, Ny))
    U_snap = U_flow[idx].reshape((Nx, Ny))
    im_m = axes_2d[0, i].contourf(X, Y, M_snap, levels=30, cmap='viridis')
    axes_2d[0, i].set_title(f'Density $(m)$ at t={t_val:.2f}')
    axes_2d[0, i].set_xlabel('x'); axes_2d[0, i].set_ylabel('y')
    fig_2d.colorbar(im_m, ax=axes_2d[0, i])
    im_u = axes_2d[1, i].contourf(X, Y, U_snap, levels=30, cmap='plasma')
    axes_2d[1, i].set_title(f'Value $(u)$ at t={t_val:.2f}')
    axes_2d[1, i].set_xlabel('x'); axes_2d[1, i].set_ylabel('y')
    fig_2d.colorbar(im_u, ax=axes_2d[1, i])
plt.tight_layout()
plt.savefig(plot_contour_filename)
plt.close(fig_2d)
print(f"Saved: {plot_contour_filename}")

fig_3d = plt.figure(figsize=(25, 10))
for i, (idx, t_val) in enumerate(zip(time_indices, time_labels)):
    M_snap = M_flow[idx].reshape((Nx, Ny))
    U_snap = U_flow[idx].reshape((Nx, Ny))
    ax_m = fig_3d.add_subplot(2, 5, i + 1, projection='3d')
    surf_m = ax_m.plot_surface(X, Y, M_snap, cmap='viridis', edgecolor='none', rstride=2, cstride=2, alpha=0.9)
    ax_m.set_title(f'Density $(m)$ at t={t_val:.2f}')
    ax_m.set_xlabel('x'); ax_m.set_ylabel('y'); ax_m.set_zlabel('m')
    ax_m.view_init(elev=30, azim=45)
    fig_3d.colorbar(surf_m, ax=ax_m, shrink=0.5, aspect=5)
    ax_u = fig_3d.add_subplot(2, 5, i + 6, projection='3d')
    surf_u = ax_u.plot_surface(X, Y, U_snap, cmap='plasma', edgecolor='none', rstride=2, cstride=2, alpha=0.9)
    ax_u.set_title(f'Value $(u)$ at t={t_val:.2f}')
    ax_u.set_xlabel('x'); ax_u.set_ylabel('y'); ax_u.set_zlabel('u')
    ax_u.view_init(elev=30, azim=45)
    fig_3d.colorbar(surf_u, ax=ax_u, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig(plot_surface_filename)
plt.close(fig_3d)
print(f"Saved: {plot_surface_filename}")
