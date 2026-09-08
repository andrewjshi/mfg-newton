"""Solve the 2D double-well MFG with the Picard method."""

import resource
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
nu = 0.1                 # Viscosity coefficient
coupling_str = 1.0       # kappa: Power-law coupling strength
tilt = 0.3               # Linear tilt parameter for the double-well potential asymmetry
well_depth = 2.0         # Depth of the wells in the V(x) potential
alpha = 2                # Power-law exponent: f(m) = kappa * m^alpha
initial_std = 0.6        # Standard deviation of the initial Gaussian density
y_confinement = 0.5      # Coefficient of y^2 in the potential

# --- Spatial and Temporal Grid ---
L = 2.5
xmin, xmax = -L, L       # Spatial domain boundaries
ymin, ymax = -L, L       # Spatial domain boundaries
T = 1.0                  # Total time horizon
Nx, Ny = 200, 200        # Number of spatial intervals
Nt = 20                   # Number of temporal intervals

# --- Solver Selection ---
# GUIDANCE ON SOLVER CHOICE:
# 1. 'direct' (spsolve):
#    - Best for 1D problems or small 2D grids (up to ~100x100).
#    - Robust; handles ill-conditioned matrices without needing preconditioning.
#    - Memory consumption scales poorly (O(N^2)); will crash on very large 2D grids.
# 2. 'iterative' (bicgstab):
#    - Essential for high-resolution 2D grids (200x200+) and 3D problems.
#    - Memory efficient (O(N)).
#    - Requires good preconditioning (ILU) to avoid stalling or diverging.
LINEAR_SOLVER = 'direct'

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 50     # Maximum number of outer HJB-FP coupling iterations
PICARD_TOL = 1e-6        # Convergence tolerance for the outer Picard loop
DAMPING = 0.8            # Damping parameter (0 < DAMPING <= 1). 1 corresponds to no damping.

NEWTON_MAX_ITER = 20     # Maximum Newton iterations for the backward HJB solves
NEWTON_TOL = 1e-9        # Convergence tolerance for the backward Newton solver

# --- Derived Quantities ---
dt = T / Nt              # Temporal step size
Dx = (xmax - xmin) / Nx  # Spatial cell width (x)
Dy = (ymax - ymin) / Ny  # Spatial cell width (y)
# Periodic grid nodes on [xmin, xmax) x [ymin, ymax); both equations use
# periodic boundary conditions.
x  = xmin + np.arange(Nx) * Dx
y  = ymin + np.arange(Ny) * Dy
xx, yy = np.meshgrid(x, y)
num_x = Nx * Ny
norm_const = np.sqrt(Dx * Dy * dt)

# --- Saved File Names ---
history_filename  = "doublewell-2d-picard-history.txt"
plot_contour_filename = "doublewell-2d-picard-contour.png"
plot_surface_filename = "doublewell-2d-picard-surface.png"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

V_2d    = well_depth * (xx**2 - 1.0)**2 + tilt * xx + y_confinement * yy**2
V_flat  = V_2d.flatten()
m0_2d   = np.exp(-(xx**2 + yy**2) / (2 * initial_std**2))
m0_flat = m0_2d.flatten()
m0_flat /= (np.sum(m0_flat) * Dx * Dy)
uT_flat = V_flat.copy()

def compute_f(m_dist):
    return V_flat + coupling_str * (np.maximum(m_dist, 0)**alpha)

def discrete_Hamiltonian(dpx, dmx, dpy, dmy, f_val):
    p_min_x = np.minimum(dpx, 0)
    p_max_x = np.maximum(dmx, 0)
    p_min_y = np.minimum(dpy, 0)
    p_max_y = np.maximum(dmy, 0)
    return 0.5 * (p_min_x**2 + p_max_x**2 + p_min_y**2 + p_max_y**2) - f_val

def dH_dp(p):
    return p

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I_x  = sp.eye(Nx)
I_y  = sp.eye(Ny)
I    = sp.eye(num_x)

# One-sided x differences with periodic wrap-around.
data_p_x = np.array([-np.ones(Nx), np.ones(Nx)])
Dp_x_1d = sp.spdiags(data_p_x, [0, 1], Nx, Nx).tolil()
Dp_x_1d[-1, -1] = -1.0
Dp_x_1d[-1, 0] = 1.0
Dp_x_1d = Dp_x_1d.tocsr() / Dx

# Backward x difference with periodic wrap-around.
Dm_x_1d = sp.spdiags(data_p_x, [-1, 0], Nx, Nx).tolil()
Dm_x_1d[0, 0] = 1.0
Dm_x_1d[0, -1] = -1.0
Dm_x_1d = Dm_x_1d.tocsr() / Dx

D_plus_x  = sp.kron(I_y, Dp_x_1d)
D_minus_x = sp.kron(I_y, Dm_x_1d)

# One-sided y differences with periodic wrap-around.
data_p_y = np.array([-np.ones(Ny), np.ones(Ny)])
Dp_y_1d = sp.spdiags(data_p_y, [0, 1], Ny, Ny).tolil()
Dp_y_1d[-1, -1] = -1.0
Dp_y_1d[-1, 0] = 1.0
Dp_y_1d = Dp_y_1d.tocsr() / Dy

# Backward y difference with periodic wrap-around.
Dm_y_1d = sp.spdiags(data_p_y, [-1, 0], Ny, Ny).tolil()
Dm_y_1d[0, 0] = 1.0
Dm_y_1d[0, -1] = -1.0
Dm_y_1d = Dm_y_1d.tocsr() / Dy

D_plus_y  = sp.kron(Dp_y_1d, I_x)
D_minus_y = sp.kron(Dm_y_1d, I_x)

# Periodic five-point Laplacian.  Its row and column sums vanish, so the same
# operator gives HJB diffusion and conservative FP diffusion.  Together with
# J_H.T below, this makes the full FP spatial operator the exact discrete
# adjoint of the linearized HJB spatial operator.
Laplacian = -(D_plus_x.transpose() @ D_plus_x +
              D_plus_y.transpose() @ D_plus_y).tocsr()

# ==========================================
# SECTION 4: HJB AND FP SOLVERS
# ==========================================

A_diff = (I - dt * nu * Laplacian).tocsr()

def solve_hjb_backward(M_flow, is_first_picard=False):
    U = np.zeros((Nt + 1, num_x))
    U[Nt] = uT_flat
    total_newton = 0

    for n in range(Nt - 1, -1, -1):
        u_next = U[n+1]
        u_curr = u_next.copy()
        f_val  = compute_f(M_flow[n+1])

        for iter_n in range(NEWTON_MAX_ITER):
            total_newton += 1
            dpx, dmx = D_plus_x @ u_curr, D_minus_x @ u_curr
            dpy, dmy = D_plus_y @ u_curr, D_minus_y @ u_curr
            F = A_diff @ u_curr + dt * discrete_Hamiltonian(dpx, dmx, dpy, dmy, f_val) - u_next

            J = A_diff + dt * (
                sp.diags(dH_dp(np.minimum(dpx, 0))) @ D_plus_x +
                sp.diags(dH_dp(np.maximum(dmx, 0))) @ D_minus_x +
                sp.diags(dH_dp(np.minimum(dpy, 0))) @ D_plus_y +
                sp.diags(dH_dp(np.maximum(dmy, 0))) @ D_minus_y
            )

            # --- Numerical Linear Algebra Study (Diagnostic Block) ---
            if is_first_picard and n == Nt - 1 and iter_n == 0:
                print(f"\n--- NUMERICAL LINEAR ALGEBRA STUDY (Nx={Nx}) ---")
                nnz_A = J.nnz
                t0_diag = time.time()
                LU = spla.splu(J.tocsc())
                diag_solve_time = time.time() - t0_diag
                nnz_LU = LU.L.nnz + LU.U.nnz
                raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_ram_gb = raw_rss / (1024**3) if sys.platform == 'darwin' else raw_rss / (1024**2)
                print(f"Nh^2 (DOFs):      {num_x}")
                print(f"Bandwidth (Nx):   {Nx}")
                print(f"Factor NNZ:       {nnz_LU}")
                print(f"Fill-in Ratio:    {nnz_LU / nnz_A:.2f}x")
                print(f"Solve Time (s):   {diag_solve_time:.6f}")
                print(f"Peak RAM (GB):    {peak_ram_gb:.4f}")
                print("-----------------------------------------------\n")

            if np.linalg.norm(F, np.inf) < NEWTON_TOL:
                break

            if LINEAR_SOLVER == 'direct':
                du = spla.spsolve(J, F)
            else:
                ilu   = spla.spilu(J.tocsc(), drop_tol=1e-3, fill_factor=5)
                M_pre = spla.LinearOperator(J.shape, matvec=ilu.solve)
                du, info = spla.bicgstab(J, F, M=M_pre, tol=1e-8, maxiter=200)
                if info != 0:
                    raise RuntimeError(f"BiCGSTAB failed to converge (info={info})")
            u_curr -= du
        U[n] = u_curr
    return U, total_newton

def solve_fp_forward(U_flow):
    M    = np.zeros((Nt + 1, num_x))
    M[0] = m0_flat
    for n in range(Nt):
        dpx, dmx = D_plus_x @ U_flow[n], D_minus_x @ U_flow[n]
        dpy, dmy = D_plus_y @ U_flow[n], D_minus_y @ U_flow[n]
        J_H = (sp.diags(dH_dp(np.minimum(dpx, 0))) @ D_plus_x +
               sp.diags(dH_dp(np.maximum(dmx, 0))) @ D_minus_x +
               sp.diags(dH_dp(np.minimum(dpy, 0))) @ D_plus_y +
               sp.diags(dH_dp(np.maximum(dmy, 0))) @ D_minus_y).tocsr()
        A = (I - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
        if LINEAR_SOLVER == 'direct':
            M[n+1] = spla.spsolve(A, M[n])
        else:
            ilu   = spla.spilu(A.tocsc(), drop_tol=1e-3, fill_factor=5)
            M_pre = spla.LinearOperator(A.shape, matvec=ilu.solve)
            M[n+1], _ = spla.bicgstab(A, M[n], M=M_pre, tol=1e-8, maxiter=200)
    return M

# ==========================================
# SECTION 5: DAMPED PICARD ITERATION
# ==========================================

start_time_all = time.time()

with open(history_filename, "w", buffering=1) as f_log:
    header = (
        f"{'='*118}\n"
        f"   DOUBLE WELL MFG EXAMPLE 2D (Picard)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
        f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
        f"  nu   = {nu}, alpha = {alpha}, kappa = {coupling_str}\n"
        f"Solver Parameters:\n"
        f"  PICARD_MAX_ITER = {PICARD_MAX_ITER}, PICARD_TOL = {PICARD_TOL}\n"
        f"  DAMPING = {DAMPING}, LINEAR_SOLVER = {LINEAR_SOLVER}\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  tilt = {tilt}, well_depth = {well_depth}, y_confinement = {y_confinement}\n"
        f"  initial_std = {initial_std}, boundary = periodic\n"
        f"{'-'*118}\n"
        f"{'Iter':<5} | {'Abs Err U':<12} | {'Rel Err U':<12} | {'Abs Err M':<12} | {'Rel Err M':<12} | {'Newton It':<10} | {'Time (s)':<10}\n"
        f"{'-'*118}\n"
    )
    print(header, end=''); f_log.write(header)

    M_flow = np.tile(m0_flat, (Nt+1, 1))
    U_flow = np.zeros((Nt + 1, num_x))

    for k in range(1, PICARD_MAX_ITER + 1):
        t0 = time.time()
        U_candidate, n_iters = solve_hjb_backward(M_flow, is_first_picard=(k==1))
        U_flow_new  = DAMPING * U_candidate + (1 - DAMPING) * U_flow
        M_candidate = solve_fp_forward(U_flow_new)
        M_flow_new  = DAMPING * M_candidate + (1 - DAMPING) * M_flow

        delta_u_norm = np.linalg.norm((U_flow_new - U_flow).ravel())
        delta_m_norm = np.linalg.norm((M_flow_new - M_flow).ravel())
        previous_u_norm = np.linalg.norm(U_flow.ravel())
        previous_m_norm = np.linalg.norm(M_flow.ravel())
        abs_err_u = delta_u_norm * norm_const
        rel_err_u = delta_u_norm / previous_u_norm if previous_u_norm > 1e-12 else (0.0 if delta_u_norm <= 1e-12 else 1.0)
        abs_err_m = delta_m_norm * norm_const
        rel_err_m = delta_m_norm / previous_m_norm if previous_m_norm > 1e-12 else (0.0 if delta_m_norm <= 1e-12 else 1.0)

        iter_time = time.time() - t0
        log_str = f"{k:<5} | {abs_err_u:.4e}   | {rel_err_u:.4e}   | {abs_err_m:.4e}   | {rel_err_m:.4e}   | {n_iters:<10} | {iter_time:.4f}"
        print(log_str); f_log.write(log_str + "\n")

        U_flow, M_flow = U_flow_new, M_flow_new
        if rel_err_u < PICARD_TOL and rel_err_m < PICARD_TOL:
            conv_msg = f"{'-'*118}\nCONVERGED at nu={nu} in {k} iterations.\n"
            print(conv_msg); f_log.write(conv_msg)
            break

    total_time = time.time() - start_time_all
    time_msg = f"Total Execution Time: {total_time:.4f} seconds.\n"
    print(time_msg); f_log.write(time_msg)

# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
time_labels  = [0.0, T/4, T/2, 3*T/4, T]

fig_2d, axes_2d = plt.subplots(2, 5, figsize=(25, 9))
for i, (idx, t_val) in enumerate(zip(time_indices, time_labels)):
    M_snap = M_flow[idx].reshape((Ny, Nx))
    U_snap = U_flow[idx].reshape((Ny, Nx))
    im_m = axes_2d[0, i].contourf(xx, yy, M_snap, levels=30, cmap='viridis')
    axes_2d[0, i].set_title(f'Density $(m)$ at t={t_val:.2f}')
    axes_2d[0, i].set_xlabel('x'); axes_2d[0, i].set_ylabel('y')
    fig_2d.colorbar(im_m, ax=axes_2d[0, i])
    im_u = axes_2d[1, i].contourf(xx, yy, U_snap, levels=30, cmap='plasma')
    axes_2d[1, i].set_title(f'Value $(u)$ at t={t_val:.2f}')
    axes_2d[1, i].set_xlabel('x'); axes_2d[1, i].set_ylabel('y')
    fig_2d.colorbar(im_u, ax=axes_2d[1, i])
plt.tight_layout()
plt.savefig(plot_contour_filename, dpi=150)
plt.close(fig_2d)
print(f"Saved: {plot_contour_filename}")

fig_3d = plt.figure(figsize=(25, 10))
for i, (idx, t_val) in enumerate(zip(time_indices, time_labels)):
    M_snap = M_flow[idx].reshape((Ny, Nx))
    U_snap = U_flow[idx].reshape((Ny, Nx))
    ax_m = fig_3d.add_subplot(2, 5, i + 1, projection='3d')
    surf_m = ax_m.plot_surface(xx, yy, M_snap, cmap='viridis', edgecolor='none', rstride=2, cstride=2, alpha=0.9)
    ax_m.set_title(f'Density $(m)$ at t={t_val:.2f}')
    ax_m.set_xlabel('x'); ax_m.set_ylabel('y'); ax_m.set_zlabel('m')
    ax_m.view_init(elev=30, azim=45)
    fig_3d.colorbar(surf_m, ax=ax_m, shrink=0.5, aspect=5)
    ax_u = fig_3d.add_subplot(2, 5, i + 6, projection='3d')
    surf_u = ax_u.plot_surface(xx, yy, U_snap, cmap='plasma', edgecolor='none', rstride=2, cstride=2, alpha=0.9)
    ax_u.set_title(f'Value $(u)$ at t={t_val:.2f}')
    ax_u.set_xlabel('x'); ax_u.set_ylabel('y'); ax_u.set_zlabel('u')
    ax_u.view_init(elev=30, azim=45)
    fig_3d.colorbar(surf_u, ax=ax_u, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig(plot_surface_filename, dpi=150)
plt.show()
print(f"Saved: {plot_surface_filename}")
