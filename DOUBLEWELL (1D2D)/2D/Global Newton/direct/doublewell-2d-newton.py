"""Solve the 2D double-well MFG with Global Newton and viscosity continuation."""

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
y_confinement = 0.5      # Coefficient of y^2 in the potential

# --- Spatial and Temporal Grid ---
L = 2.5
xmin, xmax = -L, L       # Spatial domain boundaries
ymin, ymax = -L, L       # Spatial domain boundaries
T = 1.0                  # Total time horizon
Nx, Ny = 20, 20          # Number of cell-centered spatial grid points
Nt = 20                  # Number of temporal intervals

# --- Newton Iteration Parameters ---
NEWTON_MAX_ITER = 20     # Maximum Global Newton iterations per viscosity level
NEWTON_TOL = 1e-9        # Convergence tolerance for the monolithic residual

# --- Continuation Parameters ---
NU_START = 0.5           # Starting viscosity
NU_TARGET = 0.1          # Target viscosity
TAU = 0.75              # Reduction factor for viscosity step
GAMMA = 0.5              # Backtracking factor
CONT_TOL = 1e-4          # Minimum allowable step size

# --- Derived Quantities ---
L_x = xmax - xmin        # Length of the spatial domain (x)
L_y = ymax - ymin        # Length of the spatial domain (y)
Dx  = L_x / Nx           # Spatial step size (x)
Dy  = L_y / Ny           # Spatial step size (y)
dt  = T / Nt             # Temporal step size
x   = xmin + np.arange(Nx) * Dx  # Grid nodes on the periodic domain [xmin, xmax)
y   = ymin + np.arange(Ny) * Dy  # Grid nodes on the periodic domain [ymin, ymax)
xx, yy = np.meshgrid(x, y)
num_x = Nx * Ny

# --- Saved File Names ---
history_filename = "doublewell-2d-newton-history.txt"
plot_contour_filename = "doublewell-2d-newton-contour.png"
plot_surface_filename = "doublewell-2d-newton-surface.png"

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

I_1Dx = sp.eye(Nx)
I_1Dy = sp.eye(Ny)
I     = sp.eye(num_x)

# Forward difference with periodic wrap-around in x.
data_px = np.array([-np.ones(Nx), np.ones(Nx)])
D_plus_1Dx = sp.spdiags(data_px, [0, 1], Nx, Nx).tolil()
D_plus_1Dx[Nx-1, Nx-1] = -1.0
D_plus_1Dx[Nx-1, 0] = 1.0
D_plus_1Dx = D_plus_1Dx.tocsr() / Dx

# Backward difference with periodic wrap-around in x.
data_mx = np.array([-np.ones(Nx), np.ones(Nx)])
D_minus_1Dx = sp.spdiags(data_mx, [-1, 0], Nx, Nx).tolil()
D_minus_1Dx[0, 0] = 1.0
D_minus_1Dx[0, Nx-1] = -1.0
D_minus_1Dx = D_minus_1Dx.tocsr() / Dx

# Corresponding periodic differences in y.
data_py = np.array([-np.ones(Ny), np.ones(Ny)])
D_plus_1Dy = sp.spdiags(data_py, [0, 1], Ny, Ny).tolil()
D_plus_1Dy[Ny-1, Ny-1] = -1.0
D_plus_1Dy[Ny-1, 0] = 1.0
D_plus_1Dy = D_plus_1Dy.tocsr() / Dy

data_my = np.array([-np.ones(Ny), np.ones(Ny)])
D_minus_1Dy = sp.spdiags(data_my, [-1, 0], Ny, Ny).tolil()
D_minus_1Dy[0, 0] = 1.0
D_minus_1Dy[0, Ny-1] = -1.0
D_minus_1Dy = D_minus_1Dy.tocsr() / Dy

# 2D Kronecker Lift
D_plus_x  = sp.kron(I_1Dy, D_plus_1Dx ).tocsr()
D_minus_x = sp.kron(I_1Dy, D_minus_1Dx).tocsr()
D_plus_y  = sp.kron(D_plus_1Dy,  I_1Dx).tocsr()
D_minus_y = sp.kron(D_minus_1Dy, I_1Dx).tocsr()

# Periodic five-point Laplacian.  This symmetric negative semidefinite
# operator has zero row and column sums, so its FP contribution preserves
# total mass exactly.
Laplacian = -(D_plus_x.transpose() @ D_plus_x +
              D_plus_y.transpose() @ D_plus_y).tocsr()

# ==========================================
# SECTION 4: GLOBAL NEWTON SOLVER
# ==========================================

def global_newton_step(nu_val, W_guess, log_file=None):
    """Apply monolithic Global Newton at one fixed viscosity value."""
    if W_guess is None:
        U = np.tile(uT_flat, (Nt + 1, 1))
        M = np.tile(m0_flat, (Nt + 1, 1))
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
        F_U[Nt] = U[Nt] - uT_flat
        J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        # Initial condition for M
        J_MM[0][0] = I.tocsr()
        F_M[0] = M[0] - m0_flat
        J_MU[0][0] = J_MU[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        for n in range(Nt):
            un, un_next = U[n], U[n+1]
            mn, mn_next = M[n], M[n+1]
            dpx, dmx = D_plus_x @ un, D_minus_x @ un
            dpy, dmy = D_plus_y @ un, D_minus_y @ un
            p_min_x, p_max_x = np.minimum(dpx, 0), np.maximum(dmx, 0)
            p_min_y, p_max_y = np.minimum(dpy, 0), np.maximum(dmy, 0)
            f_val = compute_f(mn_next)

            # Jacobian of the monotone numerical Hamiltonian.  Its transpose
            # is used *verbatim* in the FP equation, ensuring the exact
            # discrete HJB--FP adjoint relation on the periodic domain.
            J_H = (sp.diags(p_min_x) @ D_plus_x + sp.diags(p_max_x) @ D_minus_x +
                   sp.diags(p_min_y) @ D_plus_y + sp.diags(p_max_y) @ D_minus_y).tocsr()

            # --- Backward HJB Equation ---
            F_U[n] = (un - un_next) / dt - nu_val * Laplacian @ un + discrete_Hamiltonian(dpx, dmx, dpy, dmy, f_val)

            dH_dm = -coupling_str * alpha * (np.maximum(mn_next, 0)**(alpha-1))
            J_UU[n][n]   = ((1.0/dt) * I - nu_val * Laplacian + J_H).tocsr()
            J_UU[n][n+1] = (-(1.0/dt) * I).tocsr()
            J_UM[n][n]   = sp.csr_matrix((num_x, num_x))
            J_UM[n][n+1] = sp.diags(dH_dm).tocsr()

            # --- Forward FP Equation ---
            F_M[n+1] = ((mn_next - mn) / dt - nu_val * Laplacian @ mn_next +
                        J_H.transpose() @ mn_next)

            J_MM[n+1][n+1] = ((1.0/dt) * I - nu_val * Laplacian +
                              J_H.transpose()).tocsr()
            J_MM[n+1][n]   = (-(1.0/dt) * I).tocsr()
            J_MU[n+1][n]   = (D_plus_x.transpose() @ sp.diags(mn_next * (dpx < 0)) @ D_plus_x +
                               D_minus_x.transpose() @ sp.diags(mn_next * (dmx > 0)) @ D_minus_x +
                               D_plus_y.transpose() @ sp.diags(mn_next * (dpy < 0)) @ D_plus_y +
                               D_minus_y.transpose() @ sp.diags(mn_next * (dmy > 0)) @ D_minus_y).tocsr()

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
        f"   DOUBLE WELL MFG EXAMPLE 2D (Global Newton)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
        f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
        f"  nu   = {NU_TARGET}, alpha = {alpha}, kappa = {coupling_str}\n"
        f"Solver Parameters:\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  Continuation: NU_START={NU_START}, TAU={TAU}, GAMMA={GAMMA}, CONT_TOL={CONT_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  tilt = {tilt}, well_depth = {well_depth}, y_confinement = {y_confinement}\n"
        f"  initial_std = {initial_std}, boundary = periodic\n"
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
plt.close(fig_3d)
print(f"Saved: {plot_surface_filename}")
