"""Solve the 2D local-congestion MFG with Global Newton and viscosity continuation."""

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
beta = 1.5               # Congestion exponent
zeta = 1.0               # Coupling strength

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0    # Spatial domain boundaries
ymin, ymax = 0.0, 1.0    # Spatial domain boundaries
T = 1.0                  # Total time horizon
Nx, Ny = 20, 20          # Number of spatial intervals
Nt = 20                  # Number of temporal intervals

# --- Newton Iteration Parameters ---
NEWTON_MAX_ITER = 20     # Maximum Global Newton iterations per viscosity level
NEWTON_TOL = 1e-9        # Convergence tolerance for the monolithic residual

# --- Continuation Parameters ---
NU_START = 0.2           # Starting viscosity
NU_TARGET = 0.01         # Target viscosity
TAU = 0.75              # Reduction factor for viscosity step
GAMMA = 0.5              # Backtracking factor
CONT_TOL = 1e-4          # Minimum allowable step size

# --- Derived Quantities ---
Dx    = (xmax - xmin) / Nx # Spatial step size (x)
Dy    = (ymax - ymin) / Ny # Spatial step size (y)
dt    = T / Nt             # Temporal step size
x     = np.linspace(xmin, xmax, Nx, endpoint=False) # Computational grid points (Periodic)
y     = np.linspace(ymin, ymax, Ny, endpoint=False) # Computational grid points (Periodic)
X, Y  = np.meshgrid(x, y, indexing='ij')
num_x = Nx * Ny

# --- Saved File Names ---
history_filename      = "congestionlocal-2d-newton-history.txt"
plot_contour_filename = "congestionlocal-2d-newton-contour.png"
plot_surface_filename = "congestionlocal-2d-newton-surface.png"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# Initial condition m0: Square blob on [0.375, 0.625]^2
m0_2d = np.zeros((Nx, Ny))
mask  = (X >= 0.375) & (X <= 0.625) & (Y >= 0.375) & (Y <= 0.625)
m0_2d[mask] = 4.0
m0_2d = m0_2d / (np.sum(m0_2d) * Dx * Dy)
m0    = m0_2d.flatten()

# Terminal cost: double quadratic well at (0.3, 0.5) and (0.7, 0.5)
uT_2d = 15.0 * np.minimum((X - 0.3)**2 + (Y - 0.5)**2,
                           (X - 0.7)**2 + (Y - 0.5)**2)
uT = uT_2d.flatten()

def compute_f(m_dist):
    """Coupling/running cost f(x, m) = zeta * m."""
    return zeta * m_dist

def discrete_Hamiltonian(dpx, dmx, dpy, dmy, m_curr, f_val):
    """H = (|p_x|^2 + |p_y|^2) / (2*(1+4m)^beta) - zeta*m"""
    m_safe     = np.maximum(m_curr, 1e-10)
    cong_coeff = 1.0 / (2.0 * (1.0 + 4.0 * m_safe)**beta)
    px_sq      = np.minimum(dpx, 0)**2 + np.maximum(dmx, 0)**2
    py_sq      = np.minimum(dpy, 0)**2 + np.maximum(dmy, 0)**2
    return cong_coeff * (px_sq + py_sq) - f_val

def dH_dp(p, m_curr):
    """D_p H = p / (1+4m)^beta"""
    m_safe     = np.maximum(m_curr, 1e-10)
    cong_coeff = 1.0 / (2.0 * (1.0 + 4.0 * m_safe)**beta)
    return 2.0 * cong_coeff * p

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I_1Dx = sp.eye(Nx)
I_1Dy = sp.eye(Ny)
I     = sp.eye(num_x)

# Forward Difference Matrix (D+x) - Periodic
data_px = np.array([-np.ones(Nx), np.ones(Nx)])
D_plus_1Dx = sp.spdiags(data_px, [0, 1], Nx, Nx).tolil()
D_plus_1Dx[Nx-1, Nx-1] = -1.0
D_plus_1Dx[Nx-1, 0]    =  1.0
D_plus_1Dx = D_plus_1Dx.tocsr() / Dx

# Backward Difference Matrix (D-x) - Periodic
data_mx = np.array([-np.ones(Nx), np.ones(Nx)])
D_minus_1Dx = sp.spdiags(data_mx, [-1, 0], Nx, Nx).tolil()
D_minus_1Dx[0, 0]    =  1.0
D_minus_1Dx[0, Nx-1] = -1.0
D_minus_1Dx = D_minus_1Dx.tocsr() / Dx

# Forward Difference Matrix (D+y) - Periodic
data_py = np.array([-np.ones(Ny), np.ones(Ny)])
D_plus_1Dy = sp.spdiags(data_py, [0, 1], Ny, Ny).tolil()
D_plus_1Dy[Ny-1, Ny-1] = -1.0
D_plus_1Dy[Ny-1, 0]    =  1.0
D_plus_1Dy = D_plus_1Dy.tocsr() / Dy

# Backward Difference Matrix (D-y) - Periodic
data_my = np.array([-np.ones(Ny), np.ones(Ny)])
D_minus_1Dy = sp.spdiags(data_my, [-1, 0], Ny, Ny).tolil()
D_minus_1Dy[0, 0]    =  1.0
D_minus_1Dy[0, Ny-1] = -1.0
D_minus_1Dy = D_minus_1Dy.tocsr() / Dy

# 2D Kronecker Lift
D_plus_x  = sp.kron(D_plus_1Dx,  I_1Dy).tocsr()
D_minus_x = sp.kron(D_minus_1Dx, I_1Dy).tocsr()
D_plus_y  = sp.kron(I_1Dx, D_plus_1Dy ).tocsr()
D_minus_y = sp.kron(I_1Dx, D_minus_1Dy).tocsr()

# Laplacian Matrix
Laplacian = D_plus_x @ D_minus_x + D_plus_y @ D_minus_y

# ==========================================
# SECTION 4: GLOBAL NEWTON SOLVER
# ==========================================

def global_newton_step(nu_val, W_guess, log_file=None):
    """Apply monolithic Global Newton at one fixed viscosity value."""
    if W_guess is None:
        U = np.tile(uT, (Nt + 1, 1))
        # Initialise M by diffusing m0 forward
        M    = np.zeros((Nt + 1, num_x))
        M[0] = m0.copy()
        A_init = (I - dt * nu_val * Laplacian).tocsr()
        for n in range(Nt):
            M[n+1] = spla.spsolve(A_init, M[n])
    else:
        U = W_guess[:(Nt+1)*num_x].reshape((Nt+1, num_x))
        M = W_guess[(Nt+1)*num_x:].reshape((Nt+1, num_x))

    I_neg = -I.tocsr()

    for it in range(NEWTON_MAX_ITER):
        J_UU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_UM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]

        F_U = np.zeros((Nt+1, num_x))
        F_M = np.zeros((Nt+1, num_x))
        A_diff = I - dt * nu_val * Laplacian

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
            dpx, dmx = D_plus_x @ un, D_minus_x @ un
            dpy, dmy = D_plus_y @ un, D_minus_y @ un
            p_min_x, p_max_x = np.minimum(dpx, 0), np.maximum(dmx, 0)
            p_min_y, p_max_y = np.minimum(dpy, 0), np.maximum(dmy, 0)
            f_val = compute_f(mn_next)

            # --- Backward HJB Equation ---
            F_U[n] = A_diff @ un + dt * discrete_Hamiltonian(dpx, dmx, dpy, dmy, mn_next, f_val) - un_next

            m_safe     = np.maximum(mn_next, 1e-10)
            cong_coeff = 1.0 / (2.0 * (1.0 + 4.0 * m_safe)**beta)
            dH_dM      = 0.5 * (p_min_x**2 + p_max_x**2 + p_min_y**2 + p_max_y**2) * (-4.0 * beta / ((1.0 + 4.0 * m_safe)**(beta+1))) - zeta

            J_UU[n][n]   = (A_diff +
                            dt * sp.diags(dH_dp(p_min_x, mn_next)) @ D_plus_x +
                            dt * sp.diags(dH_dp(p_max_x, mn_next)) @ D_minus_x +
                            dt * sp.diags(dH_dp(p_min_y, mn_next)) @ D_plus_y +
                            dt * sp.diags(dH_dp(p_max_y, mn_next)) @ D_minus_y).tocsr()
            J_UU[n][n+1] = I_neg
            J_UM[n][n]   = sp.csr_matrix((num_x, num_x))
            J_UM[n][n+1] = (dt * sp.diags(dH_dM)).tocsr()

            # --- Forward FP Equation ---
            mobility = (1.0 + 4.0 * m_safe)**beta
            phi = mn_next / mobility
            trans = (D_minus_x @ (phi * p_min_x) + D_plus_x @ (phi * p_max_x) +
                     D_minus_y @ (phi * p_min_y) + D_plus_y @ (phi * p_max_y))
            F_M[n+1] = A_diff @ mn_next - mn - dt * trans

            dphi = (1.0 / mobility -
                    4.0 * beta * mn_next / (1.0 + 4.0 * m_safe)**(beta + 1))
            J_MM[n+1][n+1] = (A_diff - dt * (
                D_minus_x @ sp.diags(p_min_x * dphi) +
                D_plus_x  @ sp.diags(p_max_x * dphi) +
                D_minus_y @ sp.diags(p_min_y * dphi) +
                D_plus_y  @ sp.diags(p_max_y * dphi))).tocsr()
            J_MM[n+1][n]   = I_neg
            J_MU[n+1][n]   = -(dt * (
                D_minus_x @ sp.diags(phi * (dpx < 0)) @ D_plus_x +
                D_plus_x  @ sp.diags(phi * (dmx > 0)) @ D_minus_x +
                D_minus_y @ sp.diags(phi * (dpy < 0)) @ D_plus_y +
                D_plus_y  @ sp.diags(phi * (dmy > 0)) @ D_minus_y)).tocsr()

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
        f"   LOCAL CONGESTION MFG EXAMPLE 2D (Global Newton)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
        f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
        f"  nu   = {NU_TARGET}, beta = {beta}, zeta = {zeta}\n"
        f"Solver Parameters:\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  Continuation: NU_START={NU_START}, TAU={TAU}, GAMMA={GAMMA}, CONT_TOL={CONT_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  initial_support = [0.375, 0.625]^2, initial_height = 4.0, boundary = periodic\n"
        f"  terminal_wells = [(0.3, 0.5), (0.7, 0.5)], terminal_scale = 15.0, congestion_scale = 4.0\n"
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
plt.savefig(plot_contour_filename, dpi=150)
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
plt.savefig(plot_surface_filename, dpi=150)
plt.show()
print(f"Saved: {plot_surface_filename}")
