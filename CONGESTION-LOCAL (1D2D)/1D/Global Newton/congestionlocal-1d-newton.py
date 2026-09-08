"""Solve the 1D local-congestion MFG with Global Newton and viscosity continuation."""

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
nu   = 0.01              # Viscosity coefficient
beta = 1.5               # Congestion exponent
zeta = 1.0               # Coupling strength

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0   # Spatial domain boundaries
T  = 1.0                 # Total time horizon
Nx = 200                 # Number of spatial intervals
Nt = 100                 # Number of temporal intervals

# --- Newton Iteration Parameters ---
NEWTON_MAX_ITER = 20     # Maximum Global Newton iterations per viscosity level
NEWTON_TOL = 1e-9        # Convergence tolerance for the monolithic residual

# --- Continuation Parameters ---
NU_START  = 1.0          # Starting viscosity
NU_TARGET = 0.01         # Target viscosity
TAU      = 0.75         # Reduction factor for viscosity step
GAMMA     = 0.5          # Backtracking factor
CONT_TOL  = 1e-4         # Minimum allowable step size

# --- Derived Quantities ---
L     = xmax - xmin
Dx    = L / Nx
dt    = T / Nt
x     = np.linspace(xmin, xmax, Nx, endpoint=False)
num_x = x.size

# --- Saved File Names ---
history_filename      = "congestionlocal-1d-newton-history.txt"
plot_snapshot_filename = "congestionlocal-1d-newton-plot.png"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# Initial condition: square wave on [0.375, 0.625]
m0 = np.zeros(num_x)
m0[(x >= 0.375) & (x <= 0.625)] = 4.0
m0 /= (np.sum(m0) * Dx)

# Terminal cost: double quadratic well at x=0.3 and x=0.7
uT = 15.0 * np.minimum((x - 0.3)**2, (x - 0.7)**2)

def cong_coeff(m):
    """1 / (1 + 4m)^beta"""
    return 1.0 / (1.0 + 4.0 * np.maximum(m, 0.0))**beta

def discrete_Hamiltonian(dp, dm, m_val):
    """H = (p_min^2 + p_max^2) / (2*(1+4m)^beta) - zeta*m"""
    p_sq = np.minimum(dp, 0)**2 + np.maximum(dm, 0)**2
    return 0.5 * cong_coeff(m_val) * p_sq - zeta * m_val

def dH_dp1(p_min, m_val):
    """dH/d(D+u) = p_min / (1+4m)^beta"""
    return cong_coeff(m_val) * p_min

def dH_dp2(p_max, m_val):
    """dH/d(D-u) = p_max / (1+4m)^beta"""
    return cong_coeff(m_val) * p_max

def dH_dm_val(p_min, p_max, m_val):
    """dH/dm = -2*beta*(p_min^2+p_max^2)/(1+4m)^{beta+1} - zeta"""
    p_sq = p_min**2 + p_max**2
    return -2.0 * beta * p_sq / (1.0 + 4.0 * np.maximum(m_val, 0.0))**(beta + 1) - zeta

def dphi_dm(m_val):
    """d/dm [m/(1+4m)^beta] = (1 + 4m(1-beta)) / (1+4m)^{beta+1}"""
    m_s = np.maximum(m_val, 0.0)
    return (1.0 + 4.0 * m_s * (1.0 - beta)) / (1.0 + 4.0 * m_s)**(beta + 1)

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
data_m_op = np.array([-np.ones(num_x), np.ones(num_x)])
D_minus = sp.spdiags(data_m_op, [-1, 0], num_x, num_x).tolil()
D_minus[0, 0]       =  1.0
D_minus[0, num_x-1] = -1.0
D_minus = D_minus.tocsr() / Dx

# Laplacian Matrix
Laplacian = D_plus @ D_minus

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

    I_neg = -I.tocsr()

    for it in range(NEWTON_MAX_ITER):
        J_UU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_UM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MU = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]
        J_MM = [[None for _ in range(Nt+1)] for _ in range(Nt+1)]

        F_U = np.zeros((Nt+1, num_x))
        F_M = np.zeros((Nt+1, num_x))

        # Terminal condition for U: U[Nt] = uT
        J_UU[Nt][Nt] = I.tocsr()
        F_U[Nt]      = U[Nt] - uT
        J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        # Initial condition for M: M[0] = m0
        J_MM[0][0] = I.tocsr()
        F_M[0]     = M[0] - m0
        J_MU[0][0] = J_MU[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        for n in range(Nt):
            un, un_next = U[n], U[n+1]
            mn, mn_next = M[n], M[n+1]

            dp    = D_plus  @ un
            dm    = D_minus @ un
            p_min = np.minimum(dp, 0)
            p_max = np.maximum(dm, 0)
            cc    = cong_coeff(mn_next)   # 1/(1+4m)^beta at M^{n+1}

            # --- Backward HJB Equation ---
            F_U[n] = (un - un_next) / dt - nu_val * Laplacian @ un + discrete_Hamiltonian(dp, dm, mn_next)

            J_UU[n][n]   = ((1.0/dt) * I - nu_val * Laplacian
                            + sp.diags(dH_dp1(p_min, mn_next)) @ D_plus
                            + sp.diags(dH_dp2(p_max, mn_next)) @ D_minus).tocsr()
            J_UU[n][n+1] = (-(1.0/dt) * I).tocsr()
            J_UM[n][n]   = sp.csr_matrix((num_x, num_x))
            J_UM[n][n+1] = sp.diags(dH_dm_val(p_min, p_max, mn_next)).tocsr()

            # --- Forward FP Equation ---
            # Transport: D_-(m * p_min/(1+4m)^b) + D_+(m * p_max/(1+4m)^b)
            trans = D_minus @ (mn_next * cc * p_min) + D_plus @ (mn_next * cc * p_max)
            F_M[n+1] = (mn_next - mn) / dt - nu_val * Laplacian @ mn_next - trans

            dphi = dphi_dm(mn_next)
            J_MM[n+1][n+1] = ((1.0/dt) * I - nu_val * Laplacian
                              - D_minus @ sp.diags(p_min * dphi)
                              - D_plus  @ sp.diags(p_max * dphi)).tocsr()
            J_MM[n+1][n]   = (-(1.0/dt) * I).tocsr()
            J_MU[n+1][n]   = -(D_minus @ sp.diags(mn_next * cc * (dp < 0)) @ D_plus
                               + D_plus  @ sp.diags(mn_next * cc * (dm > 0)) @ D_minus).tocsr()

        J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)],
                     [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()
        F = np.concatenate([F_U.flatten(), F_M.flatten()])
        res = np.linalg.norm(F, np.inf)

        if np.isnan(res) or np.isinf(res):
            return np.concatenate([U.flatten(), M.flatten()]), False, res
        if res < NEWTON_TOL:
            iter_log = f"      Iter {it:2d} | res = {res:.4e} (Converged)"
            print(iter_log)
            if log_file: log_file.write(iter_log + "\n")
            return np.concatenate([U.flatten(), M.flatten()]), True, res

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
        f"   LOCAL CONGESTION MFG EXAMPLE (Global Newton)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {NU_TARGET}, beta = {beta}, zeta = {zeta}\n"
        f"Solver Parameters:\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  Continuation: NU_START={NU_START}, TAU={TAU}, GAMMA={GAMMA}, CONT_TOL={CONT_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  initial_support = [0.375, 0.625], initial_height = 4.0, boundary = periodic\n"
        f"  terminal_wells = [0.3, 0.7], terminal_scale = 15.0, congestion_scale = 4.0\n"
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
            print(f"  Attempting nu = {nu_next:.6f}..."); f.write(f"  Attempting nu = {nu_next:.6f}...\n")
            t_nu_start = time.time()
            W_trial, conv, r = global_newton_step(nu_next, W, f)
            t_nu_elapsed = time.time() - t_nu_start
            if conv:
                W, nu_curr, success = W_trial, nu_next, True
                conv_msg = f"    -> Converged! (res={r:.4e}) Total for nu={nu_next:.6f}: {t_nu_elapsed:.4f}s\n"
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
target_times = [0.0, T/5, 2*T/5, 3*T/5, 4*T/5, T]
colors = plt.cm.jet(np.linspace(0, 1, len(target_times)))

axes[0].imshow(M_flow, aspect='auto', extent=[xmin, xmax, T, 0], cmap='viridis')
axes[0].set_title('Density Evolution (m)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('t')

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[1].plot(x, M_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val:.2f}')
axes[1].set_title(f'Density Snapshots (T={T})')
axes[1].set_xlabel('x'); axes[1].grid(True); axes[1].legend(fontsize=8)

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[2].plot(x, U_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val:.2f}')
axes[2].set_title(f'Value Function Snapshots (T={T})')
axes[2].set_xlabel('x'); axes[2].grid(True); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(plot_snapshot_filename, dpi=150)
plt.show()
