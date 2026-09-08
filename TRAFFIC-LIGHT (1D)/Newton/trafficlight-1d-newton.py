"""Solve the 1D traffic-light MFG with Global Newton and viscosity continuation."""

import sys
import time

import matplotlib.animation as animation
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
beta = 2                 # Congestion exponent

# --- Traffic-Light Parameters ---
LIGHT_POS = 0.5          # Position of the light
LIGHT_START = 0.3        # Time light turns RED
LIGHT_END = 0.7          # Time light turns GREEN
LIGHT_PENALTY = 150.0    # Cost of running red light (Extreme barrier)
LIGHT_WIDTH = 0.05       # Width of the stop line

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0    # Spatial domain boundaries
T = 1.0                  # Total time horizon
Nx = 200                 # Number of spatial finite-volume cells
Nt = 100                 # Number of temporal intervals

# --- Newton Iteration Parameters ---
NEWTON_MAX_ITER = 20     # Maximum monolithic Global Newton iterations at each viscosity
NEWTON_TOL = 1e-9        # Convergence tolerance for the monolithic residual

# --- Continuation Parameters ---
NU_START = 0.2           # Starting viscosity
NU_TARGET = 0.001        # Target viscosity
TAU = 0.5                # Reduction factor for viscosity step
GAMMA = 0.5              # Backtracking factor
CONT_TOL = 1e-4          # Minimum allowable step size

# --- Derived Quantities ---
L  = xmax - xmin         # Length of the spatial domain
dt = T / Nt              # Temporal step size
Dx = L / Nx              # Spatial cell width
x  = xmin + (np.arange(Nx) + 0.5) * Dx  # Cell centers
num_x = x.size

# --- Saved File Names ---
history_filename = "trafficlight-1d-newton-history.txt"
plot_snapshot_filename = "trafficlight-1d-newton-plot.png"
video_filename = "trafficlight-1d-newton-animation.mp4"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# --- 1. The Instantaneous "Traffic Light" Potential ---
V_field = np.zeros((Nt + 1, num_x))
barrier_x = np.exp(-((x - LIGHT_POS)**2) / (2 * (LIGHT_WIDTH/2)**2))

for n in range(Nt + 1):
    t = n * dt
    intensity = 1.0 if (LIGHT_START <= t <= LIGHT_END) else 0.0
    V_field[n, :] = LIGHT_PENALTY * intensity * barrier_x

# --- 2. Initial Config (Traffic Jam on Left) ---
m0 = np.exp(-((x - 0.1)**2) / 0.01)
m0 = m0 / (np.sum(m0) * Dx)

# --- 3. Terminal Cost (Desire to reach Right) ---
uT = 5.0 * (x - 1.0)**2

def discrete_Hamiltonian(dp, dm):
    """Monotone Godunov Hamiltonian for H(p)=|p|^2/2."""
    return 0.5 * np.minimum(dp, 0.0)**2 + 0.5 * np.maximum(dm, 0.0)**2

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I = sp.eye(num_x, format="csr")

D_plus = sp.diags([-np.ones(num_x), np.ones(num_x - 1)], [0, 1], shape=(num_x, num_x), format="lil")
D_plus[-1, :] = 0.0
D_plus = D_plus.tocsr() / Dx

D_minus = sp.diags([-np.ones(num_x - 1), np.ones(num_x)], [-1, 0], shape=(num_x, num_x), format="lil")
D_minus[0, :] = 0.0
D_minus = D_minus.tocsr() / Dx

# Conservative symmetric Neumann diffusion operator.
D2 = -(D_plus.transpose() @ D_plus).tocsr()

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

        A_diff_U = I - dt * nu_val * D2

        # Terminal condition for U
        J_UU[Nt][Nt] = I.tocsr()
        F_U[Nt] = U[Nt] - uT
        J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        # Initial condition for M
        J_MM[0][0] = I.tocsr()
        F_M[0] = M[0] - m0
        J_MU[0][0] = J_MU[Nt][Nt] = sp.csr_matrix((num_x, num_x))

        for n in range(Nt):
            u_curr, u_next = U[n], U[n+1]
            m_curr, m_next = M[n], M[n+1]

            dp, dm = D_plus @ u_curr, D_minus @ u_curr
            p_min, p_max = np.minimum(dp, 0.0), np.maximum(dm, 0.0)
            m_next_safe = np.maximum(m_next, 1e-12)
            f_val = V_field[n] + m_next_safe**beta

            # --- Backward HJB Equation ---
            F_U[n] = A_diff_U @ u_curr + dt * discrete_Hamiltonian(dp, dm) - u_next - dt * f_val

            J_G = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()
            J_UU[n][n]   = (A_diff_U + dt * J_G).tocsr()
            J_UU[n][n+1] = I_neg
            J_UM[n][n]   = sp.csr_matrix((num_x, num_x))
            coupling_derivative = beta * m_next_safe**(beta-1) * (m_next > 1e-12)
            J_UM[n][n+1] = (-dt * sp.diags(coupling_derivative)).tocsr()

            # --- Forward FP Equation ---
            Adv_matrix = J_G.transpose().tocsr()
            A_M = I - dt * nu_val * D2 + dt * Adv_matrix

            F_M[n+1] = A_M @ m_next - m_curr

            J_MM[n+1][n+1] = A_M.tocsr()
            J_MM[n+1][n]   = I_neg
            J_MU[n+1][n] = dt * (
                D_plus.transpose() @ sp.diags(m_next * (dp < 0.0)) @ D_plus
                + D_minus.transpose() @ sp.diags(m_next * (dm > 0.0)) @ D_minus
            ).tocsr()

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
        f"   TRAFFIC LIGHT MFG EXAMPLE (Global Newton)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {NU_TARGET}, beta = {beta}, penalty = {LIGHT_PENALTY}\n"
        f"Solver Parameters:\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"  Continuation: NU_START={NU_START}, TAU={TAU}, GAMMA={GAMMA}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  light_position = {LIGHT_POS}, light_active_interval = [{LIGHT_START}, {LIGHT_END}]\n"
        f"  light_penalty = {LIGHT_PENALTY}, light_width = {LIGHT_WIDTH}\n"
        f"  Hamiltonian = Godunov discretization of 0.5*p^2, boundary = conservative homogeneous Neumann\n"
        f"  initial_density = normalized exp(-(x-0.1)^2/0.01)\n"
        f"  terminal_cost = 5.0*(x-1.0)^2\n"
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
    final_U = W[:(Nt+1)*num_x].reshape((Nt+1, num_x))
    final_M = W[(Nt+1)*num_x:].reshape((Nt+1, num_x))
    time_msg = (f"{'-'*118}\nStatus: CONVERGED at nu={nu_curr}\n"
                f"Maximum mass error: {np.max(np.abs(Dx*np.sum(final_M, axis=1)-1.0)):.6e}\n"
                f"Minimum density: {np.min(final_M):.6e}\n"
                f"Total Execution Time: {total_time:.4f} seconds.\n")
    print(time_msg); f.write(time_msg)

# ==========================================
# SECTION 6: VISUALIZATION
# ==========================================

U_flow = W[:(Nt+1)*num_x].reshape((Nt+1, num_x))
M_flow = W[(Nt+1)*num_x:].reshape((Nt+1, num_x))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
target_times = [0.0, T/5, 2*T/5, 3*T/5, 4*T/5, T]
colors = plt.cm.jet(np.linspace(0, 1, len(target_times)))

im = axes[0].imshow(M_flow, aspect='auto', extent=[xmin, xmax, T, 0], cmap='viridis')
axes[0].set_title('Density Evolution (m)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
fig.colorbar(im, ax=axes[0])

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[1].plot(x, M_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val}')
axes[1].set_title('Density Snapshots')
axes[1].legend()
axes[1].grid(True)

for i, t_val in enumerate(target_times):
    n_idx = min(int(round(t_val / dt)), Nt)
    axes[2].plot(x, U_flow[n_idx], color=colors[i], linewidth=2, label=f't={t_val}')
axes[2].set_title('Value Function Snapshots (u)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(plot_snapshot_filename)
plt.close()
print(f"Static plots saved to {plot_snapshot_filename}")

print("Generating Animation...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, 10.0)
ax.set_xlabel("Position (0=Start, 1=End)")
ax.set_ylabel("Density of Crowd")

line, = ax.plot([], [], 'b-', lw=2, label='Car Density')
light_patch = plt.Rectangle((LIGHT_POS-0.02, 0), 0.04, 10.0, color='green', alpha=0.1)
ax.add_patch(light_patch)
crowd_fill = [ax.fill_between([], [], color='blue', alpha=0.2)]

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

def init():
    line.set_data([], [])
    time_text.set_text('')
    status_text.set_text('')
    return line, time_text, status_text, light_patch

def update(frame):
    y = M_flow[frame]
    line.set_data(x, y)
    crowd_fill[0].remove()
    crowd_fill[0] = ax.fill_between(x, 0, y, color='blue', alpha=0.2)
    t = frame * dt
    time_text.set_text(f"Time: {t:.2f}s")
    if LIGHT_START <= t <= LIGHT_END:
        light_patch.set_color('red');  light_patch.set_alpha(0.3)
        status_text.set_text("STATUS: RED LIGHT");  status_text.set_color('red')
    else:
        light_patch.set_color('green'); light_patch.set_alpha(0.1)
        status_text.set_text("STATUS: GREEN LIGHT"); status_text.set_color('green')
    return line, time_text, status_text, light_patch

ani = animation.FuncAnimation(fig, update, frames=Nt+1, init_func=init, interval=50, blit=False)
ani.save(video_filename, writer='ffmpeg', fps=30)
print(f"Animation saved to {video_filename}")
