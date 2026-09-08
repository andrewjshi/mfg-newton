"""Solve the 1D traffic-light MFG with the Picard method."""

import os
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
nu = 0.02                # Viscosity coefficient
beta = 2                 # Congestion exponent

# --- Traffic Light (Instantaneous) ---
LIGHT_POS = 0.5          # Position of the light
LIGHT_START = 0.3        # Time light turns RED
LIGHT_END = 0.7          # Time light turns GREEN
LIGHT_PENALTY = 150.0    # Cost of running red light (Extreme barrier)
LIGHT_WIDTH = 0.05       # Width of the stop line

# --- Spatial and Temporal Grid ---
xmin, xmax = 0.0, 1.0    # Spatial domain boundaries
T  = 1.0                 # Total time horizon
Nx = 200                 # Number of spatial grid nodes
Nt = 100                 # Number of temporal intervals

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 250    # Maximum number of outer HJB-FP coupling iterations
PICARD_TOL = 1e-6        # Convergence tolerance for the outer Picard loop
DAMPING = 0.1            # Damping parameter (0 < DAMPING <= 1). 1 corresponds to no damping.

NEWTON_MAX_ITER = 20     # Maximum Newton iterations for the backward HJB solves
NEWTON_TOL = 1e-9        # Convergence tolerance for the backward Newton solver

# --- Derived Quantities ---
L  = xmax - xmin         # Length of the spatial domain
dt = T / Nt              # Temporal step size
Dx = (xmax - xmin) / Nx  # Spatial cell width
x  = xmin + (np.arange(Nx) + 0.5) * Dx  # Cell centers (Neumann/no-flux)
num_x = x.size
norm_const = np.sqrt(Dx * dt)

# --- Directories & Saved File Names ---
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

history_filename = "trafficlight-1d-picard-history.txt"
plot_snapshot_filename = "trafficlight-1d-picard-plot.png"
video_filename = "trafficlight-1d-picard-animation.mp4"

# ==========================================
# SECTION 2: PHYSICS DEFINITIONS (INCLUDING HAMILTONIAN)
# ==========================================

# --- 1. The Instantaneous "Traffic Light" Potential ---
V_field = np.zeros((Nt + 1, num_x))
barrier_x = np.exp(-((x - LIGHT_POS)**2) / (2 * (LIGHT_WIDTH/2)**2))

for n in range(Nt + 1):
    t = n * dt
    # Instantaneous Step Activation
    intensity = 1.0 if (LIGHT_START <= t <= LIGHT_END) else 0.0

    # The Red Light Barrier (Base pull -x removed)
    V_field[n, :] = (LIGHT_PENALTY * intensity * barrier_x)

# --- 2. Initial Config (Traffic Jam on Left) ---
m0 = np.exp(-((x - 0.1)**2) / 0.01)
m0 = m0 / (np.sum(m0) * Dx)

# --- 3. Terminal Cost (Desire to reach Right) ---
# High cost at Left, Zero cost at Right
uT = 5.0 * (x - 1.0)**2

def discrete_Hamiltonian(dp, dm):
    """Monotone Godunov Hamiltonian for H(p)=|p|^2/2."""
    return 0.5 * np.minimum(dp, 0.0)**2 + 0.5 * np.maximum(dm, 0.0)**2

# ==========================================
# SECTION 3: DISCRETE OPERATORS
# ==========================================

I = sp.eye(num_x, format="csr")

# One-sided differences.  The missing outward differences are zero, which
# enforces the hard-wall condition in the numerical Hamiltonian.
D_plus = sp.diags([-np.ones(num_x), np.ones(num_x - 1)], [0, 1], shape=(num_x, num_x), format="lil")
D_plus[-1, :] = 0.0
D_plus = D_plus.tocsr() / Dx

D_minus = sp.diags([-np.ones(num_x - 1), np.ones(num_x)], [-1, 0], shape=(num_x, num_x), format="lil")
D_minus[0, :] = 0.0
D_minus = D_minus.tocsr() / Dx

# Conservative symmetric Neumann diffusion operator.  Its row and column sums
# vanish, so the FP solve preserves the discrete mass without renormalization.
D2 = -(D_plus.transpose() @ D_plus).tocsr()

# ==========================================
# SECTION 4: HJB AND FP SOLVERS
# ==========================================

def solve_hjb_backward(M_flow):
    U = np.zeros((Nt + 1, Nx))
    U[Nt] = uT

    A_diff = I - dt * nu * D2

    total_newton = 0
    for n in range(Nt - 1, -1, -1):
        u_next = U[n+1]
        u_curr = u_next.copy()
        f_val = V_field[n] + np.maximum(M_flow[n+1], 0.0)**beta

        for _ in range(NEWTON_MAX_ITER):
            total_newton += 1
            dp, dm = D_plus @ u_curr, D_minus @ u_curr
            p_min, p_max = np.minimum(dp, 0.0), np.maximum(dm, 0.0)

            F = A_diff @ u_curr + dt * discrete_Hamiltonian(dp, dm) - u_next - dt * f_val

            if np.linalg.norm(F, np.inf) < NEWTON_TOL:
                break

            J_G = sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus
            J = A_diff + dt * J_G
            u_curr -= spla.spsolve(J, F)
        U[n] = u_curr
    return U, total_newton

def solve_fp_forward(U_flow):
    M = np.zeros((Nt + 1, Nx))
    M[0] = m0

    for n in range(Nt):
        m_curr = M[n]
        dp, dm = D_plus @ U_flow[n], D_minus @ U_flow[n]
        J_G = (sp.diags(np.minimum(dp, 0.0)) @ D_plus
               + sp.diags(np.maximum(dm, 0.0)) @ D_minus).tocsr()

        # Exact discrete adjoint of the HJB Hamiltonian linearization.
        Adv_matrix = J_G.transpose().tocsr()
        A = I - dt*nu*D2 + dt*Adv_matrix

        M[n+1] = spla.spsolve(A, m_curr)

    return M

# ==========================================
# SECTION 5: DAMPED PICARD ITERATION
# ==========================================

start_time_all = time.time()

with open(history_filename, "w", buffering=1) as f_log:
    header = (
        f"{'='*118}\n"
        f"   TRAFFIC LIGHT MFG EXAMPLE (Picard)\n"
        f"{'='*118}\n"
        f"Parameters:\n"
        f"  xmin = {xmin}, xmax = {xmax}\n"
        f"  T    = {T}, Nx = {Nx}, Nt = {Nt}\n"
        f"  nu   = {nu}, beta = {beta}, penalty = {LIGHT_PENALTY}\n"
        f"Solver Parameters:\n"
        f"  PICARD_MAX_ITER = {PICARD_MAX_ITER}, PICARD_TOL = {PICARD_TOL}\n"
        f"  DAMPING = {DAMPING}\n"
        f"  NEWTON_MAX_ITER = {NEWTON_MAX_ITER}, NEWTON_TOL = {NEWTON_TOL}\n"
        f"Grid Info:\n"
        f"  dt = {dt:.6f}, dx = {Dx:.6f}\n"
        f"Problem Specific Parameters:\n"
        f"  light_position = {LIGHT_POS}, light_active_interval = [{LIGHT_START}, {LIGHT_END}]\n"
        f"  light_penalty = {LIGHT_PENALTY}, light_width = {LIGHT_WIDTH}\n"
        f"  Hamiltonian = Godunov discretization of 0.5*p^2, boundary = conservative homogeneous Neumann\n"
        f"  initial_density = normalized exp(-(x-0.1)^2/0.01)\n"
        f"  terminal_cost = 5.0*(x-1.0)^2\n"
        f"{'-'*118}\n"
        f"{'Iter':<5} | {'Abs Err U':<12} | {'Rel Err U':<12} | {'Abs Err M':<12} | {'Rel Err M':<12} | {'Newton It':<10} | {'Time (s)':<10}\n"
        f"{'-'*118}\n"
    )
    print(header, end='')
    f_log.write(header)

    M_flow = np.zeros((Nt + 1, Nx))
    for n in range(Nt+1): M_flow[n] = m0
    U_flow = np.zeros((Nt + 1, Nx))

    for k in range(1, PICARD_MAX_ITER + 1):
        t0 = time.time()

        U_candidate, n_iters = solve_hjb_backward(M_flow)
        U_flow_new = DAMPING * U_candidate + (1 - DAMPING) * U_flow
        M_candidate = solve_fp_forward(U_flow_new)
        M_flow_new = DAMPING * M_candidate + (1 - DAMPING) * M_flow

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

        print(log_str)
        f_log.write(log_str + "\n")

        U_flow, M_flow = U_flow_new, M_flow_new
        if rel_err_u < PICARD_TOL and rel_err_m < PICARD_TOL:
            conv_msg = (f"{'-'*118}\nCONVERGED at nu={nu} in {k} iterations.\n"
                        f"Maximum mass error: {np.max(np.abs(Dx*np.sum(M_flow_new, axis=1)-1.0)):.6e}\n"
                        f"Minimum density: {np.min(M_flow_new):.6e}\n")
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

print("Generating Static Plots...")

# --- ONLY CHANGE: INCREASE FONT SIZES FOR PAPER ---
plt.rcParams.update({
    'font.size': 25,            # General text
    'axes.labelsize': 20,       # x and y labels
    'axes.titlesize': 22,       # titles
    'xtick.labelsize': 16,      # x-ticks
    'ytick.labelsize': 16,      # y-ticks
    'legend.fontsize': 16,      # legend
})
# ---

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

# --- Image Sequence Saving ---
print(f"Saving individual frames to {IMG_DIR}/ directory...")

fig_frame, ax_frame = plt.subplots(figsize=(10, 6))

for n in range(Nt + 1):
    ax_frame.clear()
    ax_frame.set_xlim(xmin, xmax)
    ax_frame.set_ylim(0, 10.0)
    ax_frame.set_xlabel("Position (0=Start, 1=End)")
    ax_frame.set_ylabel("Density of Crowd")

    t_val = n * dt

    if LIGHT_START <= t_val <= LIGHT_END:
        color = 'red'
        alpha = 0.3
        status = "STATUS: RED LIGHT"
    else:
        color = 'green'
        alpha = 0.1
        status = "STATUS: GREEN LIGHT"

    ax_frame.add_patch(plt.Rectangle((LIGHT_POS-0.02, 0), 0.04, 10.0, color=color, alpha=alpha))

    y_val = M_flow[n]
    ax_frame.plot(x, y_val, 'b-', lw=2)
    ax_frame.fill_between(x, 0, y_val, color='blue', alpha=0.2)

    # --- MOVED TIME AND STATUS TO TITLE ---
    ax_frame.set_title(f"Time = {t_val:.2f}s | {status}", color=color, fontweight='bold')

    frame_path = os.path.join(IMG_DIR, f"frame_{n:03d}.png")
    plt.savefig(frame_path)

plt.close(fig_frame)
print(f"Successfully saved {Nt + 1} frames to the {IMG_DIR}/ directory.")

# --- Animation Saving ---
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

def init():
    line.set_data([], [])
    ax.set_title("")
    return line, light_patch

def update(frame):
    y = M_flow[frame]
    line.set_data(x, y)

    crowd_fill[0].remove()
    crowd_fill[0] = ax.fill_between(x, 0, y, color='blue', alpha=0.2)

    t_curr = frame * dt

    if LIGHT_START <= t_curr <= LIGHT_END:
        light_color = 'red'
        light_patch.set_color('red')
        light_patch.set_alpha(0.3)
        status_str = "STATUS: RED LIGHT"
    else:
        light_color = 'green'
        light_patch.set_color('green')
        light_patch.set_alpha(0.1)
        status_str = "STATUS: GREEN LIGHT"

    # --- MOVED TIME AND STATUS TO TITLE ---
    ax.set_title(f"Time = {t_curr:.2f}s | {status_str}", color=light_color, fontweight='bold')

    return line, light_patch

ani = animation.FuncAnimation(fig, update, frames=Nt+1, init_func=init, interval=50, blit=False)
ani.save(video_filename, writer='ffmpeg', fps=30)
print(f"Animation saved to {video_filename}")
