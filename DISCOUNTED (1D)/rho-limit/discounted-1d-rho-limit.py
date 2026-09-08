import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import csv

# ==========================================
# SECTION 1: GLOBAL PARAMETERS
# ==========================================
sigma, nu, mu = 2.0, 2.0, 0.0             
width, Nx = 2.0, 200                 

T, Nt = 10.0, 100                

NEWTON_MAX_ITER, NEWTON_TOL = 15, 2e-9        
RHO_START, RHO_TARGET, BETA = 1e-1, 1e-4, 0.5               

L, Dx, dt = 2.0 * width, (2.0 * width) / Nx, T / Nt              
x = np.linspace(mu - width, mu + width, Nx + 1) 
num_x = x.size
history_filename = f"discounted-1d-rho-limit-history_T{T}.txt"

def compute_f(m_dist):
    return -np.log(np.maximum(m_dist, 1e-15))

# ==========================================
# SECTION 2: DISCRETE OPERATORS
# ==========================================
I, I_neg = sp.eye(num_x), -sp.eye(num_x).tocsr()
data_p = np.array([-np.ones(num_x), np.ones(num_x)])

D_plus = sp.spdiags(data_p, [0, 1], num_x, num_x).tolil()
D_plus[num_x-1, :] = 0 
D_plus = D_plus.tocsr() / Dx

D_minus = sp.spdiags(data_p, [-1, 0], num_x, num_x).tolil()
D_minus[0, :] = 0 
D_minus = D_minus.tocsr() / Dx

Laplacian = D_plus @ D_minus

# ==========================================
# SECTION 3: PART A - COMPUTE DISCRETE BENCHMARK
# ==========================================
print(f"{'='*60}\n PART A: COMPUTING EXACT DISCRETE BENCHMARK\n{'='*60}")

eta_stat = 1.0 / (2.0 * nu)
s_sq_stat = nu / (2.0 * eta_stat)
m_peak_stat = 1.0 / np.sqrt(2.0 * np.pi * s_sq_stat)
m_exact_stat = m_peak_stat * np.exp(-(x - mu)**2 / (2.0 * s_sq_stat))
target_mass = np.sum(m_exact_stat) * Dx 

u_exact_stat = eta_stat * (x - mu)**2
m_bnd_L, m_bnd_R = m_exact_stat[0], m_exact_stat[-1]
u_bnd_L, u_bnd_R = u_exact_stat[0], u_exact_stat[-1]

in_idx = slice(1, -1)
num_in = num_x - 2
X = np.zeros(2 * num_in + 1)
X[:num_in] = u_exact_stat[in_idx]
X[num_in:2*num_in] = m_exact_stat[in_idx]
X[-1] = 1.0 + 0.5 * np.log(np.pi * sigma**4 / 2.0)   # continuum lambda = d - (d/2) ln(2/(pi sigma^4)), d = 1 

LAMBDA_H = None
with open(history_filename, "w") as f_hist:
    f_hist.write("PART A: STATIONARY NEWTON HISTORY\n")
    for it in range(20):
        u_in, m_in, lam_var = X[:num_in], X[num_in:2*num_in], X[-1]
        u = np.concatenate(([u_bnd_L], u_in, [u_bnd_R]))
        m = np.concatenate(([m_bnd_L], m_in, [m_bnd_R]))
        dp, dm = D_plus @ u, D_minus @ u
        p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
        H_p = 0.5 * p_min**2 + 0.5 * p_max**2
        J_G = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()
        R_u_full = -nu * (Laplacian @ u) + H_p - compute_f(m) + lam_var
        Adv_Op = J_G.transpose().tocsr()
        R_m_full = -nu * (Laplacian @ m) + Adv_Op @ m
        R = np.concatenate((R_u_full[in_idx], R_m_full[in_idx], [np.sum(m) * Dx - target_mass]))
        res_norm = np.linalg.norm(R, np.inf)
        
        stat_log = f"Stat Iter {it:<2} | Res: {res_norm:<10.4e} | Lambda: {lam_var:.6f}"
        print(stat_log); f_hist.write(stat_log + "\n")
        
        if res_norm < 1e-10:
            LAMBDA_H = lam_var
            print(f"---> EXACT DISCRETE BENCHMARK LOCKED: {LAMBDA_H:.6f} <---\n")
            f_hist.write(f"\n---> EXACT DISCRETE BENCHMARK LOCKED: {LAMBDA_H:.6f} <---\n\n")
            break 
            
        J_uu_full = -nu * Laplacian + J_G
        J_um_full = sp.diags(1.0 / np.maximum(m, 1e-15)).tocsr()
        J_mu_full = (
            D_plus.transpose() @ sp.diags(m * (dp < 0)) @ D_plus
            + D_minus.transpose() @ sp.diags(m * (dm > 0)) @ D_minus
        ).tocsr()
        J_mm_full = -nu * Laplacian + Adv_Op
        J = sp.bmat([[J_uu_full[in_idx, in_idx], J_um_full[in_idx, in_idx], np.ones((num_in, 1))], 
                     [J_mu_full[in_idx, in_idx], J_mm_full[in_idx, in_idx], np.zeros((num_in, 1))], 
                     [np.zeros((1, num_in)), np.full((1, num_in), Dx), np.zeros((1, 1))]]).tocsr()
        delta = spla.spsolve(J, R)
        step = 1.0
        while np.any(m_in - step * delta[num_in:2*num_in] <= 1e-8): step *= 0.5 
        X -= step * delta

# ==========================================
# SECTION 4: PART B - DYNAMIC RHO SOLVER
# ==========================================
def get_exact_params(rho_val):
    eta = (1.0 - rho_val * nu) / (2.0 * nu)
    s_sq = nu / (2.0 * eta)
    m_peak = 1.0 / np.sqrt(2.0 * np.pi * s_sq)
    return eta, s_sq, m_peak

def global_newton_step(rho_val, W_guess, log_file):
    eta, s_sq, m_peak = get_exact_params(rho_val)
    m_exact_bnd = m_peak * np.exp(-(x - mu)**2 / (2.0 * s_sq))
    u_exact_bnd = eta * (x - mu)**2 + (LAMBDA_H / rho_val)
    
    if W_guess is None:
        U, M = np.tile(u_exact_bnd, (Nt+1, 1)), np.tile(m_exact_bnd, (Nt+1, 1))
    else:
        U, M = W_guess[:(Nt+1)*num_x].reshape((Nt+1, num_x)), W_guess[(Nt+1)*num_x:].reshape((Nt+1, num_x))
    
    for it in range(NEWTON_MAX_ITER):
        J_UU, J_UM = [[None]* (Nt+1) for _ in range(Nt+1)], [[None]* (Nt+1) for _ in range(Nt+1)]
        J_MU, J_MM = [[None]* (Nt+1) for _ in range(Nt+1)], [[None]* (Nt+1) for _ in range(Nt+1)]
        J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))
        J_MU[0][Nt]  = sp.csr_matrix((num_x, num_x))
        F_U, F_M = np.zeros((Nt+1, num_x)), np.zeros((Nt+1, num_x))
        A_diff_U = (1.0 + rho_val * dt) * I - dt * nu * Laplacian
        J_UU[Nt][Nt], F_U[Nt] = sp.eye(num_x).tocsr(), U[Nt] - u_exact_bnd
        J_MM[0][0], F_M[0] = sp.eye(num_x).tocsr(), M[0] - m_exact_bnd
        for n in range(Nt):
            u_curr, u_next, m_curr, m_next = U[n], U[n+1], M[n], M[n+1]
            dp, dm = D_plus @ u_curr, D_minus @ u_curr
            p_min, p_max, m_safe = np.minimum(dp, 0), np.maximum(dm, 0), np.maximum(m_curr, 1e-15)
            F_u = A_diff_U @ u_curr + dt * ((0.5 * p_min**2 + 0.5 * p_max**2) - compute_f(m_safe)) - u_next
            J_G = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()
            J_u = A_diff_U + dt * J_G
            F_u[0], F_u[-1] = u_curr[0] - u_exact_bnd[0], u_curr[-1] - u_exact_bnd[-1]
            J_u_lil = J_u.tolil(); J_u_lil[0, :], J_u_lil[-1, :], J_u_lil[0, 0], J_u_lil[-1, -1] = 0, 0, 1, 1
            J_um_lil = (dt * sp.diags(1.0 / m_safe)).tolil(); J_um_lil[0, :], J_um_lil[-1, :] = 0, 0
            I_neg_u = I_neg.tolil(); I_neg_u[0, 0], I_neg_u[-1, -1] = 0, 0
            F_U[n], J_UU[n][n], J_UU[n][n+1], J_UM[n][n] = F_u, J_u_lil.tocsr(), I_neg_u.tocsr(), J_um_lil.tocsr()
            Adv_Op = J_G.transpose().tocsr()
            A_M = I - dt * nu * Laplacian + dt * Adv_Op
            F_m = A_M @ m_next - m_curr
            J_mu = dt * (
                D_plus.transpose() @ sp.diags(m_next * (dp < 0)) @ D_plus
                + D_minus.transpose() @ sp.diags(m_next * (dm > 0)) @ D_minus
            ).tocsr()
            F_m[0], F_m[-1] = m_next[0] - m_exact_bnd[0], m_next[-1] - m_exact_bnd[-1]
            A_M_lil = A_M.tolil(); A_M_lil[0, :], A_M_lil[-1, :], A_M_lil[0, 0], A_M_lil[-1, -1] = 0, 0, 1, 1
            J_mu_lil = J_mu.tolil(); J_mu_lil[0, :], J_mu_lil[-1, :] = 0, 0
            I_neg_m = I_neg.tolil(); I_neg_m[0, 0], I_neg_m[-1, -1] = 0, 0
            F_M[n+1], J_MM[n+1][n+1], J_MM[n+1][n], J_MU[n+1][n] = F_m, A_M_lil.tocsr(), I_neg_m.tocsr(), J_mu_lil.tocsr()
        J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)], [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()
        F = np.concatenate([F_U.flatten(), F_M.flatten()])
        # Scale the residual by the solution magnitude: u = O(lambda/rho) grows
        # without bound as rho -> 0, so an absolute tolerance cannot be met
        # below the floating-point floor at the smallest discount factors.
        res = np.linalg.norm(F, np.inf) / max(1.0, np.max(np.abs(U)))
        
        t_solve = time.time()
        if res < NEWTON_TOL: 
            iter_str = f"      Iter {it:<2} | res = {res:.4e} (Converged)"
            print(iter_str); log_file.write(iter_str + "\n")
            return np.concatenate([U.flatten(), M.flatten()]), True
        dW = spla.spsolve(J, F)
        iter_str = f"      Iter {it:<2} | res = {res:.4e} | Solve Time: {time.time()-t_solve:.4f}s"
        print(iter_str); log_file.write(iter_str + "\n")
        
        U = (U.flatten() - dW[:(Nt+1)*num_x]).reshape((Nt+1, num_x))
        M = (M.flatten() - dW[(Nt+1)*num_x:]).reshape((Nt+1, num_x))
    return np.concatenate([U.flatten(), M.flatten()]), False

# ==========================================
# SECTION 5: RHO CONTINUATION & DATA COLLECTION
# ==========================================
fig_master = plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.4, 1, 7))
color_idx = 0

rho_curr = RHO_START
W = None

# TARGETS: Exactly hit these values
parabola_targets = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
loglog_targets = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
all_targets = sorted(list(set(parabola_targets + loglog_targets)), reverse=True)

convergence_table = []
parabola_profiles = {"x": x}

with open(history_filename, "a") as f_hist:
    f_hist.write("\nPART B: DYNAMIC RHO CONTINUATION HISTORY\n")
    while rho_curr >= RHO_TARGET:
        header = f"\nSolving rho = {rho_curr:.4e}"
        print(header); f_hist.write(header + "\n")
        W, converged = global_newton_step(rho_curr, W, f_hist)
        if not converged: break
        
        u_mid = W[:(Nt+1)*num_x].reshape((Nt+1, num_x))[Nt // 2]
        current_error = np.max(np.abs(rho_curr * u_mid - LAMBDA_H))
        
        # Collect Log-Log data for target decades
        if any(np.isclose(rho_curr, t, atol=1e-9) for t in loglog_targets):
            convergence_table.append((rho_curr, current_error))
        
        # Collect Parabola data for specific targets
        if any(np.isclose(rho_curr, t, atol=1e-9) for t in parabola_targets):
            parabola_profiles[f"rho_{rho_curr}"] = rho_curr * u_mid
            plt.plot(x, rho_curr * u_mid, color=colors[color_idx % 7], linewidth=2.5, label=f'$\\rho={rho_curr}$')
            color_idx += 1
        
        if np.isclose(rho_curr, RHO_TARGET, atol=1e-9): break
        
        # Force hit logic for all targets
        next_rho = BETA * rho_curr
        for t in all_targets:
            if rho_curr > t > next_rho - 1e-10:
                next_rho = t
                break
        rho_curr = max(RHO_TARGET, next_rho)

# ==========================================
# SECTION 6: TIKZ EXPORT & FINAL PLOTS
# ==========================================
# 1. Export Convergence Table to CSV
with open('convergence_data_tikz.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['rho', 'error'])
    for r, err in convergence_table:
        writer.writerow([f"{r:.4e}", f"{err:.4e}"])

# 2. Export Parabola Profiles to CSV
with open('parabolas_data_tikz.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = list(parabola_profiles.keys())
    writer.writerow(header)
    rows = zip(*parabola_profiles.values())
    writer.writerows(rows)

# Generate plots for verification
plt.axhline(LAMBDA_H, color='r', linestyle='--', linewidth=3, label=f'Benchmark $\\lambda_h$')
plt.title(f'Vanishing Discount Limit $\\rho u \\to \\lambda_h$ (T = {T})'); plt.xlabel('Space (x)'); plt.ylabel('Value ($\\rho u$)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
plt.savefig(f'squashing_parabolas_T{T}.png')

fig_loglog = plt.figure(figsize=(10, 6))
rhos, errs = zip(*convergence_table)
plt.loglog(rhos, errs, marker='o', linestyle='-', linewidth=2.5, color='darkblue', markersize=8, label='Numerical Error')
ref_errors = [errs[0] * (r / rhos[0]) for r in rhos]
plt.loglog(rhos, ref_errors, linestyle='--', color='gray', label='$\mathcal{O}(\\rho)$ Reference')
plt.gca().invert_xaxis(); plt.title(f'Log-Log Convergence (T = {T})'); plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(); plt.tight_layout(); plt.savefig(f'loglog_convergence_T{T}.png')

print(f"\nFinal Discrete Benchmark Lambda_h: {LAMBDA_H:.6f}")
print(f"Saved TiKZ data to 'convergence_data_tikz.csv' and 'parabolas_data_tikz.csv'.")
