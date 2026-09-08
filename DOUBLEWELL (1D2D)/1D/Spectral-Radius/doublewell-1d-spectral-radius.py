import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ==========================================
# SECTION 1: PARAMETERS
# ==========================================

coupling_str = 1.0
tilt         = 0.3
well_depth   = 2.0
alpha        = 2

xmin, xmax = -2.5, 2.5
T  = 1.0
Nx = 100
Nt = 30

PICARD_MAX_ITER = 300
PICARD_TOL      = 1e-6
NEWTON_MAX_ITER = 20     # sufficient for quadratic convergence from a good warm start
NEWTON_TOL      = 1e-9
DAMPING         = 0.8

# sweep down to very low nu
NU_LIST = [0.5, 0.4, 0.3, 0.2, 0.17, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07, 0.065, 0.06, 0.055, 0.05]

# continuation params for Global Newton fallback
TAU     = 0.75   # reduction factor
GAMMA    = 0.5    # backtracking factor
CONT_TOL = 1e-5

Dx    = (xmax - xmin) / Nx
dt    = T / Nt
x     = xmin + (np.arange(Nx) + 0.5) * Dx
num_x = Nx

# ==========================================
# SECTION 2: PHYSICS
# ==========================================

V  = well_depth * (x**2 - 1.0)**2 + tilt * x
m0 = np.exp(-x**2 / (2 * 0.6**2))
m0 /= np.sum(m0) * Dx
uT = V.copy()

def compute_f(m):
    return V + coupling_str * np.maximum(m, 0)**alpha

def discrete_H(dp, dm, f):
    return 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2 - f

def dHdp(p):
    return p

# ==========================================
# SECTION 3: OPERATORS
# ==========================================

I = sp.eye(num_x, format='csr')

D_plus = sp.spdiags([-np.ones(num_x), np.ones(num_x)], [0, 1], num_x, num_x).tolil()
D_plus[num_x-1, :] = 0.0
D_plus = D_plus.tocsr() / Dx

D_minus = sp.spdiags([-np.ones(num_x), np.ones(num_x)], [-1, 0], num_x, num_x).tolil()
D_minus[0, :] = 0.0
D_minus = D_minus.tocsr() / Dx

Laplacian = -(D_plus.transpose() @ D_plus).tocsr()

# ==========================================
# SECTION 4: NEWTON-PICARD (TYPE 2)
# ==========================================

def hjb_backward(nu, M_flow):
    U = np.zeros((Nt+1, num_x))
    U[Nt] = uT
    Ad = (I - dt * nu * Laplacian).tocsr()
    for n in range(Nt-1, -1, -1):
        u = U[n+1].copy()
        f = compute_f(M_flow[n+1])
        for _ in range(NEWTON_MAX_ITER):
            dp, dm = D_plus @ u, D_minus @ u
            F = Ad @ u + dt * discrete_H(dp, dm, f) - U[n+1]
            if np.linalg.norm(F, np.inf) < NEWTON_TOL:
                break
            pm, px = np.minimum(dp, 0), np.maximum(dm, 0)
            J = Ad + dt * (sp.diags(dHdp(pm)) @ D_plus + sp.diags(dHdp(px)) @ D_minus)
            u -= spla.spsolve(J, F)
        U[n] = u
    return U

def fp_forward(nu, U_flow):
    """FP solve using the exact adjoint of the Godunov HJB linearization."""
    M = np.zeros((Nt+1, num_x))
    M[0] = m0
    for n in range(Nt):
        dp, dm = D_plus @ U_flow[n], D_minus @ U_flow[n]
        J_H = (sp.diags(dHdp(np.minimum(dp, 0))) @ D_plus +
               sp.diags(dHdp(np.maximum(dm, 0))) @ D_minus).tocsr()
        A = (I - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
        M[n+1] = spla.spsolve(A, M[n])
    return M

def newton_picard(nu, damping=DAMPING, U_init=None, M_init=None):
    """Returns (U, M, converged)."""
    if M_init is not None:
        M_flow = M_init.copy()
    else:
        M_flow = np.tile(m0, (Nt+1, 1))
        A0 = (I - dt*nu*Laplacian).tocsr()
        for n in range(Nt):
            M_flow[n+1] = spla.spsolve(A0, M_flow[n])

    U_flow = U_init.copy() if U_init is not None else np.zeros((Nt+1, num_x))

    for k in range(1, PICARD_MAX_ITER+1):
        U_new = damping * hjb_backward(nu, M_flow) + (1-damping) * U_flow
        M_new = damping * fp_forward(nu, U_new)    + (1-damping) * M_flow
        delta_u_norm = np.linalg.norm((U_new - U_flow).ravel())
        delta_m_norm = np.linalg.norm((M_new - M_flow).ravel())
        previous_u_norm = np.linalg.norm(U_flow.ravel())
        previous_m_norm = np.linalg.norm(M_flow.ravel())
        rel_u = delta_u_norm / previous_u_norm if previous_u_norm > 1e-12 else (0.0 if delta_u_norm <= 1e-12 else 1.0)
        rel_m = delta_m_norm / previous_m_norm if previous_m_norm > 1e-12 else (0.0 if delta_m_norm <= 1e-12 else 1.0)
        U_flow, M_flow = U_new, M_new
        if rel_u < PICARD_TOL and rel_m < PICARD_TOL:
            print(f"    [NP] Converged in {k} iters (rel_u={rel_u:.2e})")
            newton_picard._last_iters = k
            return U_flow, M_flow, True
    print(f"    [NP] Did NOT converge (rel_u={rel_u:.2e}, rel_m={rel_m:.2e})")
    newton_picard._last_iters = PICARD_MAX_ITER
    return U_flow, M_flow, False

# ==========================================
# SECTION 5: GLOBAL NEWTON (TYPE 1)
# ==========================================

def assemble_global_residual_and_jacobian(nu, U, M):
    I_neg = -I.tocsr()
    J_UU = [[sp.csr_matrix((num_x, num_x))]*(Nt+1) for _ in range(Nt+1)]
    J_UM = [[sp.csr_matrix((num_x, num_x))]*(Nt+1) for _ in range(Nt+1)]
    J_MU = [[sp.csr_matrix((num_x, num_x))]*(Nt+1) for _ in range(Nt+1)]
    J_MM = [[sp.csr_matrix((num_x, num_x))]*(Nt+1) for _ in range(Nt+1)]
    # convert to mutable lists
    J_UU = [[sp.csr_matrix((num_x, num_x)) for _ in range(Nt+1)] for _ in range(Nt+1)]
    J_UM = [[sp.csr_matrix((num_x, num_x)) for _ in range(Nt+1)] for _ in range(Nt+1)]
    J_MU = [[sp.csr_matrix((num_x, num_x)) for _ in range(Nt+1)] for _ in range(Nt+1)]
    J_MM = [[sp.csr_matrix((num_x, num_x)) for _ in range(Nt+1)] for _ in range(Nt+1)]

    F_U = np.zeros((Nt+1, num_x))
    F_M = np.zeros((Nt+1, num_x))

    J_UU[Nt][Nt] = I.tocsr()
    F_U[Nt]      = U[Nt] - uT
    J_MM[0][0]   = I.tocsr()
    F_M[0]       = M[0] - m0

    for n in range(Nt):
        un, mn1 = U[n], M[n+1]
        dp, dm  = D_plus @ un, D_minus @ un
        pm, px  = np.minimum(dp, 0), np.maximum(dm, 0)
        f_val   = compute_f(mn1)

        F_U[n] = (un - U[n+1])/dt - nu*Laplacian@un + discrete_H(dp, dm, f_val)

        dH_dm = -coupling_str * alpha * np.maximum(mn1, 0)**(alpha-1)
        J_UU[n][n]   = ((1/dt)*I - nu*Laplacian
                        + sp.diags(dHdp(pm))@D_plus
                        + sp.diags(dHdp(px))@D_minus).tocsr()
        J_UU[n][n+1] = (-(1/dt)*I).tocsr()
        J_UM[n][n+1] = sp.diags(dH_dm).tocsr()

        F_M[n+1] = (mn1 - M[n])/dt - nu*Laplacian@mn1 - (D_minus@(mn1*pm) + D_plus@(mn1*px))
        J_MM[n+1][n+1] = ((1/dt)*I - nu*Laplacian
                          - (D_minus@sp.diags(pm) + D_plus@sp.diags(px))).tocsr()
        J_MM[n+1][n]   = I_neg
        J_MU[n+1][n]   = -(D_minus@sp.diags(mn1*(dp<0))@D_plus
                           + D_plus@sp.diags(mn1*(dm>0))@D_minus).tocsr()

    J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)],
                 [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()
    F = np.concatenate([F_U.flatten(), F_M.flatten()])
    return J, F, J_UU, J_UM, J_MU, J_MM

def global_newton_solve(nu, U_init, M_init):
    """Single Global Newton run at fixed nu. Returns (U, M, converged)."""
    U = U_init.copy()
    M = M_init.copy()
    for it in range(NEWTON_MAX_ITER):
        J, F, _, _, _, _ = assemble_global_residual_and_jacobian(nu, U, M)
        res = np.linalg.norm(F, np.inf)
        print(f"      [GN] iter {it:2d} | res={res:.3e}")
        if np.isnan(res) or np.isinf(res):
            return U, M, False
        if res < NEWTON_TOL:
            return U, M, True
        try:
            dW  = spla.spsolve(J, F)
            U   = (U.flatten() - dW[:(Nt+1)*num_x]).reshape((Nt+1, num_x))
            M   = (M.flatten() - dW[(Nt+1)*num_x:]).reshape((Nt+1, num_x))
        except Exception as e:
            print(f"      [GN] solve failed: {e}")
            return U, M, False
    return U, M, False

def global_newton_with_continuation(nu_target, U_warm, M_warm):
    """
    Viscosity continuation for Global Newton starting from nu=0.5.
    U_warm/M_warm used only if direct solve at nu_target succeeds.
    Returns (U, M, converged).
    """
    # try direct solve first with the warm start
    print(f"    [GN] Trying direct solve at nu={nu_target:.4f}")
    U, M, conv = global_newton_solve(nu_target, U_warm, M_warm)
    if conv:
        print(f"    [GN] Direct solve converged.")
        return U, M, True

    # full continuation from nu=0.5 with naive init
    nu_start = 0.5
    print(f"    [GN] Continuation from nu={nu_start} -> {nu_target:.4f} (naive init)")
    U = np.tile(uT, (Nt+1, 1))
    M = np.tile(m0, (Nt+1, 1))

    nu_curr = nu_start
    U, M, conv = global_newton_solve(nu_curr, U, M)
    if not conv:
        print(f"    [GN] Failed even at nu_start={nu_start}")
        return U, M, False

    while nu_curr > nu_target:
        nu_next = max(nu_target, TAU * nu_curr)
        success = False
        while not success:
            U_trial, M_trial, conv = global_newton_solve(nu_next, U, M)
            if conv:
                U, M, nu_curr, success = U_trial, M_trial, nu_next, True
                print(f"    [GN] Converged at nu={nu_next:.5f}")
            else:
                nu_next = nu_curr - GAMMA * (nu_curr - nu_next)
                if abs(nu_curr - nu_next) < CONT_TOL:
                    print(f"    [GN] Stalled at nu={nu_curr:.5f}")
                    return U, M, False
    return U, M, True

# ==========================================
# SECTION 6: SPECTRAL RADIUS VIA NUMERICAL LINEARIZATION
#
# Instead of assembling Jacobian blocks analytically
# (which requires NP and GN to share the same FP scheme),
# we compute the linearized Picard operator P numerically:
#
#   P delta_U = [delta_U from one linearized Picard step]
#
# Step 1 (FP):   delta_M = [fp_forward(U*+eps*dU) - fp_forward(U*)] / eps
# Step 2 (HJB):  delta_U = [hjb_backward(M*+eps*dM) - hjb_backward(M*)] / eps
#
# This works with the NP's native solvers directly,
# no Jacobian derivation needed.
# ==========================================

def apply_P_numerical(delta_U_flat, U_star, M_star, nu, eps=1e-4):
    """Apply linearized Picard operator P to a flat perturbation vector."""
    delta_U = delta_U_flat.reshape(Nt+1, num_x)

    # Step 1: linearized FP step
    M_pert  = fp_forward(nu, U_star + eps * delta_U)
    delta_M = (M_pert - M_star) / eps

    # Step 2: linearized HJB step (Newton from U_star as init)
    U_pert  = hjb_backward(nu, M_star + eps * delta_M)
    delta_U_new = (U_pert - U_star) / eps

    return delta_U_new.flatten()

def spectral_radius_numerical(U_star, M_star, nu, damping=DAMPING, n_iter=25, n_restarts=2):
    """
    Estimate rho(P) and rho(P_theta) via power iteration with random restarts.
    P_theta = (1-theta)*I + theta*P
    Both computed independently via their own power iterations.
    """
    N = (Nt+1) * num_x

    def apply_P(v_flat):
        return apply_P_numerical(v_flat, U_star, M_star, nu)

    def apply_Ptheta(v_flat):
        return (1 - damping) * v_flat + damping * apply_P(v_flat)

    def power_iter(apply_fn, n_iter, n_restarts):
        rho_best = 0.0
        for _ in range(n_restarts):
            v = np.random.randn(N)
            v /= np.linalg.norm(v)
            rho = 0.0
            for _ in range(n_iter):
                Av = apply_fn(v)
                rho = np.linalg.norm(Av)
                if rho < 1e-14:
                    break
                v = Av / rho
            rho_best = max(rho_best, rho)
        return rho_best

    np.random.seed(42)
    rho_P      = power_iter(apply_P,      n_iter, n_restarts)
    np.random.seed(42)
    rho_Ptheta = power_iter(apply_Ptheta, n_iter, n_restarts)

    return rho_P, rho_Ptheta

# ==========================================
# SECTION 8: MAIN SWEEP
# Results are cached to disk so the plot can be
# updated without rerunning the computation.
# ==========================================

import os, json

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          f'spectral_results_Nx{Nx}_Nt{Nt}.json')

if os.path.exists(CACHE_FILE):
    print(f"Loading cached results from {CACHE_FILE}")
    with open(CACHE_FILE) as f:
        results = json.load(f)
else:
    results = []
    U_prev, M_prev, nu_prev = None, None, None

    for nu in NU_LIST:
        print(f"\n{'='*60}")
        print(f"  nu = {nu}")
        print(f"{'='*60}")
        t0 = time.time()

        U_star, M_star, conv = newton_picard(nu, U_init=U_prev, M_init=M_prev)
        solver_used = 'NP'
        picard_iters = getattr(newton_picard, '_last_iters', np.nan)

        if not conv:
            print(f"    Newton-Picard failed, trying Global Newton...")
            U_warm = U_prev if U_prev is not None else np.tile(uT, (Nt+1, 1))
            M_warm = M_prev if M_prev is not None else np.tile(m0, (Nt+1, 1))
            U_star, M_star, conv = global_newton_with_continuation(nu, U_warm, M_warm)
            solver_used = 'GN'
            picard_iters = float('nan')

        if not conv:
            print(f"  SKIP: both solvers failed at nu={nu}")
            results.append([nu, None, None, 'failed', None])
            continue

        print(f"  Solver: {solver_used} | Picard iters: {picard_iters} | Wall time: {time.time()-t0:.2f}s")
        U_prev, M_prev, nu_prev = U_star, M_star, nu

        print(f"  Computing spectral radius numerically...")
        t_rho = time.time()
        rho_P, rho_Ptheta = spectral_radius_numerical(U_star, M_star, nu, damping=DAMPING)
        print(f"  rho(P)       = {rho_P:.6f}  ({time.time()-t_rho:.1f}s)")
        print(f"  rho(P_theta) = {rho_Ptheta:.6f}  [theta={DAMPING}]")

        results.append([nu, rho_P, rho_Ptheta, solver_used, int(picard_iters) if not np.isnan(picard_iters) else None])

    with open(CACHE_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {CACHE_FILE}")

# ==========================================
# SECTION 9: PLOTS — two separate figures
# ==========================================

nu_vals    = np.array([r[0] for r in results], dtype=float)
rho_P_vals = np.array([r[1] if r[1] is not None else np.nan for r in results], dtype=float)
rho_Pt_vals= np.array([r[2] if r[2] is not None else np.nan for r in results], dtype=float)
solvers    = [r[3] for r in results]

plot_mask   = nu_vals <= 0.3
nu_plot     = nu_vals[plot_mask]
rho_P_plot  = rho_P_vals[plot_mask]
rho_Pt_plot = rho_Pt_vals[plot_mask]

def make_panel(nu, rho_vals, ylabel, save_path):
    valid = ~np.isnan(rho_vals)
    nv = nu[valid]
    rv = rho_vals[valid]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(nv, rv, 'k-', lw=1.2, alpha=0.4, zorder=1)
    for n, r in zip(nv, rv):
        ax.scatter(n, r, color='steelblue', s=100, zorder=5,
                   edgecolors='k', linewidths=0.5)
        offset = 12 if r < 1.0 else -18
        ax.annotate(f'{r:.3f}', xy=(n, r),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', fontsize=8.5, color='steelblue')
    ax.axhline(1.0, color='red', ls='--', lw=1.8, zorder=3)
    ax.set_xlabel(r'Viscosity $\nu$', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.invert_xaxis()
    ax.set_ylim(0.0, 2.25)
    ax.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))

make_panel(
    nu_plot, rho_P_plot,
    r'Spectral radius $\rho(P)$',
    os.path.join(script_dir, 'rho_P.png')
)

make_panel(
    nu_plot, rho_Pt_plot,
    r'Spectral radius $\rho(P_\theta)$',
    os.path.join(script_dir, 'rho_P_theta.png')
)

# --- Summary table ---
print("\n" + "="*65)
print(f"{'nu':>8} | {'rho(P)':>10} | {'rho(P_th)':>10} | {'Picard':>7} | solver")
print("-"*65)
for r in results:
    nu_v, rho_p, rho_pt, s = r[0], r[1], r[2], r[3]
    pic = r[4] if len(r) > 4 else None
    rho_p_f  = float(rho_p)  if rho_p  is not None else float('nan')
    rho_pt_f = float(rho_pt) if rho_pt is not None else float('nan')
    flag_p  = " *" if (rho_p  is not None and rho_p_f  > 1.0) else "  "
    flag_pt = " *" if (rho_pt is not None and rho_pt_f > 1.0) else "  "
    rho_p_str  = f"{rho_p_f:>10.4f}"  if rho_p  is not None else f"{'---':>10}"
    rho_pt_str = f"{rho_pt_f:>10.4f}" if rho_pt is not None else f"{'---':>10}"
    pic_str = f"{int(pic):>7}" if pic is not None else "   FAIL"
    print(f"{nu_v:>8.3f} | {rho_p_str}{flag_p} | {rho_pt_str}{flag_pt} | {pic_str} | {s}")
print("="*65)
print("  * denotes rho > 1")
