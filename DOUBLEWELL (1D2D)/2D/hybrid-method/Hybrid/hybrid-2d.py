"""Standalone hybrid solver for the 2D double-well MFG (Algorithm 4).

Single self-contained file. The Newton-Picard (Type 2) and Global Newton
(Type 1) solvers are inlined below -- taken verbatim (minus plotting) from
doublewell-2d-picard.py and doublewell-2d-newton.py, the codes that
produced Tables 4 and 5 -- followed by the hybrid driver. No external solver
modules are imported, so this file runs on its own.

------------------------------------------------------------------------------
Phase accounting (the four labeled pieces)
------------------------------------------------------------------------------
Algorithm 4 has three phases, but its Phase 1 (the Newton-Picard descent) does
two timing-distinct things, so the run is broken into four labeled phases:

  Phase 1 : Newton-Picard descent to nu_limit.
            All NP work (theta retries included) for the viscosity levels that
            converge, down to nu_limit = the lowest nu NP can reach on this grid.

  Phase 2 : Newton-Picard failure at the next level.
            NP attempts the next viscosity below nu_limit, walking the sticky
            theta schedule down to theta_floor, and fails every time. This is
            the work that triggers the handoff (reported as "-" in Table 4).

  Phase 3 : Global Newton handoff re-solve at nu_limit.
            One Global Newton solve at nu_limit, receiving the converged
            NP solution; drives the residual to machine precision.

  Phase 4 : Global Newton continuation to nu_target.
            Global Newton continuation from nu_limit down to
            nu_target, hitting every level in between (including the one NP
            failed at). Equals the Table 5 cumulative-time difference
            cum(nu_target) - cum(nu_limit).

Phases 1 and 2 are the same loop in Algorithm 4 (the NP descent); they are split
only so the timing breakdown matches the four-step reasoning. If NP reaches
nu_target unaided, Phases 2-4 are skipped (pure Newton-Picard).

Note on Phase 4 / Algorithm 4: when NP fails, the loop variable nu points at the
FAILED viscosity. Phase 4 resets nu <- nu_limit before the continuation loop so
the level NP failed at is re-attempted by Global Newton rather than skipped
(the corrected Phase 3 of Algorithm 4).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================================
# Newton-Picard (Type 2) solver -- inlined from doublewell-2d-picard.py
# ============================================================================
def solve_np(
    nu=0.1,
    Nx=50,
    Ny=50,
    Nt=100,
    T=1.0,
    xmin=-2.5,
    xmax=2.5,
    ymin=-2.5,
    ymax=2.5,
    coupling_str=1.0,
    tilt=0.3,
    well_depth=2.0,
    alpha=2,
    picard_max_iter=50,
    residual_tol=1e-6,
    damping=0.8,
    newton_max_iter=20,
    newton_tol=1e-9,
    initial_guess=None,
    stall_patience=None,
    linear_solver="direct",
    log_file=None,
    verbose=True,
):
    """Newton-Picard solver for the 2D double-well MFG (periodic BCs).

    Returns a dict:
      status        : 'converged' | 'maxiter' | 'stalled' | 'diverged'
      residual      : final inf-norm residual (HJB+FP combined)
      picard_iters  : number of Picard iterations performed
      time          : total wall-clock seconds
      U, M          : final flows, shape (Nt+1, Nx*Ny) (row-major flatten of (Ny, Nx))
      history       : per-iter list of dicts {iter, residual, r_hjb, r_fp, newton_iters, time}
    """
    # --- Derived quantities ---
    dt = T / Nt
    Dx = (xmax - xmin) / Nx
    Dy = (ymax - ymin) / Ny
    x = np.linspace(xmin, xmax, Nx, endpoint=False)
    y = np.linspace(ymin, ymax, Ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    num_x = Nx * Ny

    # --- Physics ---
    V_2d = well_depth * (xx**2 - 1.0)**2 + tilt * xx + 0.5 * yy**2
    V_flat = V_2d.flatten()
    m0_2d = np.exp(-(xx**2 + yy**2) / (2 * 0.6**2))
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

    # --- Discrete operators (periodic, 2D via Kronecker) ---
    I_x = sp.eye(Nx)
    I_y = sp.eye(Ny)
    I_op = sp.eye(num_x)

    data_p_x = np.array([-np.ones(Nx), np.ones(Nx)])
    Dp_x_1d = sp.spdiags(data_p_x, [0, 1], Nx, Nx).tolil()
    Dp_x_1d[-1, -1] = -1.0
    Dp_x_1d[-1, 0] = 1.0
    Dp_x_1d = Dp_x_1d.tocsr() / Dx

    Dm_x_1d = sp.spdiags(data_p_x, [-1, 0], Nx, Nx).tolil()
    Dm_x_1d[0, 0] = 1.0
    Dm_x_1d[0, -1] = -1.0
    Dm_x_1d = Dm_x_1d.tocsr() / Dx

    D_plus_x = sp.kron(I_y, Dp_x_1d)
    D_minus_x = sp.kron(I_y, Dm_x_1d)

    data_p_y = np.array([-np.ones(Ny), np.ones(Ny)])
    Dp_y_1d = sp.spdiags(data_p_y, [0, 1], Ny, Ny).tolil()
    Dp_y_1d[-1, -1] = -1.0
    Dp_y_1d[-1, 0] = 1.0
    Dp_y_1d = Dp_y_1d.tocsr() / Dy

    Dm_y_1d = sp.spdiags(data_p_y, [-1, 0], Ny, Ny).tolil()
    Dm_y_1d[0, 0] = 1.0
    Dm_y_1d[0, -1] = -1.0
    Dm_y_1d = Dm_y_1d.tocsr() / Dy

    D_plus_y = sp.kron(Dp_y_1d, I_x)
    D_minus_y = sp.kron(Dm_y_1d, I_x)

    Laplacian = D_plus_x @ D_minus_x + D_plus_y @ D_minus_y
    A_diff = (I_op - dt * nu * Laplacian).tocsr()

    # --- Linear-solve wrapper ---
    def lin_solve(A, b):
        if linear_solver != "direct":
            raise ValueError("This benchmark requires direct linear solves.")
        return spla.spsolve(A, b)

    # --- Coupled MFG residual ---
    def compute_residual(U, M):
        r_hjb_max = 0.0
        r_fp_max = 0.0
        for n in range(Nt):
            dpx = D_plus_x @ U[n]
            dmx = D_minus_x @ U[n]
            dpy = D_plus_y @ U[n]
            dmy = D_minus_y @ U[n]
            H_total = discrete_Hamiltonian(dpx, dmx, dpy, dmy, compute_f(M[n + 1]))
            R_HJB = A_diff @ U[n] + dt * H_total - U[n + 1]
            r_hjb_max = max(r_hjb_max, np.linalg.norm(R_HJB, np.inf))

            J_H = (sp.diags(dH_dp(np.minimum(dpx, 0))) @ D_plus_x +
                   sp.diags(dH_dp(np.maximum(dmx, 0))) @ D_minus_x +
                   sp.diags(dH_dp(np.minimum(dpy, 0))) @ D_plus_y +
                   sp.diags(dH_dp(np.maximum(dmy, 0))) @ D_minus_y).tocsr()
            A_FP = (I_op - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
            R_FP = A_FP @ M[n + 1] - M[n]
            r_fp_max = max(r_fp_max, np.linalg.norm(R_FP, np.inf))
        return max(r_hjb_max, r_fp_max), r_hjb_max, r_fp_max

    # --- HJB / FP solvers ---
    def solve_hjb_backward(M_flow):
        U = np.zeros((Nt + 1, num_x))
        U[Nt] = uT_flat
        total_newton = 0
        for n in range(Nt - 1, -1, -1):
            u_next = U[n + 1]
            u_curr = u_next.copy()
            f_val = compute_f(M_flow[n + 1])
            for _ in range(newton_max_iter):
                total_newton += 1
                dpx, dmx = D_plus_x @ u_curr, D_minus_x @ u_curr
                dpy, dmy = D_plus_y @ u_curr, D_minus_y @ u_curr
                H_total = discrete_Hamiltonian(dpx, dmx, dpy, dmy, f_val)
                F = A_diff @ u_curr + dt * H_total - u_next
                if np.linalg.norm(F, np.inf) < newton_tol:
                    break
                J = A_diff + dt * (
                    sp.diags(dH_dp(np.minimum(dpx, 0))) @ D_plus_x
                    + sp.diags(dH_dp(np.maximum(dmx, 0))) @ D_minus_x
                    + sp.diags(dH_dp(np.minimum(dpy, 0))) @ D_plus_y
                    + sp.diags(dH_dp(np.maximum(dmy, 0))) @ D_minus_y
                )
                u_curr -= lin_solve(J, F)
            U[n] = u_curr
        return U, total_newton

    def solve_fp_forward(U_flow):
        M = np.zeros((Nt + 1, num_x))
        M[0] = m0_flat
        for n in range(Nt):
            dpx, dmx = D_plus_x @ U_flow[n], D_minus_x @ U_flow[n]
            dpy, dmy = D_plus_y @ U_flow[n], D_minus_y @ U_flow[n]
            J_H = (sp.diags(dH_dp(np.minimum(dpx, 0))) @ D_plus_x +
                   sp.diags(dH_dp(np.maximum(dmx, 0))) @ D_minus_x +
                   sp.diags(dH_dp(np.minimum(dpy, 0))) @ D_plus_y +
                   sp.diags(dH_dp(np.maximum(dmy, 0))) @ D_minus_y).tocsr()
            A = (I_op - dt * nu * Laplacian + dt * J_H.transpose()).tocsr()
            M[n + 1] = lin_solve(A, M[n])
        return M

    # --- Initialization ---
    start_time = time.time()
    if initial_guess is not None:
        U_init, M_init = initial_guess
        U_flow = U_init.copy()
        M_flow = M_init.copy()
    else:
        # Default: M is heat-flow from m0; U starts at zero
        M_flow = np.tile(m0_flat, (Nt + 1, 1))
        A_diff_only = I_op - dt * nu * Laplacian
        for n in range(Nt):
            M_flow[n + 1] = lin_solve(A_diff_only.tocsr(), M_flow[n])
        U_flow = np.zeros((Nt + 1, num_x))

    log_fp = open(log_file, "w") if log_file else None
    if log_fp or verbose:
        header = (
            f"{'='*118}\n"
            f"   DOUBLE WELL MFG 2D (NEWTON TYPE 2: NEWTON-PICARD)\n"
            f"{'='*118}\n"
            f"Parameters:\n"
            f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
            f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
            f"  nu   = {nu}, alpha = {alpha}, kappa = {coupling_str}\n"
            f"Solver Parameters:\n"
            f"  picard_max_iter = {picard_max_iter}, residual_tol = {residual_tol}\n"
            f"  damping = {damping}, linear_solver = {linear_solver}\n"
            f"  newton_max_iter = {newton_max_iter}, newton_tol = {newton_tol}\n"
            f"Grid Info:\n"
            f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
            f"{'-'*118}\n"
            f"{'Iter':<5} | {'Residual':<12} | {'R_HJB':<12} | {'R_FP':<12} | {'Newton It':<10} | {'Time (s)':<10}\n"
            f"{'-'*118}\n"
        )
        if verbose:
            print(header, end="")
        if log_fp:
            log_fp.write(header)

    history = []
    status = "maxiter"
    res_total = float("inf")
    best_res = float("inf")
    iters_since_best = 0
    k = 0

    for k in range(1, picard_max_iter + 1):
        t0 = time.time()
        try:
            U_candidate, n_iters = solve_hjb_backward(M_flow)
            U_flow_new = damping * U_candidate + (1 - damping) * U_flow
            M_candidate = solve_fp_forward(U_flow_new)
            M_flow_new = damping * M_candidate + (1 - damping) * M_flow

            res_total, res_hjb, res_fp = compute_residual(U_flow_new, M_flow_new)
            if not np.isfinite(res_total):
                status = "diverged"
                msg = f"DIVERGED at iter {k}: non-finite residual\n"
                if verbose:
                    print(msg, end="")
                if log_fp:
                    log_fp.write(msg)
                break
        except Exception as e:
            status = "diverged"
            msg = f"DIVERGED at iter {k}: {e}\n"
            if verbose:
                print(msg, end="")
            if log_fp:
                log_fp.write(msg)
            break

        iter_time = time.time() - t0
        history.append({"iter": k, "residual": res_total, "r_hjb": res_hjb,
                        "r_fp": res_fp, "newton_iters": n_iters, "time": iter_time})
        log_str = (f"{k:<5} | {res_total:.4e}   | {res_hjb:.4e}   | "
                   f"{res_fp:.4e}   | {n_iters:<10} | {iter_time:.4f}")
        if verbose:
            print(log_str)
        if log_fp:
            log_fp.write(log_str + "\n")

        U_flow, M_flow = U_flow_new, M_flow_new

        # Stall detection
        if res_total < best_res:
            best_res = res_total
            iters_since_best = 0
        else:
            iters_since_best += 1
        if stall_patience is not None and iters_since_best >= stall_patience:
            status = "stalled"
            msg = (f"STALLED at iter {k}: no improvement on best residual "
                   f"({best_res:.4e}) for {iters_since_best} iters.\n")
            if verbose:
                print(msg, end="")
            if log_fp:
                log_fp.write(msg)
            break

        if res_total < residual_tol:
            status = "converged"
            conv_msg = (f"{'-'*118}\nCONVERGED at nu={nu} in {k} iterations "
                        f"(residual={res_total:.4e}).\n")
            if verbose:
                print(conv_msg)
            if log_fp:
                log_fp.write(conv_msg)
            break

    total_time = time.time() - start_time
    time_msg = f"Total Execution Time: {total_time:.4f} seconds.\n"
    if verbose:
        print(time_msg)
    if log_fp:
        log_fp.write(time_msg)
        log_fp.close()

    return {
        "status": status,
        "residual": res_total,
        "picard_iters": k,
        "time": total_time,
        "U": U_flow,
        "M": M_flow,
        "history": history,
    }


# ============================================================================
# Global Newton (Type 1) solver -- inlined from doublewell-2d-newton.py
# ============================================================================
def solve_gn(
    nu_target=0.1,
    Nx=50,
    Ny=50,
    Nt=50,
    T=1.0,
    xmin=-2.5,
    xmax=2.5,
    ymin=-2.5,
    ymax=2.5,
    coupling_str=1.0,
    tilt=0.3,
    well_depth=2.0,
    alpha=2,
    newton_max_iter=20,
    newton_tol=1e-9,
    use_continuation=True,
    nu_start=0.5,
    tau=0.75,
    gamma=0.5,
    cont_tol=1e-4,
    initial_guess=None,
    linear_solver="direct",
    log_file=None,
    verbose=True,
):
    """Global Newton on the stacked (HJB; FP) space-time system for 2D MFG, with viscosity continuation.

    Returns a dict:
      status         : 'converged' | 'failed'
      residual       : final inf-norm of stacked discrete residual
      time           : total wall-clock seconds
      newton_iters   : total Newton iterations across all continuation steps
      n_cont_steps   : number of continuation steps taken (including initial solve)
      n_backtracks   : number of backtrack attempts triggered
      nu_reached     : the lowest nu at which Newton converged
      U, M           : final flows, shape (Nt+1, Nx*Ny)
    """
    if linear_solver != "direct":
        raise ValueError("This benchmark requires direct linear solves.")

    # --- Derived ---
    dt = T / Nt
    Dx = (xmax - xmin) / Nx
    Dy = (ymax - ymin) / Ny
    x = np.linspace(xmin, xmax, Nx, endpoint=False)
    y = np.linspace(ymin, ymax, Ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    num_x = Nx * Ny

    # --- Physics ---
    V_2d = well_depth * (xx**2 - 1.0)**2 + tilt * xx + 0.5 * yy**2
    V_flat = V_2d.flatten()
    m0_2d = np.exp(-(xx**2 + yy**2) / (2 * 0.6**2))
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

    # --- Discrete operators (periodic, 2D via Kronecker) ---
    I_x = sp.eye(Nx)
    I_y = sp.eye(Ny)
    I_op = sp.eye(num_x)

    data_p_x = np.array([-np.ones(Nx), np.ones(Nx)])
    Dp_x_1d = sp.spdiags(data_p_x, [0, 1], Nx, Nx).tolil()
    Dp_x_1d[-1, -1] = -1.0
    Dp_x_1d[-1, 0] = 1.0
    Dp_x_1d = Dp_x_1d.tocsr() / Dx

    Dm_x_1d = sp.spdiags(data_p_x, [-1, 0], Nx, Nx).tolil()
    Dm_x_1d[0, 0] = 1.0
    Dm_x_1d[0, -1] = -1.0
    Dm_x_1d = Dm_x_1d.tocsr() / Dx

    D_plus_x = sp.kron(I_y, Dp_x_1d).tocsr()
    D_minus_x = sp.kron(I_y, Dm_x_1d).tocsr()

    data_p_y = np.array([-np.ones(Ny), np.ones(Ny)])
    Dp_y_1d = sp.spdiags(data_p_y, [0, 1], Ny, Ny).tolil()
    Dp_y_1d[-1, -1] = -1.0
    Dp_y_1d[-1, 0] = 1.0
    Dp_y_1d = Dp_y_1d.tocsr() / Dy

    Dm_y_1d = sp.spdiags(data_p_y, [-1, 0], Ny, Ny).tolil()
    Dm_y_1d[0, 0] = 1.0
    Dm_y_1d[0, -1] = -1.0
    Dm_y_1d = Dm_y_1d.tocsr() / Dy

    D_plus_y = sp.kron(Dp_y_1d, I_x).tocsr()
    D_minus_y = sp.kron(Dm_y_1d, I_x).tocsr()

    Laplacian = D_plus_x @ D_minus_x + D_plus_y @ D_minus_y

    # --- Linear-solve wrapper ---
    def lin_solve(A, b):
        return spla.spsolve(A, b)

    # --- Algorithm A (Achdou & Perez 2012): Schur-complement bicgstab preconditioned by A_MM ---
    # Solves the GN Jacobian system via:
    #   1. U_tilde = A_UU^-1 * F_U  (backward time march, one 2D spatial solve per step)
    #   2. bicgstab on  (A_MM - A_MU * A_UU^-1 * A_UM) * dM = F_M - A_MU * U_tilde
    #      preconditioned by A_MM (forward time march). The preconditioned matrix
    #      I - A_MM^-1 * A_MU * A_UU^-1 * A_UM is a compact perturbation of identity,
    #      so iteration count is grid-independent.
    #   3. dU = U_tilde - A_UU^-1 * A_UM * dM (one backward march for the correction).
    def lin_solve_schur(J_UU, J_UM, J_MU, J_MM, F):
        inv_dt = 1.0 / dt
        # Pre-factorize per-time-step spatial blocks
        D_lu = [spla.splu(J_UU[n][n].tocsc()) for n in range(Nt)]
        E_lu = [spla.splu(J_MM[n + 1][n + 1].tocsc()) for n in range(Nt)]
        # Pull out dH_dm diagonals and E_tilde sparse blocks
        dH_dm = [np.asarray(J_UM[n][n + 1].diagonal()) for n in range(Nt)]
        E_tilde = [J_MU[n + 1][n] for n in range(Nt)]
        E_diag = [J_MM[n + 1][n + 1] for n in range(Nt)]  # for forward A_MM apply

        F_U = F[:(Nt + 1) * num_x].reshape((Nt + 1, num_x))
        F_M = F[(Nt + 1) * num_x:].reshape((Nt + 1, num_x))

        def apply_AUU_inv(f):
            """Solve A_UU * U = f via backward time march. f, U shape (Nt+1, num_x)."""
            U_out = np.empty((Nt + 1, num_x))
            U_out[Nt] = f[Nt]
            for n in range(Nt - 1, -1, -1):
                rhs = f[n] + inv_dt * U_out[n + 1]
                U_out[n] = D_lu[n].solve(rhs)
            return U_out

        def apply_AMM_inv(g):
            """Solve A_MM * M = g via forward time march."""
            M_out = np.empty((Nt + 1, num_x))
            M_out[0] = g[0]
            for n in range(1, Nt + 1):
                rhs = g[n] + inv_dt * M_out[n - 1]
                M_out[n] = E_lu[n - 1].solve(rhs)
            return M_out

        def apply_AUM(v):
            """Apply A_UM to v (in M-space). J_UM[n][n+1] = diag(dH_dm[n]) for n<Nt."""
            out = np.zeros((Nt + 1, num_x))
            for n in range(Nt):
                out[n] = dH_dm[n] * v[n + 1]
            return out

        def apply_AMU(v):
            """Apply A_MU to v (in U-space). J_MU[n+1][n] = E_tilde[n] for n<Nt."""
            out = np.zeros((Nt + 1, num_x))
            for n in range(Nt):
                out[n + 1] = E_tilde[n] @ v[n]
            return out

        def apply_AMM(v):
            """Apply A_MM to v. Row 0 is identity; rows n>0 have E_diag[n-1]*v[n] - (1/dt)*v[n-1]."""
            out = np.empty((Nt + 1, num_x))
            out[0] = v[0]
            for n in range(Nt):
                out[n + 1] = E_diag[n] @ v[n + 1] - inv_dt * v[n]
            return out

        # Step 1: U_tilde = A_UU^-1 * F_U
        U_tilde = apply_AUU_inv(F_U)

        # Step 2: Schur system via preconditioned bicgstab
        rhs_M = (F_M - apply_AMU(U_tilde)).flatten()
        N_M = (Nt + 1) * num_x

        def schur_matvec(v_flat):
            v = v_flat.reshape((Nt + 1, num_x))
            tmp = apply_AMM(v) - apply_AMU(apply_AUU_inv(apply_AUM(v)))
            return tmp.flatten()

        def precond(r_flat):
            return apply_AMM_inv(r_flat.reshape((Nt + 1, num_x))).flatten()

        S_op = spla.LinearOperator((N_M, N_M), matvec=schur_matvec)
        P_op = spla.LinearOperator((N_M, N_M), matvec=precond)

        dM_flat, info = spla.bicgstab(S_op, rhs_M, M=P_op, atol=1e-12, tol=1e-3, maxiter=100)
        dM = dM_flat.reshape((Nt + 1, num_x))

        # Step 3: dU = U_tilde - A_UU^-1 * A_UM * dM
        dU = U_tilde - apply_AUU_inv(apply_AUM(dM))

        return np.concatenate([dU.flatten(), dM.flatten()])

    # --- One full Newton solve at fixed nu ---
    def newton_step(nu_val, W_guess):
        if W_guess is None:
            U = np.tile(uT_flat, (Nt + 1, 1))
            M = np.tile(m0_flat, (Nt + 1, 1))
        else:
            U = W_guess[:(Nt + 1) * num_x].reshape((Nt + 1, num_x))
            M = W_guess[(Nt + 1) * num_x:].reshape((Nt + 1, num_x))

        n_iters_used = 0
        res = float("inf")

        for it in range(newton_max_iter):
            n_iters_used += 1

            J_UU = [[None for _ in range(Nt + 1)] for _ in range(Nt + 1)]
            J_UM = [[None for _ in range(Nt + 1)] for _ in range(Nt + 1)]
            J_MU = [[None for _ in range(Nt + 1)] for _ in range(Nt + 1)]
            J_MM = [[None for _ in range(Nt + 1)] for _ in range(Nt + 1)]

            F_U = np.zeros((Nt + 1, num_x))
            F_M = np.zeros((Nt + 1, num_x))

            J_UU[Nt][Nt] = I_op.tocsr()
            F_U[Nt] = U[Nt] - uT_flat
            J_UM[Nt][Nt] = sp.csr_matrix((num_x, num_x))

            J_MM[0][0] = I_op.tocsr()
            F_M[0] = M[0] - m0_flat
            J_MU[0][0] = J_MU[Nt][Nt] = sp.csr_matrix((num_x, num_x))

            for n in range(Nt):
                un, un_next = U[n], U[n + 1]
                mn, mn_next = M[n], M[n + 1]
                dpx, dmx = D_plus_x @ un, D_minus_x @ un
                dpy, dmy = D_plus_y @ un, D_minus_y @ un
                p_min_x, p_max_x = np.minimum(dpx, 0), np.maximum(dmx, 0)
                p_min_y, p_max_y = np.minimum(dpy, 0), np.maximum(dmy, 0)
                f_val = compute_f(mn_next)

                F_U[n] = ((un - un_next) / dt - nu_val * Laplacian @ un
                          + discrete_Hamiltonian(dpx, dmx, dpy, dmy, f_val))

                dH_dm = -coupling_str * alpha * (np.maximum(mn_next, 0)**(alpha - 1))
                J_UU[n][n] = ((1.0 / dt) * I_op - nu_val * Laplacian
                              + sp.diags(dH_dp(p_min_x)) @ D_plus_x
                              + sp.diags(dH_dp(p_max_x)) @ D_minus_x
                              + sp.diags(dH_dp(p_min_y)) @ D_plus_y
                              + sp.diags(dH_dp(p_max_y)) @ D_minus_y).tocsr()
                J_UU[n][n + 1] = (-(1.0 / dt) * I_op).tocsr()
                J_UM[n][n] = sp.csr_matrix((num_x, num_x))
                J_UM[n][n + 1] = sp.diags(dH_dm).tocsr()

                F_M[n + 1] = ((mn_next - mn) / dt - nu_val * Laplacian @ mn_next
                              - (D_minus_x @ (mn_next * p_min_x) + D_plus_x @ (mn_next * p_max_x)
                                 + D_minus_y @ (mn_next * p_min_y) + D_plus_y @ (mn_next * p_max_y)))

                J_MM[n + 1][n + 1] = ((1.0 / dt) * I_op - nu_val * Laplacian
                                      - (D_minus_x @ sp.diags(p_min_x) + D_plus_x @ sp.diags(p_max_x)
                                         + D_minus_y @ sp.diags(p_min_y) + D_plus_y @ sp.diags(p_max_y))).tocsr()
                J_MM[n + 1][n] = (-(1.0 / dt) * I_op).tocsr()  # FIX: was -I (bug)
                J_MU[n + 1][n] = -(D_minus_x @ sp.diags(mn_next * (dpx < 0)) @ D_plus_x
                                   + D_plus_x @ sp.diags(mn_next * (dmx > 0)) @ D_minus_x
                                   + D_minus_y @ sp.diags(mn_next * (dpy < 0)) @ D_plus_y
                                   + D_plus_y @ sp.diags(mn_next * (dmy > 0)) @ D_minus_y).tocsr()

            F = np.concatenate([F_U.flatten(), F_M.flatten()])
            res = np.linalg.norm(F, np.inf)
            J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)], [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()

            if not np.isfinite(res):
                return np.concatenate([U.flatten(), M.flatten()]), False, res, n_iters_used
            if res < newton_tol:
                iter_log = f"      Iter {it:2d} | res = {res:.4e} (Converged)"
                if verbose:
                    print(iter_log, flush=True)
                if log_fp:
                    log_fp.write(iter_log + "\n"); log_fp.flush()
                return np.concatenate([U.flatten(), M.flatten()]), True, res, n_iters_used

            try:
                t_solve = time.time()
                dW = lin_solve(J, F)
                t_solve = time.time() - t_solve
                iter_log = f"      Iter {it:2d} | res = {res:.4e} | Solve Time: {t_solve:.3f}s"
                if verbose:
                    print(iter_log, flush=True)
                if log_fp:
                    log_fp.write(iter_log + "\n"); log_fp.flush()
                U = (U.flatten() - dW[:(Nt + 1) * num_x]).reshape((Nt + 1, num_x))
                M = (M.flatten() - dW[(Nt + 1) * num_x:]).reshape((Nt + 1, num_x))
            except Exception as exc:
                err_msg = f"      Iter {it:2d} | solve exception: {type(exc).__name__}: {exc}"
                if verbose:
                    print(err_msg, flush=True)
                if log_fp:
                    log_fp.write(err_msg + "\n"); log_fp.flush()
                return np.concatenate([U.flatten(), M.flatten()]), False, res, n_iters_used

        return np.concatenate([U.flatten(), M.flatten()]), False, res, n_iters_used

    # --- Driver: optional viscosity continuation ---
    log_fp = open(log_file, "w") if log_file else None
    if log_fp or verbose:
        header = (
            f"{'='*118}\n"
            f"   DOUBLE WELL MFG 2D (NEWTON TYPE 1: GLOBAL NEWTON)\n"
            f"{'='*118}\n"
            f"Parameters:\n"
            f"  xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}\n"
            f"  T    = {T}, Nx = {Nx}, Ny = {Ny}, Nt = {Nt}\n"
            f"  nu_target = {nu_target}, alpha = {alpha}, kappa = {coupling_str}\n"
            f"Solver Parameters:\n"
            f"  newton_max_iter = {newton_max_iter}, newton_tol = {newton_tol}\n"
            f"  continuation = {use_continuation}, nu_start = {nu_start}, tau = {tau}, gamma = {gamma}, cont_tol = {cont_tol}\n"
            f"  linear_solver = {linear_solver}\n"
            f"Grid Info:\n"
            f"  dt = {dt:.6f}, dx = {Dx:.6f}, dy = {Dy:.6f}\n"
            f"{'-'*118}\n"
        )
        if verbose:
            print(header, end="")
        if log_fp:
            log_fp.write(header)

    t0 = time.time()
    total_newton = 0
    n_cont_steps = 0
    n_backtracks = 0
    nu_reached = None
    W = None
    final_res = float("inf")

    nu_curr = nu_start if use_continuation else nu_target
    n_cont_steps += 1
    init_msg = "base initialization" if initial_guess is None else "provided initialization"
    msg = f"First solve at nu = {nu_curr:.4g} ({init_msg})\n"
    if verbose:
        print(msg, end="")
    if log_fp:
        log_fp.write(msg)
    W, conv, res, niter = newton_step(nu_curr, initial_guess)
    total_newton += niter
    if not conv:
        total_time = time.time() - t0
        if log_fp:
            log_fp.write(f"Initial solve failed at nu={nu_curr} (res={res:.4e}).\n")
            log_fp.close()
        return {
            "status": "failed", "residual": res, "time": total_time,
            "newton_iters": total_newton, "n_cont_steps": n_cont_steps,
            "n_backtracks": n_backtracks, "nu_reached": None,
            "U": None, "M": None,
        }
    nu_reached = nu_curr
    final_res = res
    msg = f"  -> converged (res={res:.4e}, iters={niter})\n"
    if verbose:
        print(msg, end="")
    if log_fp:
        log_fp.write(msg)

    if use_continuation:
        while nu_curr > nu_target:
            nu_next = max(nu_target, tau * nu_curr)
            success = False
            while not success:
                n_cont_steps += 1
                msg = f"Attempt nu = {nu_next:.4g}\n"
                if verbose:
                    print(msg, end="")
                if log_fp:
                    log_fp.write(msg)
                W_trial, conv, res, niter = newton_step(nu_next, W)
                total_newton += niter
                if conv:
                    W, nu_curr, success = W_trial, nu_next, True
                    nu_reached = nu_curr
                    final_res = res
                    msg = f"  -> converged at nu={nu_next:.4g} (res={res:.4e}, iters={niter})\n"
                    if verbose:
                        print(msg, end="")
                    if log_fp:
                        log_fp.write(msg)
                else:
                    n_backtracks += 1
                    msg = f"  -> failed (res={res:.4e}); backtracking\n"
                    if verbose:
                        print(msg, end="")
                    if log_fp:
                        log_fp.write(msg)
                    nu_next = nu_curr - gamma * (nu_curr - nu_next)
                    if abs(nu_curr - nu_next) < cont_tol:
                        total_time = time.time() - t0
                        if log_fp:
                            log_fp.write("Continuation step size below cont_tol; giving up.\n")
                            log_fp.close()
                        return {
                            "status": "failed", "residual": final_res, "time": total_time,
                            "newton_iters": total_newton, "n_cont_steps": n_cont_steps,
                            "n_backtracks": n_backtracks, "nu_reached": nu_reached,
                            "U": None, "M": None,
                        }

    total_time = time.time() - t0
    if log_fp:
        log_fp.write(f"Total time: {total_time:.4f}s\n")
        log_fp.close()

    U_flow = W[:(Nt + 1) * num_x].reshape((Nt + 1, num_x))
    M_flow = W[(Nt + 1) * num_x:].reshape((Nt + 1, num_x))

    return {
        "status": "converged", "residual": final_res, "time": total_time,
        "newton_iters": total_newton, "n_cont_steps": n_cont_steps,
        "n_backtracks": n_backtracks, "nu_reached": nu_reached,
        "U": U_flow, "M": M_flow,
    }


_here = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Hybrid driver (Algorithm 4)
# ============================================================================
# --- Problem / discretization -------------------------------------------------
T = 1.0
NT = 10
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 16  # Nx = Ny (target grid Nh)

# Viscosity schedule (NextViscosity = next value down this list)
NUS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
NU_START = NUS[0]
NU_TARGET = NUS[-1]

# --- Phase 1/2: Newton-Picard sticky-theta schedule ---------------------------
THETA = 0.8
THETA_FLOOR = 0.05
THETA_STEP = 0.1
PICARD_MAX_ITER = 200
NP_RESIDUAL_TOL = 1e-6
STALL_PATIENCE = 10

# --- Phase 3/4: Global Newton -----------------------------------------------
NEWTON_MAX_ITER = 20
GN_NEWTON_TOL = 1e-6
# Direct sparse linear solves are used in every benchmark driver.
LINEAR_SOLVER_GN = "direct"

# Single output per run: a per-nu "cell" CSV, analogous to the paper's tables.
#   Phase 1 rows  -> Table 4 cells (NP per-cell time, = sum over theta retries)
#   Phase 4 rows  -> Table 5 cells (GN per-nu time; cumulative_time mirrors the
#                    cumulative column of Table 5).
# The filename carries Nh so runs at different grids never overwrite each other.
PER_NU_CSV = os.path.join(_here, f"hybrid2d_per_nu_times_Nh{NX}.csv")
CHECKPOINT = os.path.join(_here, f"hybrid2d_checkpoint_Nh{NX}.npz")
RESUME_FROM_CHECKPOINT = False
PER_NU_FIELDS = ["phase", "nu", "method", "n_attempts", "theta_used", "iters",
                 "status", "time_at_nu", "cumulative_time"]

PHASE_LABELS = {
    1: "Newton-Picard descent to nu_limit",
    2: "Newton-Picard fails at next nu (theta exhausted)",
    3: "Global Newton handoff re-solve at nu_limit",
    4: "Global Newton continuation to nu_target",
}


def _t1(x):
    """Round a time (seconds) to 1 decimal place."""
    return round(x, 1) if x is not None else None


def _r2(x):
    """Residual to 2 decimals in scientific notation (e.g. 1.13e-08)."""
    return float(f"{x:.2e}") if x is not None else None


def next_viscosity(nu):
    """Largest scheduled viscosity strictly below nu, or None if at the bottom."""
    below = [v for v in NUS if v < nu - 1e-15]
    return max(below) if below else None


def _np_to_flat(U, M):
    return np.concatenate([U.flatten(), M.flatten()])



def _write_csv(per_nu):
    """(Re)write the per-nu CSV from the current list of cells (cheap; <20 rows)."""
    with open(PER_NU_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_NU_FIELDS)
        w.writeheader()
        for c in per_nu:
            w.writerow(c)


def _save_checkpoint(stage, nu, theta, nu_limit, cum, phase_time, reached, per_nu, U, M):
    """Atomically persist the solution (U, M) and scalar bookkeeping."""
    meta = json.dumps({
        "stage": stage, "nu": nu, "theta": theta, "nu_limit": nu_limit,
        "cum": cum, "phase_time": phase_time, "reached": reached, "per_nu": per_nu,
    })
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, U=U, M=M, meta=np.array(meta))
    os.replace(tmp, CHECKPOINT)          # atomic: never leaves a half-written file


def _load_checkpoint():
    d = np.load(CHECKPOINT)
    meta = json.loads(d["meta"].item())
    return meta, d["U"], d["M"]


def main():
    t_wall0 = time.time()
    print(f"{'='*78}\n  HYBRID (Algorithm 4)  Nh={NX}  Nt={NT}  T={T}  "
          f"nu: {NU_START} -> {NU_TARGET}\n{'='*78}", flush=True)

    # ----- Benchmark runs always begin from the prescribed initial state ------
    resumed = False
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT):
        try:
            meta, U_ck, M_ck = _load_checkpoint()
            stage = meta["stage"]
            nu = meta["nu"]
            theta = meta["theta"]
            nu_limit = meta["nu_limit"]
            cum = meta["cum"]
            phase_time = {int(k): v for k, v in meta["phase_time"].items()}
            reached_by_np_alone = meta["reached"]
            per_nu = meta["per_nu"]
            W_np = (U_ck, M_ck)
            W_flat = _np_to_flat(U_ck, M_ck)
            resumed = True
            print(f"\n[resume] checkpoint found: stage={stage}, next nu={nu}, "
                  f"nu_limit={nu_limit}, {len(per_nu)} cells done, elapsed "
                  f"{cum:.1f}s. Continuing.", flush=True)
        except Exception as e:
            print(f"\n[resume] checkpoint unreadable ({e}); starting fresh.",
                  flush=True)
    if not resumed:
        stage = "np"
        nu = NU_START
        theta = THETA
        nu_limit = NU_START
        cum = 0.0
        phase_time = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        reached_by_np_alone = False
        per_nu = []
        W_np = None
        W_flat = None
        _write_csv(per_nu)

    nu_buffer = []                      # per-call rows for the current nu (NP)

    def flush_buffer(phase):
        for b in nu_buffer:
            phase_time[phase] += b["time"]
        if nu_buffer:
            last = nu_buffer[-1]
            per_nu.append({
                "phase": phase, "nu": nu_buffer[0]["nu"], "method": "NewtonPicard",
                "n_attempts": len(nu_buffer),
                "theta_used": round(last["theta"], 2) if last["status"] == "converged" else None,
                "iters": last["iters"], "status": last["status"],
                "time_at_nu": _t1(sum(b["time"] for b in nu_buffer)),
                "cumulative_time": _t1(cum),
            })
        nu_buffer.clear()

    def persist(stage_now, nu_next, theta_now, U, M):
        """After a converged step (or the handoff transition): rewrite the CSV
        and checkpoint the solution so we can resume from here."""
        _write_csv(per_nu)
        if U is not None and M is not None:
            _save_checkpoint(stage_now, nu_next, theta_now, nu_limit, cum,
                             phase_time, reached_by_np_alone, per_nu, U, M)

    # ===== Phases 1 & 2: Newton-Picard descent (Algorithm 4, Phase 1) =========
    if stage == "np":
        print("\n[Phase 1/2] Newton-Picard descent ...", flush=True)
    while stage == "np":
        W_prev = W_np
        r = solve_np(
            nu=nu, Nx=NX, Ny=NX, Nt=NT, T=T,
            damping=theta,
            picard_max_iter=PICARD_MAX_ITER,
            residual_tol=NP_RESIDUAL_TOL,
            stall_patience=STALL_PATIENCE,
            initial_guess=None,
            linear_solver="direct",
            log_file=None, verbose=False,
        )
        cum += r["time"]
        nu_buffer.append({
            "nu": nu, "method": "NewtonPicard", "theta": theta,
            "status": r["status"], "time": _t1(r["time"]),
            "cumulative_time": _t1(cum), "iters": r["picard_iters"],
            "residual": _r2(r["residual"]),
        })
        print(f"    nu={nu:<8} theta={theta:<4} -> {r['status']:<10} "
              f"iters={r['picard_iters']:<3} time={r['time']:.1f}s "
              f"res={r['residual']:.2e}", flush=True)

        if r["status"] == "converged":
            flush_buffer(1)
            nu_limit = nu
            W_np = (r["U"], r["M"])
            if nu <= NU_TARGET * (1.0 + 1e-9):
                reached_by_np_alone = True       # NP reached the target unaided
                stage = "done"
                persist("done", nu, theta, r["U"], r["M"])
            else:
                nu = next_viscosity(nu)
                persist("np", nu, theta, r["U"], r["M"])   # save after each nu
        else:
            W_np = W_prev                         # revert
            if theta <= THETA_FLOOR:
                flush_buffer(2)                   # terminal failure -> handoff
                stage = "handoff"
                U_h, M_h = (W_np if W_np is not None else (None, None))
                persist("handoff", nu, theta, U_h, M_h)
            else:
                theta = max(THETA_FLOOR, theta - THETA_STEP)

    if reached_by_np_alone:
        print(f"\nNewton-Picard reached nu_target={NU_TARGET} unaided; "
              f"Phases 3-4 skipped.", flush=True)

    # ===== Phase 3: Global Newton handoff re-solve at nu_limit =================
    if stage == "handoff":
        print(f"\n[Phase 3] Global Newton handoff re-solve at nu_limit={nu_limit} "
              f"...", flush=True)
        W_flat = _np_to_flat(*W_np) if W_np is not None else None
        r = solve_gn(
            nu_target=nu_limit, Nx=NX, Ny=NX, Nt=NT, T=T,
            newton_max_iter=NEWTON_MAX_ITER, newton_tol=GN_NEWTON_TOL,
            use_continuation=False, initial_guess=W_flat,
            linear_solver=LINEAR_SOLVER_GN,
            log_file=None, verbose=True,   # show each Newton iteration of the handoff
        )
        cum += r["time"]
        phase_time[3] += r["time"]
        per_nu.append({
            "phase": 3, "nu": nu_limit, "method": "GlobalNewton",
            "n_attempts": 1, "theta_used": None, "iters": r["newton_iters"],
            "status": r["status"], "time_at_nu": _t1(r["time"]),
            "cumulative_time": _t1(cum),
        })
        print(f"    nu={nu_limit:<8}       -> {r['status']:<10} "
              f"iters={r['newton_iters']:<3} time={r['time']:.1f}s "
              f"res={r['residual']:.2e}", flush=True)
        if r["status"] != "converged":
            print("\nHandoff re-solve FAILED; aborting before continuation.",
                  flush=True)
            stage = "done"
            _write_csv(per_nu)
        else:
            W_flat = _np_to_flat(r["U"], r["M"])
            nu = nu_limit
            stage = "gn"
            persist("gn", nu, theta, r["U"], r["M"])

    # ===== Phase 4: Global Newton continuation to nu_target ===================
    if stage == "gn":
        print(f"\n[Phase 4] Global Newton continuation {nu_limit} -> "
              f"{NU_TARGET} ...", flush=True)
        while nu > NU_TARGET:
            nu = next_viscosity(nu)
            r = solve_gn(
                nu_target=nu, Nx=NX, Ny=NX, Nt=NT, T=T,
                newton_max_iter=NEWTON_MAX_ITER, newton_tol=GN_NEWTON_TOL,
                use_continuation=False, initial_guess=W_flat,
                linear_solver=LINEAR_SOLVER_GN,
                log_file=None, verbose=True,   # show each Newton iteration of the continuation
            )
            cum += r["time"]
            phase_time[4] += r["time"]
            per_nu.append({
                "phase": 4, "nu": nu, "method": "GlobalNewton",
                "n_attempts": 1, "theta_used": None, "iters": r["newton_iters"],
                "status": r["status"], "time_at_nu": _t1(r["time"]),
                "cumulative_time": _t1(cum),
            })
            print(f"    nu={nu:<8}       -> {r['status']:<10} "
                  f"iters={r['newton_iters']:<3} time={r['time']:.1f}s "
                  f"res={r['residual']:.2e}", flush=True)
            if r["status"] != "converged":
                print("    continuation FAILED at this nu; stopping.", flush=True)
                _write_csv(per_nu)
                stage = "done"
                break
            W_flat = _np_to_flat(r["U"], r["M"])
            persist("gn", nu, theta, r["U"], r["M"])       # save after each nu
        stage = "done"

    # ===== Finalize: single CSV, drop the checkpoint (the run is over) ========
    _write_csv(per_nu)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    total = sum(phase_time.values())

    # ----- Console per-nu table -----------------------------------------------
    print(f"\n{'='*78}\n  PER-NU TIMES  (Table-4 cells = Phase 1 | "
          f"Table-5 cells = Phase 4)\n{'-'*78}", flush=True)
    print(f"  {'ph':<3}{'nu':<9}{'method':<14}{'theta':<7}{'iters':>6}  "
          f"{'status':<11}{'time(s)':>9}{'cum(s)':>10}", flush=True)
    for c in per_nu:
        th = "" if c["theta_used"] is None else round(c["theta_used"], 2)
        print(f"  {c['phase']:<3}{c['nu']:<9}{c['method']:<14}{str(th):<7}"
              f"{c['iters']:>6}  {c['status']:<11}{c['time_at_nu']:>9}"
              f"{c['cumulative_time']:>10}", flush=True)
    print(f"{'='*78}", flush=True)

    # ----- Console summary ----------------------------------------------------
    print(f"\n{'='*78}\n  PHASE TIMING SUMMARY  (Nh={NX})\n{'-'*78}", flush=True)
    for p in (1, 2, 3, 4):
        print(f"  Phase {p}  {PHASE_LABELS[p]:<48} {_t1(phase_time[p]):>10} s",
              flush=True)
    print(f"{'-'*78}\n  {'TOTAL':<57} {_t1(total):>10} s\n{'='*78}", flush=True)
    print(f"\n  nu_limit (last NP success) = {nu_limit}", flush=True)
    print(f"  Wrote {os.path.basename(PER_NU_CSV)}", flush=True)
    print(f"  Driver wall time: {time.time() - t_wall0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
