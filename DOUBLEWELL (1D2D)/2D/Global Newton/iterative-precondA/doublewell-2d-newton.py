import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time


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
    save_plot=None,
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
        if linear_solver == "direct":
            return spla.spsolve(A, b)
        else:
            ilu = spla.spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20)
            M_pre = spla.LinearOperator(A.shape, matvec=ilu.solve)
            sol, _ = spla.bicgstab(A, b, M=M_pre, atol=1e-10, rtol=1e-8, maxiter=200)
            return sol

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
            if linear_solver in ("direct", "iterative"):
                J = sp.bmat([[sp.bmat(J_UU), sp.bmat(J_UM)], [sp.bmat(J_MU), sp.bmat(J_MM)]]).tocsr()
            else:
                J = None  # not needed for "schur"

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
                if linear_solver == "schur":
                    dW = lin_solve_schur(J_UU, J_UM, J_MU, J_MM, F)
                else:
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
    init_msg = "naive" if initial_guess is None else "warm-start"
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

    if save_plot is not None:
        time_indices = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]
        time_labels = [0.0, T / 4, T / 2, 3 * T / 4, T]
        fig, axes = plt.subplots(2, 5, figsize=(25, 9))
        for i, (idx, t_val) in enumerate(zip(time_indices, time_labels)):
            M_snap = M_flow[idx].reshape((Ny, Nx))
            U_snap = U_flow[idx].reshape((Ny, Nx))
            im_m = axes[0, i].contourf(xx, yy, M_snap, levels=30, cmap="viridis")
            axes[0, i].set_title(f"Density (m) at t={t_val:.2f}")
            fig.colorbar(im_m, ax=axes[0, i])
            im_u = axes[1, i].contourf(xx, yy, U_snap, levels=30, cmap="plasma")
            axes[1, i].set_title(f"Value (u) at t={t_val:.2f}")
            fig.colorbar(im_u, ax=axes[1, i])
        plt.tight_layout()
        plt.savefig(save_plot, dpi=150)
        plt.close(fig)

    return {
        "status": "converged", "residual": final_res, "time": total_time,
        "newton_iters": total_newton, "n_cont_steps": n_cont_steps,
        "n_backtracks": n_backtracks, "nu_reached": nu_reached,
        "U": U_flow, "M": M_flow,
    }


if __name__ == "__main__":
    nu_target = 0.1
    Nx = Ny = 100
    Nt = 10
    T = 1.0
    result = solve_gn(
        nu_target=nu_target, Nx=Nx, Ny=Ny, Nt=Nt, T=T,
        linear_solver="schur",
        log_file=f"doublewell-2d-newton-history_T{T}_Nx{Nx}_Ny{Ny}_Nt{Nt}_nu{nu_target}.txt",
        save_plot=f"type1-doublewell2d-plot_T{T}_Nx{Nx}_Ny{Ny}_Nt{Nt}_nu{nu_target}.png",
    )
    print(
        f"\nResult: status={result['status']}, residual={result['residual']:.4e}, "
        f"iters={result['newton_iters']}, cont_steps={result['n_cont_steps']}, "
        f"backtracks={result['n_backtracks']}, time={result['time']:.2f}s"
    )
