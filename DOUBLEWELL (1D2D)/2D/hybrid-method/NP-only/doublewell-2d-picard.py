import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time


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
    save_plot=None,
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
        "status": status,
        "residual": res_total,
        "picard_iters": k,
        "time": total_time,
        "U": U_flow,
        "M": M_flow,
        "history": history,
    }


if __name__ == "__main__":
    nu = 0.1
    Nx = Ny = 50
    Nt = 20
    T = 1.0
    result = solve_np(
        nu=nu, Nx=Nx, Ny=Ny, Nt=Nt, T=T,
        log_file=f"doublewell-2d-picard-history_T{T}_Nx{Nx}_Ny{Ny}_Nt{Nt}_nu{nu}.txt",
        save_plot=f"doublewell2d-plot_T{T}_Nx{Nx}_Ny{Ny}_Nt{Nt}_nu{nu}.png",
    )
    print(f"\nResult: status={result['status']}, residual={result['residual']:.4e}, "
          f"iters={result['picard_iters']}, time={result['time']:.2f}s")
