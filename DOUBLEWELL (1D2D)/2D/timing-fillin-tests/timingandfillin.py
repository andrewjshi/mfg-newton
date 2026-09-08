"""
timingandfillin.py
==================
Complexity / timing study of the Newton-Picard SPATIAL Jacobian in 2D.

For the Newton-Picard method the per-timestep HJB Jacobian J_n(U) is a single
Nh^2 x Nh^2 matrix with the 2D 5-point (pentadiagonal) stencil.  This script
builds that block for the asymmetric double-well problem and, for a sequence of
grid sizes Nh, reports:

    - DOFs            = Nh^2
    - Bandwidth       = Nh           (y-neighbour offset in lexicographic order)
    - nnz(J)          number of stored nonzeros in the Jacobian
    - Fill-in ratio   = nnz(L+U) / nnz(J)   from a direct LU factorization (splu)
    - Solve Time (s)  wall-clock for one direct solve (factorize + back-substitute)

This regenerates the data previously reported as the Newton-Picard spatial-Jacobian
complexity table.  Fill-in and timing are governed by the matrix SPARSITY PATTERN
(2D 5-point stencil) and the fill-reducing ordering, not by the specific drift
coefficients; the double-well physics is used only to produce a representative,
physically meaningful J_n(U).

Note on ordering: scipy's splu uses COLAMD by default.  A true nested-dissection
ordering (e.g. METIS) gives the asymptotically optimal O(Nh^3) factor cost quoted
in the paper; pass permc_spec below to experiment.
"""

import csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

# ----------------------------------------------------------------------
# Parameters (asymmetric double-well, d = 2; cf. Section 4.3)
# ----------------------------------------------------------------------
GRIDS   = [25, 50, 100, 200]   # Nh values to scan
nu      = 0.1                  # viscosity (only scales the diagonal; irrelevant to fill-in)
T, L    = 1.0, 2.5             # horizon and half-domain  Omega = [-L, L]^2
Nt      = 10                   # only sets dt; does not affect the spatial sparsity
PERMC   = "COLAMD"             # splu ordering: NATURAL / COLAMD / MMD_AT_PLUS_A / MMD_ATA
CSV_OUT = "timingandfillin_results.csv"


def build_operators(Nh):
    """Periodic forward/backward differences and the 5-point Laplacian (Kron lift)."""
    dx = (2.0 * L) / Nh
    # 1D periodic forward (D+) and backward (D-) differences
    Dp_1d = sp.spdiags([-np.ones(Nh), np.ones(Nh)], [0, 1], Nh, Nh).tolil()
    Dp_1d[Nh - 1, Nh - 1], Dp_1d[Nh - 1, 0] = -1.0, 1.0
    Dp_1d = Dp_1d.tocsr() / dx
    Dm_1d = sp.spdiags([-np.ones(Nh), np.ones(Nh)], [-1, 0], Nh, Nh).tolil()
    Dm_1d[0, 0], Dm_1d[0, Nh - 1] = 1.0, -1.0
    Dm_1d = Dm_1d.tocsr() / dx

    I_1d = sp.eye(Nh)
    D_plus_x  = sp.kron(I_1d, Dp_1d).tocsr()
    D_minus_x = sp.kron(I_1d, Dm_1d).tocsr()
    D_plus_y  = sp.kron(Dp_1d, I_1d).tocsr()
    D_minus_y = sp.kron(Dm_1d, I_1d).tocsr()
    Lap = D_plus_x @ D_minus_x + D_plus_y @ D_minus_y
    return D_plus_x, D_minus_x, D_plus_y, D_minus_y, Lap


def build_np_spatial_jacobian(Nh):
    """The Newton-Picard HJB spatial Jacobian block J_n(U) = d F_HJB / d U^n."""
    dt = T / Nt
    Dpx, Dmx, Dpy, Dmy, Lap = build_operators(Nh)
    N = Nh * Nh
    I = sp.eye(N)

    # representative value-function state: the double-well terminal potential V(x,y)
    x = np.linspace(-L, L, Nh, endpoint=False)
    X, Y = np.meshgrid(x, x)
    U = (2.0 * (X ** 2 - 1.0) ** 2 + 0.3 * X + 0.5 * Y ** 2).flatten()

    # Godunov upwind gradients (same construction as the double-well solvers)
    p_min_x = np.minimum(Dpx @ U, 0.0)
    p_max_x = np.maximum(Dmx @ U, 0.0)
    p_min_y = np.minimum(Dpy @ U, 0.0)
    p_max_y = np.maximum(Dmy @ U, 0.0)

    J = ((1.0 / dt) * I - nu * Lap
         + sp.diags(p_min_x) @ Dpx + sp.diags(p_max_x) @ Dmx
         + sp.diags(p_min_y) @ Dpy + sp.diags(p_max_y) @ Dmy).tocsc()
    return J


def main():
    print(f"Newton-Picard spatial-Jacobian timing / fill-in study  "
          f"(nu={nu}, Omega=[-{L},{L}]^2, ordering={PERMC})")
    print("-" * 78)
    hdr = (f"{'Grid':>10} {'DOFs':>9} {'Bandwidth':>10} "
           f"{'nnz(J)':>10} {'Fill-in':>9} {'Solve (s)':>11}")
    print(hdr)
    print("-" * 78)
    results = []
    for Nh in GRIDS:
        J = build_np_spatial_jacobian(Nh)
        N = Nh * Nh
        rng = np.random.default_rng(0)
        b = rng.standard_normal(N)
        t0 = time.time()
        lu = spla.splu(J, permc_spec=PERMC)   # factorize ...
        _ = lu.solve(b)                       # ... and one back-substitution
        solve_t = time.time() - t0
        fill = (lu.L.nnz + lu.U.nnz) / J.nnz
        print(f"{Nh:>4}x{Nh:<5} {N:>9,} {Nh:>10} "
              f"{J.nnz:>10,} {fill:>8.2f}x {solve_t:>11.4e}")
        results.append({"Nh": Nh, "DOFs": N, "bandwidth": Nh, "nnz": J.nnz,
                        "fill_in_ratio": round(fill, 4),
                        "solve_time_s": round(solve_t, 6)})
    print("-" * 78)

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
