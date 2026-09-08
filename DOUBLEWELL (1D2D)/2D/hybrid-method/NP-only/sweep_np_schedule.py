"""2D NP sweep with sticky-theta schedule + monotonicity early-stop.

For each Nx (=Ny), iterates nu from high to low. On first failure at a fixed Nx,
marks all subsequent (lower) nu as 'failed_by_monotonicity' without running them.
Each (nu, Nx) cell tries the theta schedule {0.8, 0.7, ..., 0.05} in order; stops
at the first converging theta. Writes np2d_schedule_sweep.csv incrementally.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import importlib.util
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "np2d_solver", os.path.join(_here, "doublewell-2d-picard.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
solve_np = _module.solve_np


T = 1.0
NT = 10
NUS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
NXS = [int(sys.argv[1])] if len(sys.argv) > 1 else [16, 32, 64, 128]
THETA_SCHEDULE = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
PICARD_MAX_ITER = 200
RESIDUAL_TOL = 1e-6
STALL_PATIENCE = 10
LINEAR_SOLVER = "direct"
CSV_FILE = os.path.join(_here, "np2d_schedule_sweep.csv")
FIELDS = ["nu", "Nx", "Ny", "Nt", "status", "theta_used", "n_attempts",
          "total_time", "iters_at_converged", "residual_at_converged"]


def _t1(x):
    """Round a time (seconds) to 1 decimal place."""
    return round(x, 1) if x is not None else None


def _r2(x):
    """Residual to 2 decimals in scientific notation (e.g. 1.13e-08)."""
    return float(f"{x:.2e}") if x is not None else None


def append_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_cell(nu, Nx, theta_start):
    total_time = 0.0
    n_attempts = 0
    for theta in (value for value in THETA_SCHEDULE if value <= theta_start + 1e-12):
        n_attempts += 1
        r = solve_np(
            nu=nu, Nx=Nx, Ny=Nx, Nt=NT, T=T,
            damping=theta,
            picard_max_iter=PICARD_MAX_ITER,
            residual_tol=RESIDUAL_TOL,
            stall_patience=STALL_PATIENCE,
            initial_guess=None,
            linear_solver=LINEAR_SOLVER,
            log_file=None, save_plot=None,
            verbose=False,
        )
        total_time += r["time"]
        if r["status"] == "converged":
            return {
                "nu": nu, "Nx": Nx, "Ny": Nx, "Nt": NT,
                "status": "converged", "theta_used": theta,
                "n_attempts": n_attempts, "total_time": _t1(total_time),
                "iters_at_converged": r["picard_iters"],
                "residual_at_converged": _r2(r["residual"]),
            }
    return {
        "nu": nu, "Nx": Nx, "Ny": Nx, "Nt": NT,
        "status": "failed", "theta_used": None,
        "n_attempts": n_attempts, "total_time": _t1(total_time),
        "iters_at_converged": None, "residual_at_converged": None,
    }


def implied_fail_row(nu, Nx):
    return {
        "nu": nu, "Nx": Nx, "Ny": Nx, "Nt": NT,
        "status": "failed_by_monotonicity", "theta_used": None,
        "n_attempts": 0, "total_time": 0.0,
        "iters_at_converged": None, "residual_at_converged": None,
    }


def main():
    with open(CSV_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    t_start = time.time()

    for Nx in NXS:
        failed_already = False
        theta = THETA_SCHEDULE[0]
        for nu in sorted(NUS, reverse=True):
            if failed_already:
                row = implied_fail_row(nu, Nx)
                print(f"[implied] nu={nu:<8} Nx=Ny={Nx} status=failed_by_monotonicity", flush=True)
            else:
                print(f"[run]     nu={nu:<8} Nx=Ny={Nx} ...", flush=True)
                row = run_cell(nu, Nx, theta)
                print(f"          -> status={row['status']:<10} theta={row['theta_used']!s:<6} "
                      f"attempts={row['n_attempts']:<2} time={row['total_time']:.1f}s", flush=True)
                if row["status"] == "failed":
                    failed_already = True
                else:
                    theta = row["theta_used"]

            append_row(CSV_FILE, row)

    print(f"\nTotal: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
