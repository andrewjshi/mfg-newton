"""2D GN sweep with continuation across nu (descending).

For a fixed Nx=Ny, iterate nu high -> low. Each solve at the next-lower nu uses
the previous converged (U, M) as the initial guess. On first failure, mark
remaining lower-nu cells as 'failed_by_monotonicity' and stop.

Writes gn2d_schedule_sweep.csv incrementally.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import importlib.util
import sys
import time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "gn2d_solver", os.path.join(_here, "doublewell-2d-newton.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
solve_gn = _module.solve_gn

T = 1.0
NT = 10
NUS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
NXS = [int(sys.argv[1])] if len(sys.argv) > 1 else [16, 32, 64, 128]
NEWTON_MAX_ITER = 20
NEWTON_TOL = 1e-6
LINEAR_SOLVER = "direct"
CSV_FILE = os.path.join(_here, "gn2d_schedule_sweep.csv")
FIELDS = ["nu", "Nx", "Ny", "Nt", "status", "time", "cumulative_time",
          "newton_iters", "continued_from_previous_nu", "residual"]


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


def main():
    with open(CSV_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    t_start = time.time()
    for Nx in NXS:
        prev_W = None
        cumulative_time = 0.0
        for nu in sorted(NUS, reverse=True):
            continued = prev_W is not None
            print(f"[run]     nu={nu:<8} Nx=Ny={Nx} continuation={continued} ...", flush=True)
            r = solve_gn(
                nu_target=nu, Nx=Nx, Ny=Nx, Nt=NT, T=T,
                newton_max_iter=NEWTON_MAX_ITER, newton_tol=NEWTON_TOL,
                use_continuation=False,
                initial_guess=prev_W,
                linear_solver=LINEAR_SOLVER,
                log_file=None, save_plot=None, verbose=False,
            )
            cumulative_time += r["time"]
            print(
                f"          -> status={r['status']:<10} time={r['time']:.2f}s "
                f"cumulative={cumulative_time:.2f}s iters={r['newton_iters']} "
                f"residual={r['residual']:.2e}",
                flush=True,
            )
            row = {
                "nu": nu, "Nx": Nx, "Ny": Nx, "Nt": NT,
                "status": r["status"], "time": _t1(r["time"]),
                "cumulative_time": _t1(cumulative_time),
                "newton_iters": r["newton_iters"],
                "continued_from_previous_nu": continued,
                "residual": _r2(r["residual"]),
            }
            append_row(CSV_FILE, row)

            if r["status"] == "converged":
                prev_W = np.concatenate([r["U"].flatten(), r["M"].flatten()])
            else:
                remaining = [v for v in sorted(NUS, reverse=True) if v < nu]
                for nu_lo in remaining:
                    row = {
                        "nu": nu_lo, "Nx": Nx, "Ny": Nx, "Nt": NT,
                        "status": "failed_by_monotonicity",
                        "time": 0.0, "cumulative_time": _t1(cumulative_time),
                        "newton_iters": 0,
                        "continued_from_previous_nu": False, "residual": None,
                    }
                    append_row(CSV_FILE, row)
                    print(f"[implied] nu={nu_lo:<8} Nx=Ny={Nx} status=failed_by_monotonicity", flush=True)
                break

    print(f"\nTotal: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
