Email me at andrewshi@math.berkeley.edu for bugs, questions, requests, support, etc. 

# Code Description

This repository contains the code for the numerical examples for the paper "Computational Trade-Offs Between Newton and Picard Solvers for Mean Field Game PDE Systems" by Mathieu Lauriere and Andrew Shi, 2026.

**(ADD ARXIV LINK WHEN AVAILABLE)**

There are six different examples in this paper: {discounted, ergodic, doublewell, congestionlocal, congestionnonlocal, trafficlight} which could be implemented in dimension {1d, 2d} and with method {Newton, Picard}.

So in principle could be $6x2x2 = 24$ codes with the naming convention example-dimension-method.py (e.g. discounted-1d-newton.py), but certain dimensions or methods are not implemented for all problems.

| Problem             | 1D Newton | 1D Picard | 2D Newton | 2D Picard |
|---------------------|:---------:|:---------:|:---------:|:---------:|
| Discounted          | ✅        | ✅        | ❌        | ❌        |
| Ergodic             | ❌        | ✅        | ❌        | ❌        |
| Double Well         | ✅        | ✅        | ✅        | ✅        |
| Local Congestion    | ✅        | ✅        | ✅        | ✅        |
| Nonlocal Congestion | ❌        | ✅        | ❌        | ✅        |
| Traffic Light       | ✅        | ✅        | ❌        | ❌        |

- **Newton** — the monolithic space-time Global Newton solve with viscosity continuation.
- **Picard** — the damped Picard iteration with an inner Newton solve for the backward HJB sweep.

# Recreating the Paper Figures and Tables

Each table below maps a figure/table in the paper to the script that generates it and the output file that script produces.

> **Note:** Each entry below maps a paper figure or table to its generating code and numerical artifact. Some figures are included directly as PNG output; others are TikZ renderings generated from stored numerical data or iteration histories. Tables use CSV results. Running the listed script regenerates the underlying artifact, while the manuscript supplies the final presentation.

### 1. [The Discounted Problem](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DISCOUNTED%20%281D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 1 | [discounted-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DISCOUNTED%20%281D%29/Picard) | [discounted-1d-picard-plot.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DISCOUNTED%20%281D%29/Picard/discounted-1d-picard-plot.png) |
| Figure 2 | [discounted-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DISCOUNTED%20%281D%29/Picard) | [discounted-1d-picard-history.txt](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DISCOUNTED%20%281D%29/Picard/discounted-1d-picard-history.txt) |
| Figure 3 | [discounted-1d-newton.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DISCOUNTED%20%281D%29/Global%20Newton) | [discounted-1d-newton-history.txt](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DISCOUNTED%20%281D%29/Global%20Newton/discounted-1d-newton-history.txt) |

*Figure 1 can also be produced by `discounted-1d-newton.py`. For Figures 2 and 3, both the left and right panels are drawn from the linked data file.*

### 2. [The Ergodic Problem](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/ERGODIC%20%281D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 5 | [ergodic-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/ERGODIC%20%281D%29/Picard) | [ergodic-1d-picard-plot.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/ERGODIC%20%281D%29/Picard/ergodic-1d-picard-plot.png) |
| Figure 6 | [discounted-1d-rho-limit.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DISCOUNTED%20%281D%29/rho-limit) | [squashing_parabolas_T10.0.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DISCOUNTED%20%281D%29/rho-limit/squashing_parabolas_T10.0.png) (left), [loglog_convergence_T10.0.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DISCOUNTED%20%281D%29/rho-limit/loglog_convergence_T10.0.png) (right) |

### 3. [The Double-Well Problem](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DOUBLEWELL%20%281D2D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 7 | [doublewell-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DOUBLEWELL%20%281D2D%29/1D/Picard) | [doublewell-1d-picard-plot.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/1D/Picard/doublewell-1d-picard-plot.png) |
| Figure 8 | [doublewell-1d-spectral-radius.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DOUBLEWELL%20%281D2D%29/1D/Spectral-Radius) | [rho_P.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/1D/Spectral-Radius/rho_P.png) (left), [rho_P_theta.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/1D/Spectral-Radius/rho_P_theta.png) (right) |
| Figure 9 | [doublewell-2d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DOUBLEWELL%20%281D2D%29/2D/Picard) | [doublewell-2d-picard-contour.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/Picard/doublewell-2d-picard-contour.png) |
| Table 3 | [timingandfillin.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/timing-fillin-tests/timingandfillin.py) | [timingandfillin_results.csv](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/timing-fillin-tests/timingandfillin_results.csv) |
| Table 4 | [sweep_np_schedule.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/NP-only/sweep_np_schedule.py) | [np2d_schedule_sweep.csv](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/NP-only/np2d_schedule_sweep.csv) |
| Table 5 | [sweep_gn_schedule.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/GN-only/sweep_gn_schedule.py) | [gn2d_schedule_sweep.csv](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/GN-only/gn2d_schedule_sweep.csv) |
| Table 6 | [hybrid-2d.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/Hybrid/hybrid-2d.py) | [hybrid2d_per_nu_times_Nh128.csv](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/hybrid-method/Hybrid/hybrid2d_per_nu_times_Nh128.csv) |

### 4. [Local Congestion](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-LOCAL%20%281D2D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 11 | [congestionlocal-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-LOCAL%20%281D2D%29/1D/Picard) | [congestionlocal-1d-picard-plot.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/CONGESTION-LOCAL%20%281D2D%29/1D/Picard/congestionlocal-1d-picard-plot.png) |
| Figure 12 | [congestionlocal-2d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-LOCAL%20%281D2D%29/2D/Picard) | [congestionlocal-2d-picard-contour.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/CONGESTION-LOCAL%20%281D2D%29/2D/Picard/congestionlocal-2d-picard-contour.png) |

### 5. [Non-Local Congestion](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-NONLOCAL%20%281D2D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 13 | [congestionnonlocal-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-NONLOCAL%20%281D2D%29/1D/Picard) | [congestionnonlocal-1d-picard-plot.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/CONGESTION-NONLOCAL%20%281D2D%29/1D/Picard/congestionnonlocal-1d-picard-plot.png) |
| Figure 14 | [congestionnonlocal-2d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/CONGESTION-NONLOCAL%20%281D2D%29/2D/Picard) | [congestionnonlocal-2d-picard-contour.png](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/CONGESTION-NONLOCAL%20%281D2D%29/2D/Picard/congestionnonlocal-2d-picard-contour.png) |

### 6. [Traffic Light](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/TRAFFIC-LIGHT%20%281D%29)

| Paper | Script | Output |
|---|---|---|
| Figure 16 | [trafficlight-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/TRAFFIC-LIGHT%20%281D%29/Picard) (run at θ = 0.3 and θ = 0.1) | [trafficlight-1d-picard-history.txt](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/TRAFFIC-LIGHT%20%281D%29/Picard/trafficlight-1d-picard-history.txt) (one per θ) |
| Figure 17 | [trafficlight-1d-picard.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/TRAFFIC-LIGHT%20%281D%29/Picard) | [img/](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/TRAFFIC-LIGHT%20%281D%29/Picard/img) frames: frame_000, 029, 050, 069, 080, 100 .png |

### 7. Conclusion

| Paper | Script | Output |
|---|---|---|
| Direct v. Iterative<br>(Statement in Conclusion) | [iterative-vs-direct](https://github.com/andrewjshi/2026-JSC-MFGNewton/tree/main/DOUBLEWELL%20%281D2D%29/2D/Global%20Newton/iterative-vs-direct) | [run_benchmark.py](https://github.com/andrewjshi/2026-JSC-MFGNewton/blob/main/DOUBLEWELL%20%281D2D%29/2D/Global%20Newton/iterative-vs-direct/run_benchmark.py) |
