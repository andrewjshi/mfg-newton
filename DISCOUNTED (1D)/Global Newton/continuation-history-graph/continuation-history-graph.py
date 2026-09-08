"""Plot the viscosity-continuation history of the global Newton solver.

Reads discounted-1d-newton-history.txt (written by discounted-1d-newton.py in the
parent directory) and plots, for every continuation attempt, the viscosity tried
and whether the Newton iteration converged, together with the number of Newton
updates spent in that attempt. Nothing is hardcoded: rerunning the driver and then
this script regenerates the figure from the new log.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = Path(__file__).resolve().parent
history_path = script_dir.parent / "discounted-1d-newton-history.txt"
output_path = script_dir / "discounted-1d-newton-continuation-history.png"

text = history_path.read_text()
max_iter = int(re.search(r"NEWTON_MAX_ITER = (\d+)", text).group(1))
target_match = re.search(r"NU_TARGET=(\d+\.\d+)", text)
target_nu = float(target_match.group(1)) if target_match else None

# One entry per attempt: [nu, converged, newton_updates]
attempts = []
current = None
for line in text.splitlines():
    match = re.match(r"\s*(?:First Solve:|Attempting) nu = (\d+\.\d+)", line)
    if match:
        current = [float(match.group(1)), None, None]
        attempts.append(current)
        continue
    match = re.match(r"\s*Iter\s+(\d+) \| res = [0-9.eE+-]+ \(Converged\)", line)
    if match and current is not None:
        current[1], current[2] = True, int(match.group(1))
        continue
    if re.match(r"\s*-> Failed", line) and current is not None:
        current[1], current[2] = False, max_iter

if not attempts or any(a[1] is None for a in attempts):
    raise RuntimeError(f"Could not parse continuation attempts from {history_path}")
if target_nu is None:
    target_nu = attempts[-1][0]

x = list(range(1, len(attempts) + 1))
nus = [a[0] for a in attempts]
its = [a[2] for a in attempts]
colors = ["black" if a[1] else "red" for a in attempts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x, nus, color="gray", alpha=0.5, zorder=1)
ax1.scatter(x, nus, c=colors, s=45, zorder=2)
ax1.axhline(target_nu, color="black", linestyle=":", linewidth=2, zorder=1)
ax1.set_xlabel("Continuation attempt")
ax1.set_ylabel(r"$\nu$ (viscosity)")
ax1.set_title("Viscosity continuation for global Newton")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(x, its, color="gray", alpha=0.5, zorder=1)
ax2.scatter(x, its, c=colors, s=45, zorder=2)
ax2.set_xlabel("Continuation attempt")
ax2.set_ylabel("Newton updates in attempt")
ax2.set_title("Newton updates per continuation attempt")
ax2.set_ylim(0, max_iter + 8)
ax2.grid(True, linestyle="--", alpha=0.6)

legend_lines = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=9),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=9),
    Line2D([0], [0], color="black", linestyle=":", linewidth=2),
]
ax1.legend(legend_lines, ["Converged", "Failed", rf"Target $\nu = {target_nu:g}$"], loc="upper right")
ax2.legend(legend_lines[:2], ["Converged", "Failed"], loc="upper right")

n_conv = sum(a[1] for a in attempts)
print(f"{len(attempts)} attempts, {n_conv} converged, {len(attempts) - n_conv} failed, "
      f"{sum(its)} Newton updates in total ({sum(a[2] for a in attempts if a[1])} in converged attempts)")

fig.tight_layout()
fig.savefig(output_path, dpi=200)
