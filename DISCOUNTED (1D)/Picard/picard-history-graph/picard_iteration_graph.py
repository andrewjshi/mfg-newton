from pathlib import Path
import re

import matplotlib.pyplot as plt


script_dir = Path(__file__).resolve().parent
history_path = script_dir.parent / "discounted-1d-picard-history.txt"
output_path = script_dir / "discounted-1d-picard-history.png"

iteration_pattern = re.compile(
    r"^\s*(\d+)\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)"
    r"\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)"
)

iterations = []
relative_error_u = []
relative_error_m = []
for line in history_path.read_text().splitlines():
    match = iteration_pattern.match(line)
    if match:
        iterations.append(int(match.group(1)))
        relative_error_u.append(float(match.group(2)))
        relative_error_m.append(float(match.group(3)))

if not iterations:
    raise RuntimeError(f"No Picard iterations found in {history_path}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(iterations, relative_error_u, "r-o", label=r"Relative error of $u$")
ax.semilogy(iterations, relative_error_m, "b-s", label=r"Relative error of $m$")
ax.axhline(1e-6, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
ax.text(iterations[0] - 0.3, 1.2e-6, r"$\epsilon_P=10^{-6}$", fontsize=12)
ax.set_title("Picard Iteration History")
ax.set_xlabel("Picard iteration")
ax.set_ylabel("Relative error")
ax.set_xticks(iterations)
ax.set_xlim(iterations[0] - 0.5, iterations[-1] + 0.5)
ax.grid(True, which="both", linestyle="-", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(output_path, dpi=200)
