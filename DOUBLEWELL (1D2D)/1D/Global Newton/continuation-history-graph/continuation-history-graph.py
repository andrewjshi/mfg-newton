import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


script_dir = Path(__file__).resolve().parent
history_path = script_dir.parent / "doublewell-1d-newton-history.txt"
output_path = script_dir / "doublewell-1d-newton-continuation-history.png"

# Extract every attempted viscosity and whether that attempt converged.
attempts = []
current_nu = None
with history_path.open(encoding="utf-8") as history_file:
    for line in history_file:
        match = re.search(r"(?:First Solve:|Attempting) nu = ([0-9.eE+-]+)", line)
        if match:
            if current_nu is not None:
                attempts.append((current_nu, False))
            current_nu = float(match.group(1).rstrip("."))
        elif current_nu is not None and "-> Converged!" in line:
            attempts.append((current_nu, True))
            current_nu = None

if current_nu is not None:
    attempts.append((current_nu, False))
if not attempts:
    raise RuntimeError(f"No continuation attempts found in {history_path}")

target_nu = attempts[-1][0]

# Generate the plot
x = range(1, len(attempts) + 1)
y = [a[0] for a in attempts]
colors = ['black' if a[1] else 'red' for a in attempts]

plt.figure(figsize=(12, 6))

# Plot connecting line
plt.plot(x, y, color='gray', alpha=0.5, linestyle='-', zorder=1)

# Plot the dots
plt.scatter(x, y, c=colors, s=60, zorder=2)

# Target line
plt.axhline(y=target_nu, color='black', linestyle=':', linewidth=2, zorder=1)

# Formatting
plt.xlabel('Continuation Attempt')
plt.ylabel(r'$\nu$ (Viscosity)')
plt.title('Viscosity Continuation Progress (1D Double-Well Global Newton)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(x[::2]) # Subsample x-ticks for better readability

# Custom legend
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
    Line2D([0], [0], color='black', linestyle=':', linewidth=2)
]
plt.legend(custom_lines, ['Converged', 'Failed', r'Target $\nu = 0.01$'], loc='upper right')

plt.tight_layout()
plt.savefig(output_path, dpi=200)
plt.show()
