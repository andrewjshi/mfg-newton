"""Create the traffic-light Picard comparison used in Figure 16."""

from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PICARD_DIR = HERE.parent
OUTPUT = HERE / "trafficlight-1d-picard-history-graph.png"


def read_history(path):
    """Read iteration and relative-error columns from a Picard history log."""
    iterations, rel_u, rel_m = [], [], []
    with path.open(encoding="utf-8") as history:
        for line in history:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 5:
                continue
            try:
                iterations.append(int(fields[0]))
                rel_u.append(float(fields[2]))
                rel_m.append(float(fields[4]))
            except ValueError:
                continue
    return iterations, rel_u, rel_m


experiments = [
    (0.3, PICARD_DIR / "trafficlight-1d-picard-history-theta0p3.txt"),
    (0.1, PICARD_DIR / "trafficlight-1d-picard-history-theta0p1.txt"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for axis, (theta, history_path) in zip(axes, experiments):
    iterations, rel_u, rel_m = read_history(history_path)
    axis.semilogy(iterations, rel_u, linewidth=2, label=r"Relative error in $u$")
    axis.semilogy(iterations, rel_m, linewidth=2, label=r"Relative error in $m$")
    axis.axhline(1.0e-6, color="black", linestyle="--", linewidth=1, label="Tolerance")
    axis.set_title(rf"Picard damping $\theta={theta}$")
    axis.set_xlabel("Picard iteration")
    axis.set_xlim(0, 150)
    axis.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel("Relative error")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT}")
