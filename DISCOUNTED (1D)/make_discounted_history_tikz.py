"""Generate the four discounted-problem history tikz figures from the solver logs."""
import re, sys
from pathlib import Path

picard_log, newton_log, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

# ---- Picard log ----
row = re.compile(r"^\s*(\d+)\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)\s*\|\s*(\d+)")
its, relu, relm, nits = [], [], [], []
for line in picard_log.read_text().splitlines():
    m = row.match(line)
    if m:
        its.append(int(m.group(1))); relu.append(float(m.group(2)))
        relm.append(float(m.group(3))); nits.append(int(m.group(4)))
assert its, "no Picard rows"

# ---- Newton log ----
attempts = []  # (nu, converged, n_updates)
cur = None
for line in newton_log.read_text().splitlines():
    m = re.match(r"\s*(?:First Solve:|Attempting) nu = (\d+\.\d+)", line)
    if m:
        cur = [float(m.group(1)), None, None]; attempts.append(cur); continue
    m = re.match(r"\s*Iter\s+(\d+) \| res = [0-9.eE+-]+ \(Converged\)", line)
    if m and cur is not None:
        cur[1], cur[2] = True, int(m.group(1)); continue
    m = re.match(r"\s*-> Failed", line)
    if m and cur is not None:
        cur[1] = False
maxit = int(re.search(r"NEWTON_MAX_ITER = (\d+)", newton_log.read_text()).group(1))
for a in attempts:
    if a[1] is False: a[2] = maxit
assert all(a[1] is not None for a in attempts), attempts
target = float(re.search(r"NU_TARGET=([0-9.]+)", newton_log.read_text()).group(1))

def coords(pairs, fmt="{:.4e}"):
    out, line = [], ""
    for k, (x, y) in enumerate(pairs):
        s = f"({x},{fmt.format(y)}) "
        line += s
        if (k + 1) % 5 == 0: out.append("    " + line.rstrip()); line = ""
    if line: out.append("    " + line.rstrip())
    return "\n".join(out)

n = len(its); na = len(attempts)
xt_p = ",".join(str(k) for k in range(5, n + 1, 5))
xt_a = ",".join(str(k) for k in range(5, na + 1, 5))
hdr = "% Coordinates generated from {} by make_discounted_history_tikz.py; do not edit by hand.\n"

(outdir / "picard-histories" / "discounted-1d-picard-history.tex").write_text(hdr.format(picard_log.name) + rf"""\begin{{tikzpicture}}
\begin{{semilogyaxis}}[
    scale only axis,
    width=6.75cm,
    height=4.5cm,
    title={{Picard Iteration History with $\theta = 0.8$}},
    xlabel={{Picard Iterate}},
    ylabel={{Relative Error (Log Scale)}},
    xmin=0, xmax={n + 1},
    ymin=1e-7, ymax=2,
    xtick={{{xt_p}}},
    ytick={{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}},
    grid=both,
    grid style={{line width=.1pt, draw=gray!10}},
    major grid style={{line width=.2pt, draw=gray!50}},
    legend pos=north east,
    tick label style={{font=\small}},
    label style={{font=\small}},
    title style={{font=\scriptsize\bfseries}}
]

\addplot[
    color=red,
    mark=*,
    mark size=1.5pt,
    thick
] coordinates {{
{coords(zip(its, relu))}
}};
\addlegendentry{{Rel Error $u$}}

\addplot[
    color=blue,
    mark=square*,
    mark size=1.5pt,
    thick
] coordinates {{
{coords(zip(its, relm))}
}};
\addlegendentry{{Rel Error $m$}}

% --- Threshold Line ---
\draw[dashed, thick] (axis cs:0, 1e-6) -- (axis cs:{n + 1}, 1e-6) node[pos=0.05, above right] {{$\epsilon_P = 10^{{-6}}$}};

\end{{semilogyaxis}}
\end{{tikzpicture}}
""")

lo, hi = (min(nits) // 25) * 25, (max(nits) // 25 + 1) * 25
(outdir / "total-backward-HJB-newton-its" / "discounted1d-picard-total-newton-its.tex").write_text(hdr.format(picard_log.name) + rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    scale only axis,
    width=6.75cm,
    height=4.5cm,
    title={{Newton Iterations per Picard Iteration}},
    xlabel={{Picard Iterate}},
    ylabel={{Number of Newton Iterations}},
    xmin=0, xmax={n + 1},
    ymin={lo}, ymax={hi},
    xtick={{{xt_p}}},
    grid=both,
    grid style={{line width=.1pt, draw=gray!10}},
    major grid style={{line width=.2pt, draw=gray!50}},
    legend pos=north east,
    tick label style={{font=\small}},
    label style={{font=\small}},
    title style={{font=\scriptsize\bfseries}}
]
\addplot[
    color=blue,
    mark=square*,
    mark size=1.5pt,
    thick
] coordinates {{
{coords(zip(its, nits), "{:d}")}
}};
\addlegendentry{{Newton Its}}
\end{{axis}}
\end{{tikzpicture}}
""")

conv = [(k + 1, a[0]) for k, a in enumerate(attempts) if a[1]]
fail = [(k + 1, a[0]) for k, a in enumerate(attempts) if not a[1]]
allp = [(k + 1, a[0]) for k, a in enumerate(attempts)]
(outdir / "global-newton-continuation-histories" / "discounted-1d-newton-continuation-history.tex").write_text(hdr.format(newton_log.name) + rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    scale only axis,
    width=6.75cm,
    height=4.5cm,
    ymin=0.4,
    ymax=1.1,
    ytick={{0.4, 0.6, 0.8, 1.0}},
    xmin=0,
    xmax={na + 1},
    xtick={{{xt_a}}},
    xlabel={{Continuation Attempt}},
    ylabel={{$\nu$ (Viscosity)}},
    title={{Viscosity Continuation for Global Newton}},
    grid=major,
    grid style={{dashed, gray!60}},
    legend pos=north east,
    legend cell align={{left}},
    legend style={{font=\small, nodes={{scale=0.9, transform shape}}}},
    tick label style={{font=\small}},
    label style={{font=\small}},
    title style={{font=\scriptsize\bfseries}}
]
\addplot[
    only marks,
    mark=*,
    mark size=2pt,
    color=black,
    fill=black
] coordinates {{
{coords(conv, "{:.4f}")}
}};
\addlegendentry{{Converged}}

\addplot[
    only marks,
    mark=x,
    mark size=3pt,
    thick,
    color=red
] coordinates {{
{coords(fail, "{:.4f}")}
}};
\addlegendentry{{Failed}}

\addplot[
    color=black,
    dashed,
    domain=0:{na + 1},
    samples=2,
    thick
] {{{target}}};
\addlegendentry{{Target $\nu = {target}$}}

\addplot[
    color=gray,
    draw opacity=0.5,
    thick,
    forget plot
] coordinates {{
{coords(allp, "{:.4f}")}
}};
\end{{axis}}
\end{{tikzpicture}}
""")

convi = [(k + 1, a[2]) for k, a in enumerate(attempts) if a[1]]
faili = [(k + 1, a[2]) for k, a in enumerate(attempts) if not a[1]]
alli = [(k + 1, a[2]) for k, a in enumerate(attempts)]
(outdir / "total-global-Newton-newton-its" / "discounted-1d-newton-total-iterations.tex").write_text(hdr.format(newton_log.name) + rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    scale only axis,
    width=6.75cm,
    height=4.5cm,
    title={{Newton Iterations per Continuation Attempt}},
    xlabel={{Continuation Attempt}},
    ylabel={{Number of Newton Iterations}},
    xmin=0, xmax={na + 1},
    ymin=0, ymax={maxit + 8},
    xtick={{{xt_a}}},
    ytick={{0,5,10,15,20}},
    grid=major,
    grid style={{dashed, gray!60}},
    legend pos=north east,
    legend cell align={{left}},
    legend style={{font=\small, nodes={{scale=0.9, transform shape}}}},
    tick label style={{font=\small}},
    label style={{font=\small}},
    title style={{font=\scriptsize\bfseries}}
]
\addplot[
    color=gray,
    draw opacity=0.5,
    thick,
    forget plot
] coordinates {{
{coords(alli, "{:d}")}
}};

\addplot[
    only marks,
    mark=*,
    mark size=2pt,
    color=black,
    fill=black
] coordinates {{
{coords(convi, "{:d}")}
}};
\addlegendentry{{Converged}}

\addplot[
    only marks,
    mark=x,
    mark size=3pt,
    thick,
    color=red
] coordinates {{
{coords(faili, "{:d}")}
}};
\addlegendentry{{Failed}}
\end{{axis}}
\end{{tikzpicture}}
""")

print(f"Picard: {n} iterates, newton its {min(nits)}-{max(nits)}")
print(f"Newton: {na} attempts, {sum(a[1] for a in attempts)} converged, {sum(not a[1] for a in attempts)} failed, "
      f"{sum(1 for a in attempts if not a[1] and a[0]==target)} failed direct at target; "
      f"total solves {sum(a[2] for a in attempts)}, converged solves {sum(a[2] for a in attempts if a[1])}")
