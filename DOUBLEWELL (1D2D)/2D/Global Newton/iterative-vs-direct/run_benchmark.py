"""Run the matched direct-LU and ILU-BiCGStab benchmark and save its log."""

from pathlib import Path
import platform
import subprocess
import sys

import numpy
import scipy


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "direct_lu_vs_ilu_bicgstab_results.txt"
SCRIPTS = ("direct.py", "iterative.py")


def append(text):
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()


def run(script):
    heading = f"\n{'=' * 80}\n{script}\n{'=' * 80}\n"
    append(heading)
    process = subprocess.Popen(
        [sys.executable, script],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        append(line)
    return_code = process.wait()
    append(f"Exit status: {return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, script)


def main():
    OUTPUT.write_text(
        "Direct LU versus ILU-preconditioned BiCGStab benchmark\n"
        f"Platform: {platform.platform()}\n"
        f"Python: {platform.python_version()}\n"
        f"NumPy: {numpy.__version__}\n"
        f"SciPy: {scipy.__version__}\n",
        encoding="utf-8",
    )
    print(f"Writing benchmark log to {OUTPUT.name}", flush=True)
    for script in SCRIPTS:
        run(script)
    print(f"Complete benchmark log written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

