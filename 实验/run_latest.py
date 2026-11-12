"""Reproduce the reference Figures 7–9 or the updated Fig. 8 Min-Max comparison."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
SCRIPTS = CODE / "latest/recursive_feedback"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = CODE.parent / "结果" / datetime.now().strftime("feedback_%Y%m%d_%H%M%S")
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figure8-minmax", action="store_true")
    parser.add_argument("--min-dwell-ms", type=float)
    parser.add_argument("--fig8a-k3-rcs-m2", type=float)
    args = parser.parse_args()
    comparison = ["--figure8-minmax"] if args.figure8_minmax else []
    if args.min_dwell_ms is not None:
        comparison.extend(["--min-dwell-ms", str(args.min_dwell_ms)])
    if args.fig8a_k3_rcs_m2 is not None:
        comparison.extend(["--fig8a-k3-rcs-m2", str(args.fig8a_k3_rcs_m2)])
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "run_figures_7_9.py"),
                    "--output", str(args.output), "--workers", str(args.workers), *comparison], check=True)
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "plot_figures_7_9.py"),
                    str(args.output)], check=True)


if __name__ == "__main__":
    main()
