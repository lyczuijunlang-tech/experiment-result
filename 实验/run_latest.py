"""Reproduce the current Figures 7–9 in a new output directory."""
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
    args = parser.parse_args()
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "run_figures_7_9.py"),
                    "--output", str(args.output), "--workers", str(args.workers)], check=True)
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "plot_figures_7_9.py"),
                    str(args.output)], check=True)


if __name__ == "__main__":
    main()
