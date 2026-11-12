"""Reproduce the constrained Fig. 7–10(a) release."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
SCRIPTS = CODE / "latest/recursive_feedback"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = CODE.parent / "结果" / datetime.now().strftime("ttc_threshold_%Y%m%d_%H%M%S")
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "run_threshold_figures.py"),
                    "--output", str(args.output), "--workers", str(args.workers)], check=True)
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "plot_figures_7_9.py"),
                    "--published", str(args.output), "--images", str(args.output / "图片")], check=True)


if __name__ == "__main__":
    main()
