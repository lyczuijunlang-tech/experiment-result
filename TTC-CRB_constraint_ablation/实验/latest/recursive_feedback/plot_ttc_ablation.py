"""Plot the independent posterior TTC-CRB ablation from its own CSVs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from plot_figures_7_9 import COLORS, STYLES, TOTAL_LABEL, configure_style, plt, save
from run_figures_7_9 import read_csv
from run_ttc_ablation import ARMS


LABELS = ("Original TTC-AO", "TTC bound 2%", "TTC bound 5%", "TTC bound 10%")


def aggregate(rows, field, *, target=None, predicate=lambda row: True, transform=float):
    groups = defaultdict(list)
    times = sorted({float(r["time_s"]) for r in rows})
    for row in rows:
        if (target is None or row.get("target") == target) and predicate(row) and row[field] != "":
            groups[row["arm"], float(row["time_s"])].append(transform(row[field]))
    return times, [[fmean(groups[arm, t]) if groups[arm, t] else np.nan for t in times] for arm in ARMS]


def draw(ax, times, series):
    for i, (label, values) in enumerate(zip(LABELS, series)):
        ax.plot(times, values, label=label, color=COLORS[i], linestyle=STYLES[i],
                linewidth=1.65 if i in (0, 2) else 1.4)
    ax.set(xlim=(times[0], times[-1]), xlabel="t(s)")
    ax.grid(color=".88", linewidth=.45)


def single(output, name, times, series, ylabel, *, scientific=False):
    fig, ax = plt.subplots(figsize=(6.25, 5))
    draw(ax, times, series)
    ax.set_ylabel(ylabel)
    if scientific:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.10), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, output, name)


def targets(output, rows, name, field, ylabel, transform=float):
    fig, axes = plt.subplots(2, 2, figsize=(8.75, 7))
    fig.subplots_adjust(left=.095, right=.98, bottom=.09, top=.87, hspace=.42, wspace=.34)
    for i, ax in enumerate(axes.flat, 1):
        time, series = aggregate(rows, field, target=f"k{i}", transform=transform)
        draw(ax, time, series)
        ax.set_title(f"k{i}", fontsize=10, pad=9)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98), ncol=4, frameon=False, fontsize=9)
    fig.canvas.draw()
    scale = np.array([fig.get_figwidth(), fig.get_figheight()] * 2) * 72
    boxes = np.array([ax.get_position().bounds for ax in axes.flat]).reshape(2, 2, 4) * scale
    differences = dict(row_bottom=float(np.ptp(boxes[:, :, 1], axis=1).max()),
                       column_left=float(np.ptp(boxes[:, :, 0], axis=0).max()),
                       width=float(np.ptp(boxes[:, :, 2])), height=float(np.ptp(boxes[:, :, 3])))
    if max(differences.values()) > 1.5:
        raise RuntimeError(f"Plot-area alignment failed: {differences}")
    (output / f"{name}.alignment.json").write_text(json.dumps(
        dict(rectangles_pt=boxes.tolist(), maximum_differences_pt=differences, tolerance_pt=1.5, verdict="PASS"),
        indent=2), encoding="utf-8")
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    configure_style()
    out = directory / "figures"
    out.mkdir(exist_ok=True)
    cycles = [r for r in read_csv(directory / "cycles.csv") if r["warmup"] == "0"]
    rows = [r for r in read_csv(directory / "targets.csv") if r["warmup"] == "0"]
    time, data = aggregate(cycles, "weighted_wcrb")
    single(out, "ttc_constraint_wcrb", time, data, TOTAL_LABEL, scientific=True)
    time, data = aggregate(cycles, "weighted_wpcrb", predicate=lambda r: r["status"] == "feasible")
    single(out, "ttc_constraint_wpcrb", time, data, r"$\omega\mathrm{PCRB}$", scientific=True)
    time, data = aggregate(cycles, "status", transform=lambda status: 100. * (status == "feasible"))
    single(out, "ttc_constraint_feasibility", time, data, "Feasible replays (%)")
    targets(out, rows, "ttc_constraint_target_gamma", "gamma_plus",
            r"$\sqrt{\Gamma_k^+}$ (%)", transform=lambda x: 100*np.sqrt(float(x)))
    targets(out, rows, "ttc_constraint_target_dwell", "dwell_s", "Dwell time (ms)",
            transform=lambda x: 1000*float(x))


if __name__ == "__main__":
    main()
