"""Render the original measurement wCRB for the feedback replay."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_figures_7_9 import MINIMAX, POLICIES, REPO, read_csv, write_csv
sys.path.insert(0, str(REPO / "实验/scripts"))
from export_figures import COLORS, LABELS, MARKERS, STYLES
from export_figures import alignment_gate as require_matplotlib_panel_alignment

TARGETS = ("k1", "k2", "k3", "k4")


def summarize_replay(directory, config):
    policies = config["methods"]
    repeats = len(config["seeds"])
    cycles = config["cycles"]
    totals = np.zeros((repeats, cycles, len(policies)))
    targets = np.zeros((repeats, cycles, len(TARGETS)))
    times = np.full(cycles, np.nan)
    seen = set()
    seed_indices = {seed: idx for idx, seed in enumerate(config["seeds"])}
    for row in read_csv(directory / "replay_by_seed.csv"):
        s, t, m = seed_indices[int(row["seed"])], int(row["cycle"]), policies.index(row["method"])
        assert (s, t, m) not in seen
        seen.add((s, t, m))
        time = float(row["time_s"])
        assert np.isnan(times[t]) or times[t] == time
        times[t] = time
        contributions = np.array([float(row[k]) for k in TARGETS])
        totals[s, t, m] = float(row["weighted_wcrb"])
        assert abs(contributions.sum() - totals[s, t, m]) < 1e-12
        if m == 0:
            targets[s, t] = contributions
    assert len(seen) == totals.size and np.all(np.diff(times) > 0)
    assert np.all(totals > 0)
    means = totals.mean(axis=0)
    target_means = targets.mean(axis=0)
    np.testing.assert_allclose(target_means.sum(axis=-1), means[:, 0], rtol=1e-12)
    tables = [("fig8a", policies, means)]
    if not config.get("figure8_only"):
        tables.append(("fig9", TARGETS, target_means))
    for name, keys, values in tables:
        write_csv(directory / f"{name}.csv", [
            {"time_s": t, **dict(zip(keys, row))} for t, row in zip(times, values)
        ])
    return times, means, target_means


def plot_lines(ax, x, y, labels, xlabel, ylabel, mark_stride):
    for i, label in enumerate(labels):
        ax.plot(x, y[:, i], color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i],
                markevery=(i, mark_stride), markersize=3, markerfacecolor="white",
                linewidth=1.65 if i == 0 else 1.45, label=label)
    ax.set(xlim=(x[0], x[-1]), xlabel=xlabel, ylabel=ylabel)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.grid(color=".88", linewidth=.45)


def legend(ax, n):
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2] if n == 5 else list(range(n))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3 if n == 5 else 4,
              loc="lower center", bbox_to_anchor=(.5, 1.10), columnspacing=.8,
              frameon=False, fontsize=8.2, handlelength=2.2)


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=600)
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.svg")
    plt.close(fig)


def matrix(rows, columns):
    return np.array([[float(row[key]) for key in columns] for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    config = json.loads((directory / "design.json").read_text(encoding="utf-8"))
    time, five, contributions = summarize_replay(directory, config)
    policies = config["methods"]
    labels_by_method = dict(zip(POLICIES, LABELS))
    labels_by_method[MINIMAX] = "Min-Max AO"
    labels8 = tuple(labels_by_method[method] for method in policies)
    fig8 = [(time, five, "t(s)", 40)]
    for name, xlabel in (("fig8b", "d(m)"), ("fig8c", "v(m/s)")):
        rows = read_csv(directory / f"{name}.csv")
        x = matrix(rows, ("x_value",)).ravel()
        assert len(x) == 51 and np.all(np.diff(x) > 0)
        values = matrix(rows, policies)
        fig8.append((x, values, xlabel, 5))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "axes.labelsize": 11, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "xtick.direction": "in", "ytick.direction": "in"})
    ylabel = r"$\omega$CRB"
    designs = [(name, x, values, labels8, axis, ylabel, stride)
               for name, (x, values, axis, stride) in zip(("fig8a", "fig8b", "fig8c"), fig8)]
    if not config.get("figure8_only"):
        seven = read_csv(directory / "fig7.csv")
        names = tuple(dict.fromkeys(row["scenario"] for row in seven))
        round_number = np.arange(241)
        data7 = np.array([[float(row["weighted_wcrb"]) for row in seven if row["scenario"] == name]
                          for name in names]).T
        assert data7.shape == (241, 5)
        designs.extend([
            ("fig7", round_number, data7, tuple(f"Scenario {i}" for i in range(1, 6)),
             "Alternating-optimization iteration", ylabel, 24),
            ("fig9", time, contributions, TARGETS, "t(s)", ylabel, 40),
        ])
    for name, x, values, labels, xlabel, y_axis, stride in designs:
        fig, ax = plt.subplots(figsize=(6.25, 5))
        plot_lines(ax, x, values, labels, xlabel, y_axis, stride)
        legend(ax, len(labels))
        fig.tight_layout()
        save(fig, directory, name)
    fig, axes = plt.subplots(1, 3, figsize=(18.75, 5.6))
    fig.subplots_adjust(left=.062, right=.985, bottom=.16, top=.72, wspace=.30)
    for idx, (ax, (x, values, xlabel, stride)) in enumerate(zip(axes, fig8)):
        plot_lines(ax, x, values, labels8, xlabel, ylabel, stride)
        ax.text(1, 1.035, f"({chr(97 + idx)})", transform=ax.transAxes, ha="right", va="bottom")
        legend(ax, len(labels8))
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, directory / "fig8_combined.alignment.json")
    save(fig, directory, "fig8_combined")


if __name__ == "__main__":
    main()
