"""Render the original measurement wCRB for the feedback replay."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_figures_7_9 import FIG8_POLICIES, REPO, read_csv, write_csv

TARGETS = ("k1", "k2", "k3", "k4")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#5B5B5B")
STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v")
LABELS = ("Proposed TTC-AO", "Risk-Unaware AO", "Min-Max AO", "TTC-Proportional", "Uniform-Dwell")
TOTAL_LABEL = r"$\omega\mathrm{CRB}=\sum_k\omega_k\mathrm{CRB}_k$"
TARGET_LABEL = r"Target contribution $\omega_k\mathrm{CRB}_k$"


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
    name, keys, values = ("fig8a", policies, means) if config.get("figure8_only") else ("fig9", TARGETS, target_means)
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


def check_panel_alignment(fig, path):
    helper = Path.home() / ".codex/skills/nature-figure/scripts"
    if (helper / "audit_panel_alignment.py").is_file():
        sys.path.insert(0, str(helper))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=path, tolerance_pt=1.5,
                                          gutter_tolerance_pt=1.5, strict=True)
        return
    scale = np.array([fig.get_figwidth() * 72, fig.get_figheight() * 72] * 2)
    box = np.array([ax.get_position().bounds for ax in fig.axes]) * scale
    gutters = box[1:, 0] - box[:-1, 0] - box[:-1, 2]
    differences = dict(bottom=np.ptp(box[:, 1]), width=np.ptp(box[:, 2]),
                       height=np.ptp(box[:, 3]), gutter=np.ptp(gutters))
    assert all(value <= 1.5 for value in differences.values()), differences
    path.write_text(json.dumps(dict(rectangles_pt=box.tolist(), maximum_differences_pt=differences,
                                   tolerance_pt=1.5, verdict="PASS"), indent=2), encoding="utf-8")


def configure_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "axes.labelsize": 11, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix",
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.size": 4.0,
                         "ytick.major.size": 4.0, "xtick.minor.size": 2.2, "ytick.minor.size": 2.2})


def redraw_published(directory=None, images=None):
    """Render experimental CSVs with the published figure filenames."""
    directory = directory if directory is not None else REPO / "结果"
    images = images if images is not None else REPO / "图片"
    images.mkdir(parents=True, exist_ok=True)
    release = json.loads((REPO / "结果/current_release.json").read_text(encoding="utf-8"))
    names = {key: Path(paths[0]).stem for key, paths in release["figure_files"].items()}
    seven = read_csv(directory / "fig7.csv")
    scenarios = tuple(dict.fromkeys(row["scenario"] for row in seven))
    iterations = matrix([row for row in seven if row["scenario"] == scenarios[0]], ("iteration",)).ravel()
    values7 = np.column_stack([matrix([row for row in seven if row["scenario"] == scenario],
                                     ("weighted_wcrb",)).ravel() for scenario in scenarios])
    designs = [("fig7", iterations, values7, tuple(f"Scenario {i}" for i in range(1, 6)),
                "Alternating-optimization iteration", TOTAL_LABEL, 24)]
    fig8 = []
    for name, column, xlabel, stride in (("fig8a", "time_s", "t(s)", 40),
                                         ("fig8b", "x_value", "d(m)", 5),
                                         ("fig8c", "x_value", "v(m/s)", 5)):
        rows = read_csv(directory / f"{name}.csv")
        x, values = matrix(rows, (column,)).ravel(), matrix(rows, FIG8_POLICIES)
        fig8.append((x, values, xlabel, stride))
        designs.append((name, x, values, LABELS, xlabel, TOTAL_LABEL, stride))
    nine = read_csv(directory / "fig9.csv")
    designs.append(("fig9", matrix(nine, ("time_s",)).ravel(), matrix(nine, TARGETS),
                    TARGETS, "t(s)", TARGET_LABEL, 40))
    for name, x, values, labels, xlabel, ylabel, stride in designs:
        assert np.all(np.diff(x) > 0) and np.isfinite(values).all()
        fig, ax = plt.subplots(figsize=(6.25, 5))
        plot_lines(ax, x, values, labels, xlabel, ylabel, stride)
        if name == "fig9":
            # The definition is a sum; each existing curve remains one summand.
            ax.set_title(TOTAL_LABEL, fontsize=11, pad=16)
        legend(ax, len(labels))
        fig.tight_layout()
        save(fig, images, names[name])
    fig, axes = plt.subplots(1, 3, figsize=(18.75, 5.6))
    fig.subplots_adjust(left=.062, right=.985, bottom=.16, top=.72, wspace=.30)
    for idx, (ax, (x, values, xlabel, stride)) in enumerate(zip(axes, fig8)):
        plot_lines(ax, x, values, LABELS, xlabel, TOTAL_LABEL, stride)
        ax.text(1, 1.035, f"({chr(97 + idx)})", transform=ax.transAxes, ha="right", va="bottom")
        legend(ax, len(LABELS))
    fig.canvas.draw()
    check_panel_alignment(fig, directory / "fig8_combined.alignment.json")
    save(fig, images, names["fig8_combined"])
    # Fig. 10(a) retains the separately documented posterior-bound evaluator.
    rows = read_csv(directory / "fig10a_total_wcrb.csv")
    x = matrix(rows, ("cycle_id",)).ravel()
    fig, ax = plt.subplots(figsize=(6.25, 5))
    for column, prefix, color, style in zip(("wCRB_ON", "wCRB_OFF"), ("With", "Without"),
                                            COLORS[:2], STYLES[:2]):
        ax.plot(x, matrix(rows, (column,)).ravel(), color=color, linestyle=style, linewidth=1.4,
                label=f"{prefix} historical {TOTAL_LABEL} feedback")
    ax.set(xlim=(x[0], x[-1]), xlabel="Decision cycle $t$", ylabel=TOTAL_LABEL)
    ax.grid(alpha=.2)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.07), frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, images, names["fig10a"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--published", action="store_true", help="Redraw released CSVs into 图片; no simulation")
    parser.add_argument("--images", type=Path, help="Image directory for --published")
    args = parser.parse_args()
    configure_style()
    if args.published:
        redraw_published(args.directory, args.images)
        return
    if args.directory is None:
        parser.error("directory is required unless --published is selected")
    directory = args.directory
    config = json.loads((directory / "design.json").read_text(encoding="utf-8"))
    time, five, contributions = summarize_replay(directory, config)
    ylabel = TOTAL_LABEL
    if config.get("figure8_only"):
        policies = config["methods"]
        labels_by_method = dict(zip(FIG8_POLICIES, LABELS))
        labels8 = tuple(labels_by_method[method] for method in policies)
        fig8 = [(time, five, "t(s)", 40)]
        sweep_directory = REPO / "结果" if config.get("figure8a_only") else directory
        for name, xlabel in (("fig8b", "d(m)"), ("fig8c", "v(m/s)")):
            rows = read_csv(sweep_directory / f"{name}.csv")
            x = matrix(rows, ("x_value",)).ravel()
            assert len(x) == 51 and np.all(np.diff(x) > 0)
            fig8.append((x, matrix(rows, policies), xlabel, 5))
        designs = [(name, x, values, labels8, axis, ylabel, stride)
                   for name, (x, values, axis, stride) in zip(("fig8a", "fig8b", "fig8c"), fig8)]
        if config.get("figure8a_only"):
            designs = designs[:1]
    else:
        seven = read_csv(directory / "fig7.csv")
        names = tuple(dict.fromkeys(row["scenario"] for row in seven))
        round_number = np.arange(241)
        data7 = np.array([[float(row["weighted_wcrb"]) for row in seven if row["scenario"] == name]
                          for name in names]).T
        assert data7.shape == (241, 5)
        designs = [
            ("fig7", round_number, data7, tuple(f"Scenario {i}" for i in range(1, 6)),
             "Alternating-optimization iteration", ylabel, 24),
            ("fig9", time, contributions, TARGETS, "t(s)", TARGET_LABEL, 40),
        ]
    for name, x, values, labels, xlabel, y_axis, stride in designs:
        fig, ax = plt.subplots(figsize=(6.25, 5))
        plot_lines(ax, x, values, labels, xlabel, y_axis, stride)
        if name == "fig9":
            ax.set_title(TOTAL_LABEL, fontsize=11, pad=16)
        legend(ax, len(labels))
        fig.tight_layout()
        save(fig, directory, name)
    if not config.get("figure8_only"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(18.75, 5.6))
    fig.subplots_adjust(left=.062, right=.985, bottom=.16, top=.72, wspace=.30)
    for idx, (ax, (x, values, xlabel, stride)) in enumerate(zip(axes, fig8)):
        plot_lines(ax, x, values, labels8, xlabel, ylabel, stride)
        ax.text(1, 1.035, f"({chr(97 + idx)})", transform=ax.transAxes, ha="right", va="bottom")
        legend(ax, len(labels8))
    fig.canvas.draw()
    check_panel_alignment(fig, directory / "fig8_combined.alignment.json")
    save(fig, directory, "fig8_combined")


if __name__ == "__main__":
    main()
