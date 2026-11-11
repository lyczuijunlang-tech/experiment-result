"""Summarize paired tracking trials and export three figures from the run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_feedback import ACCEL_PSD, METHODS


def load_metrics(directory):
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    repeats = config["replicates"]
    cycles = config["cycles_per_replicate"]
    time = np.full(cycles, np.nan)
    arrays = {method: {
        name: np.zeros((repeats, cycles)) for name in (
            "prior_bound", "initial_bound", "time_update_bound", "posterior_bound", "frame_bound",
            "range_sq", "closing_sq", "frame_range_sq", "frame_closing_sq", "truth_weighted_sq",
        )
    } for method in METHODS}
    seen = {}
    previous = {}
    references = np.asarray(config["references_inverse"])
    with (directory / "tracking.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            seed, cycle = int(row["seed"]), int(row["cycle"])
            index = (seed - config["first_seed"], cycle)
            time[cycle] = float(row["time_s"])
            key = (method, seed, row["target"])
            cov_old = np.array([float(row[k]) for k in ("previous_post_dd", "previous_post_dv", "previous_post_vv")])
            cov_prior = np.array([float(row[k]) for k in ("prior_dd", "prior_dv", "prior_vv")])
            cov_new = np.array([float(row[k]) for k in ("post_dd", "post_dv", "post_vv")])
            forecast = np.array([float(row[k]) for k in ("predicted_range_m", "predicted_closing_mps")])
            estimate = np.array([float(row[k]) for k in ("estimated_range_m", "estimated_closing_mps")])
            previous_time = config["warmup_time_s"]
            if cycle:
                old_cycle, previous_time, expected_cov, previous_bound, old_estimate = previous[key]
                assert old_cycle == cycle - 1 and np.array_equal(cov_old, expected_cov)
                assert float(row["previous_weighted_pcrb"]) == previous_bound
                expected_forecast = old_estimate.copy()
                expected_forecast[0] -= (time[cycle] - previous_time) * old_estimate[1]
                np.testing.assert_allclose(forecast, expected_forecast, rtol=1e-12, atol=1e-13)
            dt = time[cycle] - previous_time
            # Independently check all entries of the prior, including the sign of the cross term.
            expected_prior = np.array([
                cov_old[0] - 2 * dt * cov_old[1] + dt**2 * cov_old[2] + ACCEL_PSD * dt**3 / 3,
                cov_old[1] - dt * cov_old[2] - ACCEL_PSD * dt**2 / 2,
                cov_old[2] + ACCEL_PSD * dt,
            ])
            np.testing.assert_allclose(cov_prior, expected_prior, rtol=1e-10, atol=1e-13)
            previous[key] = cycle, time[cycle], cov_new, float(row["optimized_wPCRB"]), estimate
            variance = 1 / np.array([float(row["predicted_Jdd"]), float(row["predicted_Jvv"])])
            prior_matrix = np.array([[cov_prior[0], cov_prior[1]], [cov_prior[1], cov_prior[2]]])
            gain = np.linalg.solve(prior_matrix + np.diag(variance), prior_matrix).T
            independent_post = prior_matrix - gain @ prior_matrix
            np.testing.assert_allclose(cov_new, independent_post[[0, 0, 1], [0, 1, 1]],
                                       rtol=1e-9, atol=1e-13)
            truth = np.array([float(row["true_range_m"]), float(row["true_closing_mps"])])
            measurement = np.array([float(row["measured_range_m"]), float(row["measured_closing_mps"])])
            np.testing.assert_allclose(estimate, forecast + gain @ (measurement - forecast),
                                       rtol=1e-12, atol=1e-13)
            error, frame_error = (estimate - truth)**2, (measurement - truth)**2
            weight = float(row["predicted_weight"])
            metrics = arrays[method]
            metrics["prior_bound"][index] += weight * (cov_prior[[0, 2]] @ references)
            metrics["initial_bound"][index] += float(row["initial_wPCRB"]) / 4
            metrics["time_update_bound"][index] += float(row["time_update_wPCRB"]) / 4
            metrics["posterior_bound"][index] += weight * (cov_new[[0, 2]] @ references)
            metrics["frame_bound"][index] += weight * (variance @ references)
            metrics["range_sq"][index] += error[0] / 4
            metrics["closing_sq"][index] += error[1] / 4
            metrics["frame_range_sq"][index] += frame_error[0] / 4
            metrics["frame_closing_sq"][index] += frame_error[1] / 4
            metrics["truth_weighted_sq"][index] += float(row["truth_weight"]) * (error @ references)
            seen[method, index] = seen.get((method, index), 0) + 1
    assert len(seen) == len(METHODS) * repeats * cycles and set(seen.values()) == {4}
    assert np.isfinite(time).all() and np.all(np.diff(time) > 0)
    primary = arrays[METHODS[0]]
    for comparator in ("prior_bound", "initial_bound", "frame_bound"):
        assert np.all(primary["posterior_bound"] <= primary[comparator] * (1 + 1e-9))
    assert np.all(primary["posterior_bound"] <= primary["time_update_bound"] * (1 + 1e-9))
    assert np.all(primary["time_update_bound"] <= primary["initial_bound"] * (1 + 1e-9))
    return config, time, arrays


def plot(directory, time, series, ylabel, name, log=False):
    fig, ax = plt.subplots(figsize=(6.25, 5))
    colors = ("#0072B2", "#626262", "#D55E00")
    for (label, data), color, style in zip(series, colors, ("-", "--", "-.")):
        ax.plot(time, data, label=label, color=color, linestyle=style, linewidth=1.35)
    if log:
        if any(np.any(data <= 0) for _, data in series):
            raise ValueError("Logarithmic bounds must be positive")
        ax.set_yscale("log")
    ax.set(xlim=(time[0], time[-1]), xlabel="t(s)", ylabel=ylabel)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=600)
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    config, time, arrays = load_metrics(args.directory)
    primary, baseline = (arrays[name] for name in METHODS)
    summary = {"replicates": config["replicates"], "cycles": len(time)}
    for name, data in arrays.items():
        entry = {}
        for key in ("range_sq", "closing_sq", "frame_range_sq", "frame_closing_sq"):
            entry[key.replace("_sq", "_rmse")] = {
                "all": float(np.sqrt(data[key].mean())),
                "first_50_cycles": float(np.sqrt(data[key][:, :50].mean())),
                "last_50_cycles": float(np.sqrt(data[key][:, -50:].mean())),
            }
        entry["truth_weighted_mean_squared_error"] = float(data["truth_weighted_sq"].mean())
        summary[name] = entry
    means = {name: data.mean(axis=0) for name, data in primary.items()}
    summary["posterior_bound"] = {
        "first_cycle": float(means["posterior_bound"][0]),
        "first_cycle_before_optimization": float(means["initial_bound"][0]),
        "first_cycle_after_time_update": float(means["time_update_bound"][0]),
        "last_cycle": float(means["posterior_bound"][-1]),
        "mean": float(means["posterior_bound"].mean()),
        "same_resource_frame_mean": float(means["frame_bound"].mean()),
        "increases_between_cycles": int(np.count_nonzero(np.diff(means["posterior_bound"]) > 0)),
        "increases_by_seed": np.count_nonzero(np.diff(primary["posterior_bound"], axis=1) > 0,
                                              axis=1).tolist(),
        "strict_reductions_from_initial_resource_point": int(np.count_nonzero(
            primary["posterior_bound"] < primary["initial_bound"] * (1 - 1e-10)
        )),
    }
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.directory / "cycle_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "prediction_wPCRB", "initial_resource_wPCRB", "time_update_wPCRB",
                         "optimized_wPCRB", "same_resource_frame_wCRB",
                         "posterior_range_rmse_m", "frame_range_rmse_m", "baseline_range_rmse_m",
                         "posterior_closing_rmse_mps", "frame_closing_rmse_mps", "baseline_closing_rmse_mps"])
        writer.writerows(zip(time, means["prior_bound"], means["initial_bound"], means["time_update_bound"],
                             means["posterior_bound"], means["frame_bound"],
                             *(np.sqrt(data[key].mean(axis=0)) for state in ("range", "closing")
                               for data, key in ((primary, f"{state}_sq"), (primary, f"frame_{state}_sq"),
                                                 (baseline, f"{state}_sq")))))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    plot(args.directory, time, (
        ("Optimized Posterior", means["posterior_bound"]),
        ("Predicted Prior", means["prior_bound"]),
        ("Single Frame, Same Resources", means["frame_bound"]),
    ), "Weighted Variance Bound", "feedback_bound", log=True)
    for state, ylabel in (("range", "Range RMSE (m)"), ("closing", "Closing Speed RMSE (m/s)")):
        plot(args.directory, time, (
            ("Posterior Allocation + Tracking", np.sqrt(means[f"{state}_sq"])),
            ("Single-Frame Allocation + Tracking", np.sqrt(baseline[f"{state}_sq"].mean(axis=0))),
            ("Single Frame, Same Resources", np.sqrt(means[f"frame_{state}_sq"])),
        ), ylabel, f"tracking_rmse_{state}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
