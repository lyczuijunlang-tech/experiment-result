"""Compare constrained curves with the archived release at matching coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_figures_1_3 import FIG2_POLICIES, read_csv, write_csv


def read_archive(directory, figure):
    """Accept either current names or the immutable pre-renumbering release."""
    path = directory / f"{figure}.csv"
    if path.is_file():
        return read_csv(path)
    legacy = {"fig1": "fig7", "fig2a": "fig8a", "fig2b": "fig8b",
              "fig2c": "fig8c", "fig3": "fig9", "fig4_total_wpcrb": "fig10a_total_wcrb"}
    rows = read_csv(directory / f"{legacy[figure]}.csv")
    if figure == "fig4_total_wpcrb":
        for row in rows:
            row["wPCRB_ON"] = row.pop("wCRB_ON")
            row["wPCRB_OFF"] = row.pop("wCRB_OFF")
    return rows


def compare(before, after):
    specs = [
        ("fig1", ("scenario", "iteration"), ("weighted_wcrb",)),
        ("fig2a", ("time_s",), FIG2_POLICIES),
        ("fig2b", ("x_value",), FIG2_POLICIES),
        ("fig2c", ("x_value",), FIG2_POLICIES),
        ("fig3", ("time_s",), ("k1", "k2", "k3", "k4")),
        ("fig4_total_wpcrb", ("cycle_id", "simulation_time"), ("wPCRB_ON", "wPCRB_OFF")),
    ]
    changes = []
    for figure, coordinates, series in specs:
        archive = read_archive(before, figure)
        constrained = read_csv(after / f"{figure}.csv")
        assert len(archive) == len(constrained)
        for old, new in zip(archive, constrained):
            for x in coordinates:
                assert old[x] == new[x] if x == "scenario" else abs(float(old[x])-float(new[x])) < 1e-12
        groups = (tuple(dict.fromkeys(row["scenario"] for row in archive)) if figure == "fig1" else (None,))
        for group in groups:
            for key in series:
                old = np.array([float(r[key]) for r in archive if r.get("scenario") == group])
                new = np.array([float(r[key]) for r in constrained if r.get("scenario") == group])
                assert (old > 0).all() and (new > 0).all()
                change = new-old
                span = np.ptp(old)
                changes.append(dict(figure=figure, series=group or key, points=len(old),
                                    previous_mean=old.mean(), constrained_mean=new.mean(),
                                    mean_change_pct=100*(new.mean()/old.mean()-1),
                                    mean_absolute_point_change_pct=100*np.mean(abs(change)/old),
                                    max_absolute_point_change_pct=100*np.max(abs(change)/old),
                                    max_added_bend_pct_of_old_span=100*np.max(abs(np.diff(change, n=2)))/span))
    return changes


def audit(data):
    specs = [
        ("fig1", "fig1.csv", ("", "time_candidate_"), 5*241),
        ("fig2a", "fig2a_weak_echo/replay_by_seed.csv", ("",), 16*401*5),
        ("fig2b", "fig2b.csv", tuple(f"{m}_" for m in FIG2_POLICIES), 51),
        ("fig2c", "fig2c.csv", tuple(f"{m}_" for m in FIG2_POLICIES), 51),
        ("fig3", "replay_by_seed.csv", ("",), 16*401),
        ("fig4_on", "feedback_on/replay_by_seed.csv", ("",), 16*401),
        ("fig4_off", "feedback_off/replay_by_seed.csv", ("",), 16*401),
    ]
    reports = {}
    for name, source, prefixes, expected in specs:
        rows = read_csv(data / source)
        assert len(rows) == expected, (name, len(rows))
        gamma_max = 0.
        n_active = n_binding = rejected_stages = 0
        margins = {}
        for row in rows:
            if "Ts_s" in row and "predicted_closing_k1_mps" in row:
                dwell = np.array([float(row[f"dwell_k{k}_s"]) for k in range(1, 5)])
                minimum = .008 if name == "fig2a" else .002
                total = float(row["Ts_s"])
                assert abs(dwell.sum()-total) <= 1e-10 and min(dwell) >= minimum-1e-10
                if row["method"] == "UD":
                    np.testing.assert_allclose(dwell, total/4, atol=1e-12)
                if row["method"] == "TTC-PD":
                    raw = np.array([1/(float(row[f"predicted_range_k{k}_m"]) /
                                        float(row[f"predicted_closing_k{k}_mps"])+.1)
                                    if float(row[f"predicted_closing_k{k}_mps"]) > 0 else 0.
                                    for k in range(1, 5)])
                    shares = raw/raw.sum() if raw.sum() else np.full(4, .25)
                    np.testing.assert_allclose(dwell, minimum+(total-4*minimum)*shares, atol=1e-12)
            rejected_stages += len(row["rejected_p2_candidates"].split(";")) if row.get("rejected_p2_candidates") else 0
            for prefix in prefixes:
                assert row[f"{prefix}allocation_status"] == "feasible"
                limit = float(row[f"{prefix}gamma_limit"])
                for k in range(1, 5):
                    gamma = float(row[f"{prefix}gamma_plus_k{k}"])
                    if f"predicted_closing_k{k}_mps" in row:
                        assert np.isfinite(gamma) == (float(row[f"predicted_closing_k{k}_mps"]) > 0)
                    if np.isnan(gamma):
                        continue
                    assert gamma <= limit*(1+1e-9), (name, prefix, k, gamma, limit)
                    gamma_max = max(gamma, gamma_max)
                    n_active += 1
                    n_binding += abs(gamma-limit) <= limit*1e-7
                for key in ("time_lower_margin_s", "time_upper_margin_s", "power_lower_margin_w",
                            "power_upper_margin_w", "qos_rate_margin_bps", "energy_margin_j"):
                    value = float(row[f"{prefix}{key}"])
                    assert value >= -1e-9, (name, key, value)
                    margins[key] = min(margins.get(key, value), value)
        reports[name] = dict(feasible_rows=len(rows), checked_candidate_sets=len(prefixes), limit=limit,
                             max_gamma=gamma_max, active_target_checks=n_active,
                             binding_target_checks=n_binding, rejected_p2_intermediate_candidates=rejected_stages,
                             minimum_resource_margins=margins)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    changes = compare(args.before, args.after)
    checks = audit(args.after)
    write_csv(args.after / "threshold_curve_comparison.csv", changes)
    (args.after / "threshold_feasibility.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    for key, report in checks.items():
        print(key, report["limit"], report["max_gamma"], report["binding_target_checks"])
    for key in dict.fromkeys(row["figure"] for row in changes):
        rows = [r for r in changes if r["figure"] == key]
        print(key, max(abs(r["mean_change_pct"]) for r in rows),
              max(r["max_absolute_point_change_pct"] for r in rows),
              max(r["max_added_bend_pct_of_old_span"] for r in rows))


if __name__ == "__main__":
    main()
