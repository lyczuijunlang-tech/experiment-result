"""Recompute Figures 7–9 with history-dependent posterior bounds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from itertools import groupby, repeat
import json
from pathlib import Path

import numpy as np

from run_feedback import (
    ACCEL_PSD, INPUTS, REPO, CurrentState, evaluate_candidate, extract_snapshots, load_traffic,
    model, optimization_weights, original, track, weighted_bound,
)

POLICIES = ("TTC-AO", "RU-AO", "DA-AO", "TTC-PD", "UD")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay_seed(seed, replay, initialization, radar, system, references):
    discovery, warmup_count = initialization
    results = []
    frozen = None
    previous = {}
    for (cycle, method), group in groupby(
        track(replay, discovery, radar, system, seed, references, POLICIES),
        key=lambda row: (row["cycle"], row["method"]),
    ):
        rows = list(group)
        first = rows[0]
        assert len(rows) == 4 and abs(sum(row["truth_weight"] for row in rows) - 4) < 1e-12
        assert first["optimized_objective"] <= first["initial_objective"] * (1 + 1e-10)
        contributions = []
        for row in rows:
            key = (method, row["target"])
            if cycle:
                assert row["previous_weighted_pcrb"] == previous[key]["optimized_wPCRB"]
                for suffix in ("dd", "dv", "vv"):
                    assert row[f"previous_post_{suffix}"] == previous[key][f"post_{suffix}"]
            previous[key] = row
            contributions.append(row["truth_weight"] * (row["post_dd"] * references[0] +
                                                         row["post_vv"] * references[1]))
        if first["time_s"] < 0:
            continue
        result = dict(seed=seed, cycle=cycle - warmup_count, time_s=first["time_s"], method=method,
                      weighted_pcrb=sum(contributions), Ts_s=first["Ts_s"], Ps_w=first["Ps_w"])
        result.update({f"k{k}": v for k, v in enumerate(contributions, 1)})
        results.append(result)
        if cycle == len(replay) - 1 and method == POLICIES[0]:
            frozen = [[[r["prior_dd"], r["prior_dv"]], [r["prior_dv"], r["prior_vv"]]] for r in rows]
    if frozen is None or len(results) != (len(replay) - warmup_count) * len(POLICIES):
        raise RuntimeError(f"Incomplete run for seed {seed}")
    return results, frozen


def scene(targets, radar):
    weights = model.compute_ttc_weights(targets).omega
    beams = [model.closed_form_beams(target.azimuth_rad, radar) for target in targets]
    coefficients = np.array([
        model.information_coefficients_per_watt(target, beam, radar)[:2]
        for target, beam in zip(targets, beams)
    ])
    return CurrentState(targets, beams, radar, tuple(weights)), coefficients


def convergence(targets, radar, system, prior_info, references):
    settings = read_csv(INPUTS / "scenario_settings.csv")
    rows = []
    ts_cap = system.sensing_time_qos_max_s - 1e-12
    ps_cap = min(system.sensing_power_max_w, system.sensing_energy_max_j / ts_cap)
    for setting in settings:
        perturbed = [
            replace(target, distance_m=target.distance_m * float(setting["distance_scale"]),
                    closing_speed_mps=target.closing_speed_mps * float(setting["closing_speed_scale"]))
            for target in targets
        ]
        state, coefficients = scene(perturbed, radar)
        weights = np.array(state.physical_ttc_weights)
        ts = float(setting["initial_Ts_ms"]) / 1000
        ps = float(setting["initial_Ps_mW"]) / 1000
        initial = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights, references, True)
        best = initial
        for iteration in range(241):
            if iteration:
                ts += float(setting["time_relaxation"]) * (ts_cap - ts)
                time_update = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                                 references, True, best["dwell"])
                ps += float(setting["power_relaxation"]) * (ps_cap - ps)
                new = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                         references, True, time_update["dwell"])
                assert new["bound"] <= time_update["bound"] * (1 + 1e-10)
                assert time_update["bound"] <= best["bound"] * (1 + 1e-10)
                best = new
            rows.append(dict(scenario=setting["scenario"], iteration=iteration, weighted_pcrb=best["bound"],
                             Ts_s=ts, Ps_w=ps, distance_scale=setting["distance_scale"],
                             closing_speed_scale=setting["closing_speed_scale"],
                             time_relaxation=setting["time_relaxation"],
                             power_relaxation=setting["power_relaxation"]))
    return rows


def sensitivity(targets, radar, system, prior_info, references, scan, variable):
    rows = []
    ts = system.sensing_time_qos_max_s - 1e-12
    ps = min(system.sensing_power_max_w, system.sensing_energy_max_j / ts)
    index = next(i for i, target in enumerate(targets) if target.vehicle_id == scan["selected_vehicle_id"])
    for point, scale in enumerate(scan[variable]):
        field = "distance_m" if variable == "range_scales" else "closing_speed_mps"
        perturbed = list(targets)
        target = targets[index]
        x = getattr(target, field) * scale
        perturbed[index] = replace(target, **{field: x})
        state, coefficients = scene(perturbed, radar)
        physical_weights = np.array(state.physical_ttc_weights)
        row = dict(point_index=point, scale=scale, x_value=x)
        for method in POLICIES:
            weights = optimization_weights(method, state)
            direct = method if method in ("TTC-PD", "UD") else None
            best = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                      references, True, direct=direct)
            row[method] = weighted_bound(best["post"], physical_weights, references)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.seeds, args.workers) < 1:
        parser.error("--seeds and --workers must be positive")
    replay = list(extract_snapshots(load_traffic(INPUTS / "traffic.csv")))
    warmup = list(extract_snapshots(load_traffic(INPUTS / "warmup_traffic.csv")))
    radar, system = model.RadarConfig(), model.SystemConfig()
    references = 1 / np.array([radar.crb_d_reference_m2, radar.crb_v_reference_m2ps2])
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = range(17001, 17001 + args.seeds)
    inputs = (seeds, repeat(warmup[1:] + replay), repeat((warmup[0], len(warmup) - 1)),
              repeat(radar), repeat(system), repeat(references))
    records, frozen = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed, (batch, prior) in zip(seeds, pool.map(replay_seed, *inputs)):
            records.extend(batch)
            frozen.append(prior)
            print(f"Completed five-policy SUMO replay for seed {seed}", flush=True)
    write_csv(args.output / "replay_by_seed.csv", records)
    covariance = np.asarray(frozen).mean(axis=0)
    prior_info = np.linalg.inv(covariance)
    time, ego, targets = replay[-1]
    snr, _ = original.communication_snr_from_rsu(ego)
    fixed_system = replace(system, communication_snr_linear=snr)
    scan = json.loads((INPUTS / "scan_settings.json").read_text(encoding="utf-8"))
    assert time == scan["baseline_time_s"]
    assert [t.vehicle_id for t in targets][1] == scan["selected_vehicle_id"]
    (args.output / "design.json").write_text(json.dumps({
        "seeds": list(seeds), "cycles": len(replay), "methods": POLICIES,
        "frozen_time_s": time, "frozen_prior_covariances_by_seed": frozen,
        "shared_prior_covariance": covariance.tolist(),
        "target_order": [t.vehicle_id for t in targets],
        "reference_target": scan["selected_vehicle_id"],
        "fixed_communication_snr_linear": snr,
        "initial_discovery_time_s": warmup[0][0],
        "unplotted_tracking_warmup_updates": len(warmup) - 1,
        "acceleration_psd_m2_s3": ACCEL_PSD,
    }, indent=2), encoding="utf-8")
    write_csv(args.output / "fig7.csv", convergence(targets, radar, fixed_system, prior_info, references))
    write_csv(args.output / "fig8b.csv", sensitivity(targets, radar, fixed_system, prior_info,
                                                   references, scan, "range_scales"))
    velocity = sensitivity(targets, radar, fixed_system, prior_info, references, scan, "closing_speed_scales")
    write_csv(args.output / "fig8c.csv", velocity)
    dt = replay[-1][0] - replay[-2][0]
    unit_process_cov = np.array([[dt**3 / 3, -dt**2 / 2], [-dt**2 / 2, dt]])
    noise_checks = []
    for density in (ACCEL_PSD, 0.10, 0.20):
        # Stress only the final prediction step while fixing the preceding history.
        covariance_stressed = covariance + (density - ACCEL_PSD) * unit_process_cov
        rows = velocity if density == ACCEL_PSD else sensitivity(
            targets, radar, fixed_system, np.linalg.inv(covariance_stressed), references,
            scan, "closing_speed_scales",
        )
        for method in POLICIES:
            noise_checks.append(dict(final_step_acceleration_psd_m2_s3=density, method=method,
                                     low_speed_wPCRB=rows[0][method], high_speed_wPCRB=rows[-1][method],
                                     change_percent=100 * (rows[-1][method] / rows[0][method] - 1)))
    write_csv(args.output / "process_noise_check.csv", noise_checks)


if __name__ == "__main__":
    main()
