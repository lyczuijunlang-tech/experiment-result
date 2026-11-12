"""Replay feedback-based allocation and score it with the original weighted CRB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from itertools import groupby, repeat
import json
from pathlib import Path

import numpy as np

from run_feedback import (
    ACCEL_PSD, INPUTS, MINIMAX, REPO, CurrentState, evaluate_candidate, extract_snapshots, information, load_traffic,
    model, optimization_weights, original, track, weighted_bound,
)

POLICIES = ("TTC-AO",)
FIG8_POLICIES = ("TTC-AO", "RU-AO", MINIMAX, "TTC-PD", "UD")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def instantaneous_contributions(diagonal_information, weights, references):
    """Original normalized measurement CRB: diag(J_LFM^-1), with no tracking prior."""
    if not np.isfinite(diagonal_information).all() or np.any(diagonal_information <= 0):
        raise ValueError("Measurement Fisher information must be finite and positive")
    return weights * ((1 / diagonal_information) @ references)


def replay_seed(seed, replay, initialization, radar, system, references, policies=POLICIES, radar_overrides=None):
    discovery, warmup_count = initialization
    results = []
    frozen = None
    previous = {}
    for (cycle, method), group in groupby(
        track(replay, discovery, radar, system, seed, references, policies, radar_overrides=radar_overrides),
        key=lambda row: (row["cycle"], row["method"]),
    ):
        rows = list(group)
        first = rows[0]
        assert len(rows) == 4 and abs(sum(row["truth_weight"] for row in rows) - 4) < 1e-12
        assert first["optimized_objective"] <= first["initial_objective"] * (1 + 1e-10)
        posterior_contributions = []
        for row in rows:
            key = (method, row["target"])
            if cycle:
                assert row["previous_weighted_pcrb"] == previous[key]["optimized_wPCRB"]
                for suffix in ("dd", "dv", "vv"):
                    assert row[f"previous_post_{suffix}"] == previous[key][f"post_{suffix}"]
            previous[key] = row
            posterior_contributions.append(row["truth_weight"] * (row["post_dd"] * references[0] +
                                                                   row["post_vv"] * references[1]))
        if first["time_s"] < 0:
            continue
        measurement_info = np.array([[r["realized_Jdd"], r["realized_Jvv"]] for r in rows])
        weights = np.array([r["truth_weight"] for r in rows])
        contributions = instantaneous_contributions(measurement_info, weights, references)
        result = dict(seed=seed, cycle=cycle - warmup_count, time_s=first["time_s"], method=method,
                      weighted_wcrb=float(contributions.sum()), weighted_pcrb=sum(posterior_contributions),
                      Ts_s=first["Ts_s"], Ps_w=first["Ps_w"])
        result.update({f"k{k}": v for k, v in enumerate(contributions, 1)})
        for k, (row, (jdd, jvv)) in enumerate(zip(rows, measurement_info), 1):
            result.update({f"omega_k{k}": row["truth_weight"], f"dwell_k{k}_s": row["dwell_s"],
                           f"Jdd_k{k}": jdd, f"Jvv_k{k}": jvv})
        results.append(result)
        if cycle == len(replay) - 1 and method == POLICIES[0]:
            frozen = [[[r["prior_dd"], r["prior_dv"]], [r["prior_dv"], r["prior_vv"]]] for r in rows]
    if frozen is None or len(results) != (len(replay) - warmup_count) * len(policies):
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
            contributions = instantaneous_contributions(information(best["dwell"], ps, coefficients),
                                                         weights, references)
            rows.append(dict(scenario=setting["scenario"], iteration=iteration,
                             weighted_wcrb=float(contributions.sum()), weighted_pcrb=best["bound"],
                             Ts_s=ts, Ps_w=ps, distance_scale=setting["distance_scale"],
                             closing_speed_scale=setting["closing_speed_scale"],
                             time_relaxation=setting["time_relaxation"],
                             power_relaxation=setting["power_relaxation"]))
    return rows


def sensitivity(targets, radar, system, prior_info, references, scan, variable, policies=POLICIES):
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
        for method in policies:
            weights = optimization_weights(method, state)
            direct = method if method in ("TTC-PD", "UD") else None
            best = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                      references, True, direct=direct, minimax=method == MINIMAX)
            contributions = instantaneous_contributions(information(best["dwell"], ps, coefficients),
                                                         physical_weights, references)
            row[method] = float(contributions.sum())
            row[f"{method}_wPCRB"] = weighted_bound(best["post"], physical_weights, references)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figure8-minmax", action="store_true")
    parser.add_argument("--min-dwell-ms", type=float, help="Common minimum dwell in Fig. 8 Min-Max comparisons")
    parser.add_argument("--fig8a-k3-rcs-m2", type=float, help="Run only Fig. 8(a) with this k3 radar cross section")
    args = parser.parse_args()
    if min(args.seeds, args.workers) < 1:
        parser.error("--seeds and --workers must be positive")
    replay = list(extract_snapshots(load_traffic(INPUTS / "traffic.csv")))
    warmup = list(extract_snapshots(load_traffic(INPUTS / "warmup_traffic.csv")))
    radar, system = model.RadarConfig(), model.SystemConfig()
    policies = FIG8_POLICIES if args.figure8_minmax else POLICIES
    fig8a_only = args.fig8a_k3_rcs_m2 is not None
    radar_overrides = {}
    if fig8a_only:
        if not args.figure8_minmax or not np.isfinite(args.fig8a_k3_rcs_m2) or args.fig8a_k3_rcs_m2 <= 0:
            parser.error("--fig8a-k3-rcs-m2 requires --figure8-minmax and a finite positive RCS")
        k3 = original.FIXED_IDS["adjacent_front"]
        radar_overrides[k3] = replace(radar, rcs_m2=args.fig8a_k3_rcs_m2)
    if args.figure8_minmax:
        dwell_ms = 8.0 if args.min_dwell_ms is None else args.min_dwell_ms
        if not np.isfinite(dwell_ms) or dwell_ms <= 0:
            parser.error("--min-dwell-ms must be finite and positive")
        system = replace(system, min_dwell_s=dwell_ms / 1000)
    elif args.min_dwell_ms is not None:
        parser.error("Fig. 8 parameter overrides require --figure8-minmax")
    references = 1 / np.array([radar.crb_d_reference_m2, radar.crb_v_reference_m2ps2])
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = range(17001, 17001 + args.seeds)
    inputs = (seeds, repeat(warmup[1:] + replay), repeat((warmup[0], len(warmup) - 1)),
              repeat(radar), repeat(system), repeat(references), repeat(policies), repeat(radar_overrides))
    records, frozen = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed, (batch, prior) in zip(seeds, pool.map(replay_seed, *inputs)):
            records.extend(batch)
            frozen.append(prior)
            print(f"Completed SUMO replay for seed {seed}", flush=True)
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
        "seeds": list(seeds), "cycles": len(replay), "methods": policies,
        "figure8_only": args.figure8_minmax,
        "figure8a_only": fig8a_only,
        "rcs_m2_by_target": {t.vehicle_id: radar_overrides.get(t.vehicle_id, radar).rcs_m2 for t in targets},
        "minimum_dwell_s": system.min_dwell_s,
        "minmax_objective": "minimize the largest unweighted normalized posterior CRB" if args.figure8_minmax else None,
        "frozen_time_s": time, "frozen_prior_covariances_by_seed": frozen,
        "shared_prior_covariance": covariance.tolist(),
        "target_order": [t.vehicle_id for t in targets],
        "reference_target": scan["selected_vehicle_id"],
        "fixed_communication_snr_linear": snr,
        "initial_discovery_time_s": warmup[0][0],
        "unplotted_tracking_warmup_updates": len(warmup) - 1,
        "acceleration_psd_m2_s3": ACCEL_PSD,
        "plotted_metric": "omegaCRB = sum_k omega_k * (1/(Jdd_k*d_ref) + 1/(Jvv_k*v_ref))",
        "evaluation_weights": "Common physical TTC weights, normalized to sum K",
        "allocation_metric": "Original recursive posterior CRB, used only in allocation and diagnostics",
        "time_domain_fim": "Current simulated measurement FIM with actual range and actual gain of the scheduled beams",
        "controlled_scan_fim": "Measurement FIM at each prescribed target state with matched beams",
    }, indent=2), encoding="utf-8")
    if fig8a_only:
        return
    if not args.figure8_minmax:
        write_csv(args.output / "fig7.csv", convergence(targets, radar, fixed_system, prior_info, references))
        return
    write_csv(args.output / "fig8b.csv", sensitivity(targets, radar, fixed_system, prior_info,
                                                   references, scan, "range_scales", policies))
    velocity = sensitivity(targets, radar, fixed_system, prior_info, references, scan, "closing_speed_scales", policies)
    write_csv(args.output / "fig8c.csv", velocity)


if __name__ == "__main__":
    main()
