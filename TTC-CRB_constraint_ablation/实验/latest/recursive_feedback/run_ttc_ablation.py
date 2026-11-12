"""Run the independent posterior TTC-CRB constraint ablation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from itertools import groupby, repeat
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from run_feedback import ACCEL_PSD, ANGLE_NOISE, HERE, INIT_COV, INPUTS, extract_snapshots, load_traffic, model, track
from run_figures_7_9 import write_csv
from ttc_precision import TTCPrecision


ARMS = {"unconstrained": None, "ttc_2pct": .02, "ttc_5pct": .05, "ttc_10pct": .10}
NEAR_ZERO_MPS = .05  # Diagnostic flag only.
MARGINS = ("time_lower_margin_s", "time_upper_margin_s", "energy_margin_j",
           "power_lower_margin_w", "power_upper_margin_w", "qos_rate_margin_bps")


def replay_seed(seed, replay, discovery, radar, system, references):
    cycles, targets = [], []
    for arm, scale in ARMS.items():
        limit = None if scale is None else scale**2
        replay_rows = track(replay, discovery, radar, system, seed, references, ("TTC-AO",),
                            ttc_gamma_limit=limit)
        for _, group in groupby(replay_rows, key=lambda row: row["cycle"]):
            rows = list(group)
            first = rows[0]
            feasible = first["allocation_status"] == "feasible"
            base = dict(arm=arm, seed=seed, cycle=first["cycle"], time_s=first["time_s"],
                        warmup=int(first["time_s"] < 0), status=first["allocation_status"],
                        reason=first["infeasible_reason"])
            state = np.array([[r["predicted_range_m"], r["predicted_closing_mps"]] for r in rows])
            covariance = np.array([[[r["post_dd"], r["post_dv"]], [r["post_dv"], r["post_vv"]]] for r in rows])
            precision = TTCPrecision(state, limit)
            gamma = precision.gamma(covariance)
            if feasible and limit is not None and not np.all(gamma[precision.active] <= limit*(1+1e-9)):
                raise AssertionError(f"Hard constraint violated: {base}")
            dwell = np.array([r["dwell_s"] for r in rows]) if feasible else None
            truth_weights = np.array([r["truth_weight"] for r in rows])
            pcrb = truth_weights @ (np.diagonal(covariance, axis1=1, axis2=2) @ references)
            wcrb = None
            if feasible:
                j = np.array([[r["realized_Jdd"], r["realized_Jvv"]] for r in rows])
                wcrb = truth_weights @ (np.reciprocal(j) @ references)
            margins = {name: first.get(name) for name in MARGINS}
            qos = None if not feasible else int(margins["qos_rate_margin_bps"] >= -1e-9)
            energy = None if not feasible else int(margins["energy_margin_j"] >= -1e-9)
            if feasible:
                assert abs(dwell.sum() - first["Ts_s"]) <= 1e-10
                assert dwell.min() >= system.min_dwell_s - 1e-10
                assert qos and energy and min(margins.values()) >= -1e-9
            cycles.append(dict(**base, gamma_limit=limit, n_active=int(precision.active.sum()),
                               weighted_wcrb=wcrb, weighted_wpcrb=pcrb, Ts_s=first.get("Ts_s"),
                               Ps_w=first.get("Ps_w"), max_Ts_s=first["max_Ts_s"],
                               communication_rate_bps=first.get("communication_rate_bps"),
                               qos_pass=qos, energy_pass=energy, **margins,
                               dwell_sum_error_s=None if not feasible else abs(dwell.sum()-first["Ts_s"]),
                               minimum_dwell_margin_s=None if not feasible else dwell.min()-system.min_dwell_s,
                               rejected_p2_candidates=first.get("rejected_stages", "")))
            for k, row in enumerate(rows):
                active = bool(precision.active[k])
                d, v = state[k]
                ttc = d / v if active else None
                value = gamma[k] if active else None
                targets.append(dict(**base, target=f"k{k+1}", vehicle_id=row["target"],
                                    active=int(active), near_zero_positive=int(0 < v <= NEAR_ZERO_MPS),
                                    predicted_range_m=d, predicted_closing_mps=v, predicted_TTC_s=ttc,
                                    predicted_risk_weight=row["predicted_weight"], evaluation_risk_weight=truth_weights[k],
                                    P_dd=covariance[k, 0, 0], P_dv=covariance[k, 0, 1], P_vv=covariance[k, 1, 1],
                                    bound_source="posterior" if feasible else "prediction_only",
                                    posterior_TTC_CRB_s2=(value * ttc**2 if active and feasible else None),
                                    gamma_plus=value if feasible else None,
                                    prediction_only_gamma=None if feasible else value, gamma_limit=limit,
                                    gamma_margin=limit-value if feasible and active and limit is not None else None,
                                    dwell_s=None if not feasible else dwell[k], required_dwell_s=row.get("required_dwell_s"),
                                    predicted_Jdd=row.get("predicted_Jdd"), predicted_Jvv=row.get("predicted_Jvv"),
                                    realized_Jdd=row.get("realized_Jdd"), realized_Jvv=row.get("realized_Jvv")))
    return cycles, targets


def average(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def summarize(cycles, targets, output):
    baseline = {(r["seed"], r["cycle"]): r for r in cycles if r["arm"] == "unconstrained" and not r["warmup"]}
    threshold_rows, target_rows = [], []
    for arm, scale in ARMS.items():
        arm_cycles = [r for r in cycles if r["arm"] == arm]
        formal = [r for r in arm_cycles if not r["warmup"]]
        feasible = [r for r in formal if r["status"] == "feasible"]
        paired = [baseline[r["seed"], r["cycle"]] for r in feasible]
        data = [r for r in targets if r["arm"] == arm]
        active = [r for r in data if r["active"] and r["status"] == "feasible" and not r["warmup"]]
        warm_fail = sum(r["warmup"] and r["status"] == "infeasible" for r in arm_cycles)
        causes = Counter(r["reason"] for r in formal if r["status"] == "infeasible")
        threshold_rows.append(dict(arm=arm, relative_std_scale=scale, gamma_limit=None if scale is None else scale**2,
                                   formal_cycles=len(formal), feasible_cycles=len(feasible), infeasible_cycles=len(formal)-len(feasible),
                                   warmup_infeasible_cycles=warm_fail, feasible_fraction=len(feasible)/len(formal),
                                   mean_wcrb=average(r["weighted_wcrb"] for r in feasible),
                                   baseline_wcrb_same_cycles=average(r["weighted_wcrb"] for r in paired),
                                   mean_wpcrb=average(r["weighted_wpcrb"] for r in feasible),
                                   baseline_wpcrb_same_cycles=average(r["weighted_wpcrb"] for r in paired),
                                   mean_wpcrb_all_cycles=average(r["weighted_wpcrb"] for r in formal),
                                   mean_Ts_s=average(r["Ts_s"] for r in feasible), mean_Ps_w=average(r["Ps_w"] for r in feasible),
                                   mean_energy_j=average(r["Ts_s"]*r["Ps_w"] for r in feasible),
                                   qos_pass_fraction_feasible=average(r["qos_pass"] for r in feasible),
                                   energy_pass_fraction_feasible=average(r["energy_pass"] for r in feasible),
                                   n_active_feasible_targets=len(active),
                                   max_active_gamma=max((r["gamma_plus"] for r in active), default=None),
                                   min_gamma_margin=min((r["gamma_margin"] for r in active), default=None) if scale else None,
                                   near_zero_targets_all_cycles=sum(r["near_zero_positive"] for r in data),
                                   infeasible_reasons=json.dumps(causes, sort_keys=True)))
        for k in range(1, 5):
            rows = [r for r in data if r["target"] == f"k{k}" and not r["warmup"] and r["status"] == "feasible"]
            matching = {(r["seed"], r["cycle"]) for r in rows}
            paired_rows = [r for r in targets if r["arm"] == "unconstrained" and r["target"] == f"k{k}"
                           and (r["seed"], r["cycle"]) in matching]
            values = [r["gamma_plus"] for r in rows if r["active"]]
            target_rows.append(dict(arm=arm, target=f"k{k}", feasible_cycles=len(rows), active_targets=len(values),
                                    mean_gamma=average(values), max_gamma=max(values, default=None),
                                    mean_dwell_ms=average(r["dwell_s"]*1000 for r in rows),
                                    baseline_dwell_ms_same_cycles=average(r["dwell_s"]*1000 for r in paired_rows),
                                    mean_required_dwell_ms=average(r["required_dwell_s"]*1000 for r in rows
                                                                  if r["required_dwell_s"] is not None)))
    write_csv(output / "threshold_summary.csv", threshold_rows)
    write_csv(output / "target_summary.csv", target_rows)

    baseline_targets = {(r["seed"], r["cycle"], r["target"]): r for r in targets
                        if r["arm"] == "unconstrained" and not r["warmup"]}
    matches = []
    for arm in tuple(ARMS)[1:]:
        limit = ARMS[arm]**2
        for k in range(1, 5):
            rows = [r for r in targets if r["arm"] == arm and r["target"] == f"k{k}" and
                    not r["warmup"] and r["status"] == "feasible"]
            paired = [baseline_targets[r["seed"], r["cycle"], r["target"]] for r in rows]
            paired_active = [b["gamma_plus"] for r, b in zip(rows, paired) if r["active"] and b["active"]]
            constrained = [r["gamma_plus"] for r, b in zip(rows, paired) if r["active"] and b["active"]]
            matches.append(dict(arm=arm, target=f"k{k}", gamma_limit=limit, paired_active_cycles=len(paired_active),
                                baseline_mean_gamma=average(paired_active), constrained_mean_gamma=average(constrained),
                                baseline_pass_fraction=average(v <= limit*(1+1e-9) for v in paired_active),
                                constrained_pass_fraction=average(v <= limit*(1+1e-9) for v in constrained)))
    write_csv(output / "paired_target_comparison.csv", matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()
    if min(args.workers, args.seeds) < 1:
        parser.error("workers and seeds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    warmup_path, traffic_path = INPUTS / "warmup_traffic.csv", INPUTS / "traffic.csv"
    warmup = list(extract_snapshots(load_traffic(warmup_path)))
    formal = list(extract_snapshots(load_traffic(traffic_path)))
    replay = warmup[1:] + formal
    radar, system = model.RadarConfig(), model.SystemConfig()
    references = 1 / np.array([radar.crb_d_reference_m2, radar.crb_v_reference_m2ps2])
    seeds = list(range(17001, 17001 + args.seeds))
    manifest = dict(experiment="TTC-CRB constraint ablation", arms_relative_std_scale=ARMS,
                    arms_gamma_limit={arm: None if x is None else x**2 for arm, x in ARMS.items()},
                    primary_arm="ttc_5pct",
                    threshold_basis="Preset 5% relative standard deviation bound scale, with 2% tight and 10% loose sensitivity.",
                    seeds=seeds, formal_cycles=len(formal), warmup_updates=len(warmup)-1, radar=asdict(radar),
                    system=asdict(system), acceleration_psd=ACCEL_PSD, angle_noise_rad=ANGLE_NOISE,
                    initial_covariance=INIT_COV.tolist(), near_zero_flag_mps=NEAR_ZERO_MPS,
                    infeasible_policy="Prediction only; no sensing allocation or current measurement.",
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                            (warmup_path, traffic_path, HERE/"run_feedback.py", HERE/"ttc_precision.py",
                             HERE/"run_ttc_ablation.py", HERE/"model.py", HERE/"replay_scene.py")})
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cycles, targets = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "run.log").open("w", encoding="utf-8") as log:
        jobs = pool.map(replay_seed, seeds, repeat(replay), repeat(warmup[0]), repeat(radar), repeat(system), repeat(references))
        for seed, (cycle_rows, target_rows) in zip(seeds, jobs):
            assert len(cycle_rows) == len(ARMS)*len(replay)
            assert len(target_rows) == len(cycle_rows)*4
            cycles.extend(cycle_rows)
            targets.extend(target_rows)
            message = f"Seed {seed}: {len(cycle_rows)} cycle rows, {len(target_rows)} target rows"
            print(message, flush=True)
            print(message, file=log, flush=True)
    write_csv(args.output / "cycles.csv", cycles)
    write_csv(args.output / "targets.csv", targets)
    write_csv(args.output / "infeasible_cycles.csv", [r for r in cycles if r["status"] == "infeasible"] or
              [dict(status="no_infeasible_cycles")])
    summarize(cycles, targets, args.output)


if __name__ == "__main__":
    main()
