"""Recompute the original figures with posterior TTC precision constraints."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import repeat
import json
from pathlib import Path

import numpy as np

from run_feedback import INPUTS, extract_snapshots, load_traffic, model, original
from run_figures_7_9 import FIG8_POLICIES, POLICIES, convergence, read_csv, replay_seed, sensitivity, write_csv
from plot_figures_7_9 import summarize_replay


def run_seed(seed, replay, initialization, radar, references, configurations):
    result = {}
    for name, system, policies, radar_overrides, limit, feedback in configurations:
        try:
            result[name] = replay_seed(seed, replay, initialization, radar, system, references, policies,
                                       radar_overrides, limit, feedback)
        except RuntimeError as exc:
            raise RuntimeError(f"{name} | {exc}") from exc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reference-std", type=float, default=.064)
    parser.add_argument("--comparison-std", type=float, default=.11)
    parser.add_argument("--feedback-std", type=float, default=.12)
    parser.add_argument("--only-reference", action="store_true", help="Recompute just Fig. 7 and Fig. 9")
    args = parser.parse_args()
    levels = (args.reference_std, args.comparison_std, args.feedback_std)
    if args.workers < 1 or not all(np.isfinite(x) and x > 0 for x in levels):
        parser.error("Workers and relative standard-deviation bounds must be positive.")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(17001, 17017))
    replay = list(extract_snapshots(load_traffic(INPUTS / "traffic.csv")))
    warm = list(extract_snapshots(load_traffic(INPUTS / "warmup_traffic.csv")))
    radar, system = model.RadarConfig(), model.SystemConfig()
    system8 = replace(system, min_dwell_s=.008)
    overrides = {original.FIXED_IDS["adjacent_front"]: replace(radar, rcs_m2=2.5)}
    references = 1 / np.array([radar.crb_d_reference_m2, radar.crb_v_reference_m2ps2])
    configs = (
        ("reference", system, POLICIES, {}, args.reference_std**2, True),
        ("fig8_minmax", system8, FIG8_POLICIES, {}, args.comparison_std**2, True),
        ("fig8a_weak_echo", system8, FIG8_POLICIES, overrides, args.comparison_std**2, True),
        ("feedback_on", system, POLICIES, {}, args.feedback_std**2, True),
        ("feedback_off", system, POLICIES, {}, args.feedback_std**2, False),
    )
    if args.only_reference:
        configs = configs[:1]
    batches = {conf[0]: [] for conf in configs}
    priors = {conf[0]: [] for conf in configs}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = pool.map(run_seed, seeds, repeat(warm[1:] + replay), repeat((warm[0], len(warm)-1)),
                        repeat(radar), repeat(references), repeat(configs))
        for seed, result in zip(seeds, jobs):
            for name, (rows, prior) in result.items():
                batches[name].extend(rows)
                priors[name].append(prior)
            print(f"Seed {seed}: completed {sum(len(config[2]) for config in configs)} method/experiment combinations",
                  flush=True)

    time, ego, targets = replay[-1]
    snr, _ = original.communication_snr_from_rsu(ego)
    scan = json.loads((INPUTS / "scan_settings.json").read_text(encoding="utf-8"))
    assert time == scan["baseline_time_s"]
    assert [target.vehicle_id for target in targets][1] == scan["selected_vehicle_id"]
    for name, conf_system, policies, radar_overrides, limit, feedback in configs:
        directory = out if name == "reference" else out / name
        directory.mkdir(exist_ok=True)
        design = dict(seeds=seeds, cycles=len(replay), methods=policies, gamma_limit=limit,
                      relative_ttc_std_bound=float(np.sqrt(limit)), system=asdict(conf_system),
                      ttc_constraint_applied_from_time_s=0.0, warmup_cycles=len(warm)-1,
                      radar=asdict(radar), scheduler_feedback=feedback,
                      figure8_only=name.startswith("fig8"), figure8a_only=name=="fig8a_weak_echo",
                      rcs_m2_by_target={t.vehicle_id: radar_overrides.get(t.vehicle_id, radar).rcs_m2 for t in targets},
                      shared_prior_covariance=np.mean(priors[name], axis=0).tolist(),
                      frozen_prior_covariances_by_seed=priors[name],
                      target_order=[t.vehicle_id for t in targets],
                      fixed_communication_snr_linear=snr, fixed_snapshot_time_s=time,
                      input_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                                    (INPUTS / "traffic.csv", INPUTS / "warmup_traffic.csv",
                                     INPUTS / "scenario_settings.csv", INPUTS / "scan_settings.json")})
        (directory / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
        write_csv(directory / "replay_by_seed.csv", batches[name])
        if name in ("reference", "fig8a_weak_echo"):
            summarize_replay(directory, design)
        prior_info = np.linalg.inv(np.asarray(design["shared_prior_covariance"]))
        fixed_system = replace(conf_system, communication_snr_linear=snr)
        if name == "reference":
            write_csv(out / "fig7.csv", convergence(targets, radar, fixed_system, prior_info, references, limit))
        if name == "fig8_minmax":
            for panel, variable in (("fig8b", "range_scales"), ("fig8c", "closing_speed_scales")):
                write_csv(out / f"{panel}.csv", sensitivity(targets, radar, fixed_system, prior_info,
                                                           references, scan, variable, policies, limit))

    if args.only_reference:
        return
    write_csv(out / "fig8a.csv", read_csv(out / "fig8a_weak_echo" / "fig8a.csv"))
    feedback_scores = {name: np.array([r["weighted_pcrb"] for r in batches[name]]).reshape(len(seeds), len(replay))
                       for name in ("feedback_on", "feedback_off")}
    write_csv(out / "fig10a_total_wcrb.csv", [
        dict(cycle_id=i, simulation_time=row[0], wCRB_ON=feedback_scores["feedback_on"][:, i].mean(),
             wCRB_OFF=feedback_scores["feedback_off"][:, i].mean()) for i, row in enumerate(replay)
    ])


if __name__ == "__main__":
    main()
