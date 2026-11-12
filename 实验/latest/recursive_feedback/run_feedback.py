"""Replay the recorded SUMO route with causal radar tracking and allocation."""
import argparse
import csv
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = HERE / "inputs"
SOURCE = REPO / "实验/latest/fig8a/source_snapshot"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE.parent))
import main as original
import model
from allocation_methods import CurrentState, direct_dwell, optimization_weights

METHODS = ("Posterior allocation", "Single-frame allocation")
MINIMAX = "MinMax-AO"
INIT_COV = np.diag([0.2**2, 0.1**2])
ANGLE_NOISE = np.deg2rad(0.2)
ACCEL_PSD = 0.02  # m²/s³, continuous white relative acceleration


def load_traffic(path):
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            state = model.VehicleState(**{
                f.name: (row[f.name] if f.type in (str, "str") else
                         int(row[f.name]) if f.type in (int, "int") else float(row[f.name]))
                for f in fields(model.VehicleState)
            })
            groups.setdefault(state.time_s, []).append(state)
    return sorted(groups.items())


def predict(mean, covariance, dt):
    transition = np.array([[1.0, -dt], [0.0, 1.0]])  # Positive v means closing.
    process_noise = ACCEL_PSD * np.array([[dt**3 / 3, -dt**2 / 2], [-dt**2 / 2, dt]])
    return mean @ transition.T, transition @ covariance @ transition.T + process_noise


def information(dwell, power, coefficients):
    return power * coefficients * np.column_stack((dwell, dwell**3))


def posterior(prior_information, measurement_information):
    updated = prior_information.copy()
    updated[:, 0, 0] += measurement_information[:, 0]
    updated[:, 1, 1] += measurement_information[:, 1]
    return np.linalg.inv(updated)


def weighted_bound(covariance, weights, references):
    return weights @ (np.diagonal(covariance, axis1=-2, axis2=-1) @ references)


def allocate_dwell(total, power, minimum, coefficients, prior_info, weights, references, recursive):
    # This KKT solution remains valid for the original single-frame objective.
    instant, _ = model._allocate_convex_dwell(
        references[0] / coefficients[:, 0], references[1] / coefficients[:, 1],
        weights, total, minimum, power,
    )
    if not recursive or total <= len(weights) * minimum + 1e-12:
        return instant
    floor = minimum / total
    uniform = np.full(len(weights), 1 / len(weights))
    scale = weighted_bound(
        posterior(prior_info, information(uniform * total, power, coefficients)), weights, references
    )

    def objective(shares):
        dwell = shares * total
        covariance = posterior(prior_info, information(dwell, power, coefficients))
        # d tr(W P)/dJ_j = -[P W P]_jj.
        sensitivity = np.einsum("kij,j,kij->ki", covariance, references, covariance)
        diagonal_gradient = power * total * coefficients * np.column_stack((np.ones(len(dwell)), 3 * dwell**2))
        gradient = -weights * np.sum(sensitivity * diagonal_gradient, axis=1)
        return weighted_bound(covariance, weights, references) / scale, gradient / scale

    constraint = {"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones_like(x)}
    vertices = floor + (1 - len(weights) * floor) * np.eye(len(weights))
    starts = (uniform, instant / total, *vertices)
    best = min(starts, key=lambda x: objective(x)[0])
    successes = 0
    for start in starts:
        result = minimize(objective, start, jac=True, method="SLSQP",
                          bounds=[(floor, 1.0)] * len(weights), constraints=constraint,
                          options={"ftol": 1e-10, "maxiter": 60})
        if result.success and abs(result.x.sum() - 1) < 1e-9 and result.x.min() >= floor - 1e-10:
            successes += 1
            if objective(result.x)[0] < objective(best)[0]:
                best = result.x
    if successes == 0:
        raise RuntimeError("All posterior P1 starts failed")
    dwell = best * total
    dwell[np.argmax(dwell)] += total - dwell.sum()
    return dwell


def minimax_dwell(total, power, minimum, coefficients, prior_info, references):
    """Minimize the largest normalized posterior CRB under the dwell budget."""
    count = len(coefficients)
    floor = minimum / total
    if 1 - count * floor <= 1e-12:
        return np.full(count, total / count)

    alpha = prior_info[:, 0, 0]
    beta = prior_info[:, 1, 1]
    cross_sq = prior_info[:, 0, 1] ** 2
    range_scale = power * coefficients[:, 0] * total
    velocity_scale = power * coefficients[:, 1] * total**3
    ref_d, ref_v = references
    max_share = 1 - (count - 1) * floor

    def bound(k, share):
        dd = alpha[k] + range_scale[k] * share
        vv = beta[k] + velocity_scale[k] * share**3
        return (ref_d * vv + ref_v * dd) / (dd * vv - cross_sq[k])

    floor_bounds = [bound(k, floor) for k in range(count)]
    cap_bounds = [bound(k, max_share) for k in range(count)]

    def required_shares(threshold):
        shares = np.empty(count)
        for k in range(count):
            if threshold >= floor_bounds[k]:
                shares[k] = floor
            elif threshold <= cap_bounds[k]:
                shares[k] = max_share
            else:
                shares[k] = brentq(lambda x: bound(k, x) - threshold, floor, max_share, xtol=1e-14)
        return shares

    lower, upper = max(cap_bounds), max(floor_bounds)
    if required_shares(lower).sum() <= 1 + 1e-13:
        threshold = lower
    else:
        threshold = upper * brentq(lambda level: required_shares(upper * level).sum() - 1,
                                   lower / upper, 1.0, xtol=1e-13)
    shares = required_shares(threshold)
    shares[np.argmax(shares)] += 1 - shares.sum()
    return shares * total


def evaluate_candidate(ts, ps, system, coefficients, prior_info, weights, references,
                       recursive, carry=None, direct=None, *, minimax=False):
    if minimax:
        dwell = minimax_dwell(ts, ps, system.min_dwell_s, coefficients, prior_info, references)
    elif direct is None:
        dwell = allocate_dwell(ts, ps, system.min_dwell_s, coefficients, prior_info,
                               weights, references, recursive)
    else:
        dwell = direct_dwell(direct, weights, ts, system.min_dwell_s)
    candidates = (dwell, carry * (ts / carry.sum())) if carry is not None and direct is None else (dwell,)
    best = None
    for candidate in candidates:
        measurement = information(candidate, ps, coefficients)
        post = posterior(prior_info, measurement)
        target_bounds = np.reciprocal(measurement) @ references
        bound = weighted_bound(post, weights, references)
        if minimax:
            objective = float(np.max(np.diagonal(post, axis1=-2, axis2=-1) @ references))
        else:
            objective = bound if recursive else weights @ target_bounds
        if best is None or objective < best["objective"]:
            best = dict(time=ts, power=ps, dwell=candidate, post=post,
                        bound=bound, objective=objective)
    if (abs(best["dwell"].sum() - ts) > 1e-10 or best["dwell"].min() < system.min_dwell_s - 1e-10
            or min(model.constraint_margins(ts, ps, len(weights), system).values()) < -1e-9):
        raise RuntimeError("Resource allocation violated a constraint")
    return best


def allocate(system, coefficients, prior_info, weights, references, recursive, previous, direct=None,
             *, minimax=False):
    # Each coordinate is monotone in information. Scaling all dwell times
    # upward while holding total energy preserves Jdd and increases Jvv.
    maximum_time = system.sensing_time_qos_max_s - 1e-12
    maximum_power = min(system.sensing_power_max_w, system.sensing_energy_max_j / maximum_time)
    time, power = previous or (system.initial_sensing_time_s, system.initial_sensing_power_w)
    time = np.clip(time, len(weights) * system.min_dwell_s, maximum_time)
    power = min(power, maximum_power)
    steps = []
    for stage, ts, ps in (("initial", time, power), ("time", maximum_time, power),
                          ("power", maximum_time, maximum_power)):
        if steps and (ts, ps) == (steps[-1]["time"], steps[-1]["power"]):
            steps.append(dict(steps[-1], stage=stage))
            continue
        carry = steps[-1]["dwell"] if steps else None
        best = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                  references, recursive, carry, direct, minimax=minimax)
        if steps and best["objective"] > steps[-1]["objective"] * (1 + 1e-10):
            raise RuntimeError("Block update increased its objective")
        steps.append(dict(best, stage=stage))
    return steps


def track(replay, warmup, radar, system, seed, references, methods=METHODS):
    rng = np.random.default_rng(seed)
    warm_time, _, warm_targets = warmup
    discovery = np.array([(target.distance_m, target.closing_speed_mps) for target in warm_targets])
    init_mean = discovery + rng.standard_normal(discovery.shape) * np.sqrt(INIT_COV.diagonal())
    init_angle = np.array([target.azimuth_rad for target in warm_targets]) + ANGLE_NOISE * rng.standard_normal(len(discovery))
    histories = {}
    for method in methods:
        histories[method] = (init_mean, np.broadcast_to(INIT_COV, (len(discovery), 2, 2)),
                             init_angle, None, None)
    last_time = warm_time
    for cycle, (time, ego, targets) in enumerate(replay):
        dt = time - last_time
        truth = np.array([(target.distance_m, target.closing_speed_mps) for target in targets])
        truth_weights = model.compute_ttc_weights(targets).omega
        true_angle = np.array([target.azimuth_rad for target in targets])
        # The current angle reading becomes available only after resource selection.
        next_angle = true_angle + ANGLE_NOISE * rng.standard_normal(len(targets))
        noise = rng.standard_normal(truth.shape)
        snr, _ = original.communication_snr_from_rsu(ego)
        cycle_system = replace(system, communication_snr_linear=snr)
        for method, (mean, cov, angle, previous, previous_bound) in histories.items():
            predicted, prior = predict(mean, cov, dt)
            if np.any(predicted[:, 0] <= 0):
                raise RuntimeError("Predicted range outside the observation model")
            # Distance, closing speed and angle are from history. Geometry
            # fields unused by scheduling are inherited from the warm-up snapshot.
            predicted_targets = [
                replace(target, distance_m=x[0], closing_speed_mps=x[1], azimuth_rad=a)
                for target, x, a in zip(warm_targets, predicted, angle)
            ]
            weights = model.compute_ttc_weights(predicted_targets).omega
            beams = [model.closed_form_beams(a, radar) for a in angle]
            state = CurrentState(predicted_targets, beams, radar, tuple(weights))
            allocation_weights = optimization_weights(method, state)
            coefficients = np.array([
                model.information_coefficients_per_watt(target, beam, radar)[:2]
                for target, beam in zip(predicted_targets, beams)
            ])
            direct = method if method in ("TTC-PD", "UD") else None
            steps = allocate(cycle_system, coefficients, np.linalg.inv(prior), allocation_weights, references,
                             method != METHODS[1], previous, direct, minimax=method == MINIMAX)
            best = steps[-1]
            dwell, ts, power, post = best["dwell"], best["time"], best["power"], best["post"]
            assumed_info = information(dwell, power, coefficients)
            actual_coefficients = []
            for target, beam in zip(targets, beams):
                actual = model.closed_form_beams(target.azimuth_rad, radar)
                pointed = replace(
                    beam,
                    tx_gain_linear=abs(np.vdot(actual.tx_steering, beam.tx_beam))**2,
                    rx_gain_linear=abs(np.vdot(beam.rx_beam, actual.rx_steering))**2,
                )
                actual_coefficients.append(model.information_coefficients_per_watt(target, pointed, radar)[:2])
            realized_info = information(dwell, power, np.array(actual_coefficients))
            measurement = truth + noise / np.sqrt(realized_info)
            gain = post * assumed_info[:, None, :]
            updated = predicted + np.einsum("kij,kj->ki", gain, measurement - predicted)
            if np.linalg.eigvalsh(prior - post).min() < -1e-12 or np.linalg.eigvalsh(post).min() <= 0:
                raise RuntimeError("Posterior failed covariance ordering")
            histories[method] = updated, post, next_angle, (ts, power), weighted_bound(post, weights, references)
            for target, truth_weight, weight, estimate, observation, actual_state, forecast, bearing, prior_k, post_k, old_cov, tk, ji, jr in zip(
                targets, truth_weights, weights, updated, measurement, truth, predicted, angle, prior, post, cov,
                dwell, assumed_info, realized_info,
            ):
                yield dict(seed=seed, cycle=cycle, time_s=time, method=method, target=target.vehicle_id,
                           previous_weighted_pcrb=previous_bound,
                           truth_weight=truth_weight, predicted_weight=weight,
                           predicted_azimuth_rad=bearing,
                           predicted_range_m=forecast[0], predicted_closing_mps=forecast[1],
                           previous_post_dd=old_cov[0, 0], previous_post_dv=old_cov[0, 1], previous_post_vv=old_cov[1, 1],
                           prior_dd=prior_k[0, 0], prior_dv=prior_k[0, 1], prior_vv=prior_k[1, 1],
                           post_dd=post_k[0, 0], post_dv=post_k[0, 1], post_vv=post_k[1, 1],
                           predicted_Jdd=ji[0], predicted_Jvv=ji[1], realized_Jdd=jr[0], realized_Jvv=jr[1],
                           estimated_range_m=estimate[0], estimated_closing_mps=estimate[1],
                           measured_range_m=observation[0], measured_closing_mps=observation[1],
                           true_range_m=actual_state[0], true_closing_mps=actual_state[1],
                           Ts_s=ts, Ps_w=power, dwell_s=tk, communication_rate_bps=model.communication_rate_bps(ts, cycle_system),
                           scheduled_prior_wPCRB=weighted_bound(prior, weights, references),
                           initial_wPCRB=weighted_bound(steps[0]["post"], weights, references),
                           time_update_wPCRB=weighted_bound(steps[1]["post"], weights, references),
                           optimized_wPCRB=weighted_bound(post, weights, references),
                           initial_objective=steps[0]["objective"],
                           optimized_objective=best["objective"])
        last_time = time


def extract_snapshots(rows):
    for time, states in rows:
        ego, targets, missing = original.fixed_target_observations(states, "ego")
        if missing:
            raise RuntimeError(f"Incomplete snapshot at {time} s: {missing}")
        yield time, ego, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    traffic = INPUTS / "traffic.csv"
    warmup_file = INPUTS / "warmup_traffic.csv"
    replay = list(extract_snapshots(load_traffic(traffic)))
    warmup = next(extract_snapshots(load_traffic(warmup_file)[-1:]))
    radar, system = model.RadarConfig(), model.SystemConfig()
    references = 1 / np.array([radar.crb_d_reference_m2, radar.crb_v_reference_m2ps2])
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "tracking.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for seed in range(17001, 17001 + args.seeds):
            for row in track(replay, warmup, radar, system, seed, references):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
            print(f"Finished seed {seed}", flush=True)
    manifest = {
        "replicates": args.seeds,
        "cycles_per_replicate": len(replay),
        "first_seed": 17001,
        "warmup_time_s": warmup[0],
        "acceleration_psd_m2_s3": ACCEL_PSD,
        "discovery_covariance": INIT_COV.tolist(),
        "angle_noise_std_degrees": float(np.rad2deg(ANGLE_NOISE)),
        "references_inverse": references.tolist(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (traffic, warmup_file, SOURCE / "model.py", SOURCE / "main.py")},
    }
    (args.output / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
