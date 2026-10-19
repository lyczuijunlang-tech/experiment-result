"""Stage-3 ISAC physical model and nested P1/P2 optimization.

Formula chain implemented here:
SUMO state -> (d, radial closing speed, azimuth) -> TTC -> omega
-> closed-form Tx/Rx beams -> monostatic radar link -> LFM FIM/CRB
-> P1 target dwell allocation -> P2 alternating optimization.

No q_k sensing-difficulty proxy, risk amplification, or curve smoothing is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, inf, isfinite, log2, pi, radians, sin, sqrt
from typing import Sequence

import numpy as np


C0 = 299_792_458.0
KB = 1.380_649e-23
SLOT_ORDER = ("same_front", "same_rear", "adjacent_front", "adjacent_rear")


@dataclass(frozen=True)
class SimulationConfig:
    step_s: float = 0.1
    duration_s: float = 40.0
    # Let SUMO's car-following state settle before t=0 is recorded. This avoids
    # treating insertion transients as part of the 30 s experiment.
    warmup_s: float = 2.0
    ego_id: str = "ego"
    cutin_id: str = "cutin"
    cutin_start_s: float = 14.0
    cutin_duration_s: float = 4.0
    preferred_adjacent_delta: int = 1
    max_target_range_m: float = 220.0
    seed: int = 17


@dataclass(frozen=True)
class RadarConfig:
    # 77 GHz automotive carrier and 200 MHz ideal rectangular LFM spectrum.
    carrier_hz: float = 77e9
    bandwidth_hz: float = 200e6
    tx_elements: int = 4
    rx_elements: int = 4
    rcs_m2: float = 10.0
    system_loss_linear: float = 2.0
    noise_temperature_k: float = 290.0
    noise_figure_linear: float = 10 ** (6.0 / 10.0)
    # CRB reference variances used only to make the scalar CRB dimensionless.
    crb_d_reference_m2: float = 1.0**2
    crb_v_reference_m2ps2: float = 1.0**2
    lfm_pulse_duration_s: float = 5e-6

    @property
    def wavelength_m(self) -> float:
        return C0 / self.carrier_hz

    @property
    def element_spacing_m(self) -> float:
        return self.wavelength_m / 2.0

    @property
    def rms_bandwidth_hz(self) -> float:
        # Flat LFM spectrum over [-B/2, B/2]: E[f^2] = B^2/12.
        return self.bandwidth_hz / sqrt(12.0)

    @property
    def noise_psd_w_per_hz(self) -> float:
        return KB * self.noise_temperature_k * self.noise_figure_linear


@dataclass(frozen=True)
class SystemConfig:
    # One ISAC frame is 50 ms; the SUMO/resource decision period is 0.1 s.
    frame_s: float = 0.050
    min_dwell_s: float = 0.002
    communication_bandwidth_hz: float = 20e6
    communication_snr_linear: float = 10.0  # 10 dB
    minimum_rate_bps: float = 10e6
    sensing_energy_max_j: float = 1.0e-3
    sensing_power_max_w: float = 0.050
    initial_sensing_time_s: float = 0.025
    initial_sensing_power_w: float = 10 ** ((13.5 - 30.0) / 10.0)
    # Strict paper P2 objective: min C_risk^*(Ts, Ps). Resource quantities
    # remain hard constraints and are not added as regularizers.
    p2_objective: str = "pure_crb"
    # Under-relaxed block-coordinate AO.  Each block optimum is still solved
    # from the physical P2 objective; relaxation only controls how the online
    # iterate approaches that optimum and makes the convergence trace visible.
    ao_time_relaxation: float = 0.35
    ao_power_relaxation: float = 0.03
    ao_power_relaxation_near_time_bound: float = 0.15
    ao_warm_start_power_backoff: float = 0.995
    ao_eta: float = 1e-6
    ao_max_rounds: int = 100
    scalar_max_iterations: int = 24
    time_search_tolerance_s: float = 1e-7
    power_search_tolerance_w: float = 1e-8
    feasibility_tolerance: float = 1e-10
    infeasible_penalty: float = 1e80

    @property
    def communication_time_min_s(self) -> float:
        # Tc_min = T Rmin / (Bc log2(1+gamma_c)).
        return (
            self.frame_s
            * self.minimum_rate_bps
            / (self.communication_bandwidth_hz * log2(1.0 + self.communication_snr_linear))
        )

    @property
    def sensing_time_qos_max_s(self) -> float:
        return self.frame_s - self.communication_time_min_s


def validate_configs(sim: SimulationConfig, radar: RadarConfig, system: SystemConfig) -> None:
    numeric = [*vars(sim).values(), *vars(radar).values(), *vars(system).values()]
    for value in numeric:
        if isinstance(value, (float, int)) and not isfinite(float(value)):
            raise ValueError("All numeric configuration values must be finite")
    if sim.step_s <= 0 or sim.duration_s <= 0 or sim.cutin_duration_s <= 0 or sim.warmup_s < 0:
        raise ValueError("Simulation time values must be positive (warm-up may be zero)")
    if radar.tx_elements < 1 or radar.rx_elements < 1:
        raise ValueError("Array sizes must be positive")
    positive_radar = (
        radar.carrier_hz,
        radar.bandwidth_hz,
        radar.rcs_m2,
        radar.system_loss_linear,
        radar.noise_temperature_k,
        radar.noise_figure_linear,
        radar.crb_d_reference_m2,
        radar.crb_v_reference_m2ps2,
    )
    if min(positive_radar) <= 0:
        raise ValueError("Radar parameters must be strictly positive")
    if system.p2_objective != "pure_crb":
        raise ValueError("This fixed-target experiment uses the pure-CRB P2 objective")
    if not 0.0 < system.ao_time_relaxation <= 1.0:
        raise ValueError("AO time relaxation must be in (0, 1]")
    if not 0.0 < system.ao_power_relaxation <= 1.0:
        raise ValueError("AO power relaxation must be in (0, 1]")
    if not 0.0 < system.ao_power_relaxation_near_time_bound <= 1.0:
        raise ValueError("AO final power relaxation must be in (0, 1]")
    if not 0.0 < system.ao_warm_start_power_backoff <= 1.0:
        raise ValueError("AO warm-start power backoff must be in (0, 1]")
    if system.communication_time_min_s >= system.frame_s:
        raise ValueError("Communication QoS leaves no sensing time")
    if len(SLOT_ORDER) * system.min_dwell_s > system.sensing_time_qos_max_s:
        raise ValueError("K*Tmin exceeds the QoS-limited sensing-time upper bound")
    margins = constraint_margins(
        system.initial_sensing_time_s,
        system.initial_sensing_power_w,
        len(SLOT_ORDER),
        system,
    )
    if min(margins.values()) < -system.feasibility_tolerance:
        raise ValueError(f"Initial P2 point is infeasible: {margins}")


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + pi) % (2.0 * pi) - pi


def sumo_angle_to_math(angle_deg: float) -> float:
    """SUMO uses north=0 and clockwise-positive; math uses east=0 and CCW."""
    return wrap_angle(radians(90.0 - angle_deg))


@dataclass(frozen=True)
class VehicleState:
    time_s: float
    vehicle_id: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    speed_mps: float
    heading_rad: float
    sumo_angle_deg: float
    road_id: str
    lane_id: str
    lane_index: int
    lane_position_m: float
    lateral_lane_position_m: float
    acceleration_mps2: float
    length_m: float
    width_m: float
    velocity_source: str


@dataclass(frozen=True)
class TargetObservation:
    slot: str
    vehicle: VehicleState
    distance_m: float
    closing_speed_mps: float
    azimuth_rad: float
    longitudinal_m: float
    lateral_m: float

    @property
    def vehicle_id(self) -> str:
        return self.vehicle.vehicle_id


def relative_observation(slot: str, ego: VehicleState, target: VehicleState) -> TargetObservation:
    """Equations: r=p_k-p_ego, d=||r||, v=-DeltaV dot r/d, phi=atan2 in ego frame."""
    dx, dy = target.x_m - ego.x_m, target.y_m - ego.y_m
    distance = hypot(dx, dy)
    if distance <= 1e-9:
        raise ValueError("Coincident ego and target positions make LOS undefined")
    dvx, dvy = target.vx_mps - ego.vx_mps, target.vy_mps - ego.vy_mps
    forward = (cos(ego.heading_rad), sin(ego.heading_rad))
    left = (-forward[1], forward[0])
    longitudinal = dx * forward[0] + dy * forward[1]
    lateral = dx * left[0] + dy * left[1]
    closing = -(dvx * dx + dvy * dy) / distance
    azimuth = wrap_angle(atan2(dy, dx) - ego.heading_rad)
    return TargetObservation(slot, target, distance, closing, azimuth, longitudinal, lateral)


def select_four_targets(
    states: Sequence[VehicleState],
    ego_id: str,
    lane_count: int,
    max_range_m: float,
    preferred_adjacent_delta: int = 1,
) -> tuple[VehicleState, dict[str, TargetObservation | None], int | None]:
    """Select nearest same/adjacent-lane front and rear targets without stale-state reuse.

    If the preferred adjacent lane does not exist, the other adjacent lane is tried.
    A missing class remains None so the caller can explicitly skip that cycle.
    """
    by_id = {state.vehicle_id: state for state in states}
    if ego_id not in by_id:
        raise RuntimeError(f"Ego vehicle {ego_id!r} is absent from SUMO")
    ego = by_id[ego_id]
    deltas = [preferred_adjacent_delta, -preferred_adjacent_delta]
    adjacent_lane = next(
        (ego.lane_index + delta for delta in deltas if 0 <= ego.lane_index + delta < lane_count),
        None,
    )
    candidates: dict[str, list[TargetObservation]] = {name: [] for name in SLOT_ORDER}
    for state in states:
        if state.vehicle_id == ego_id or state.road_id != ego.road_id:
            continue
        if state.lane_index not in {ego.lane_index, adjacent_lane}:
            continue
        provisional = relative_observation("", ego, state)
        if provisional.distance_m > max_range_m:
            continue
        if abs(provisional.longitudinal_m) <= 1e-6:
            continue
        prefix = "same" if state.lane_index == ego.lane_index else "adjacent"
        suffix = "front" if provisional.longitudinal_m > 0 else "rear"
        slot = f"{prefix}_{suffix}"
        candidates[slot].append(relative_observation(slot, ego, state))
    selected: dict[str, TargetObservation | None] = {}
    for slot in SLOT_ORDER:
        pool = candidates[slot]
        selected[slot] = min(pool, key=lambda target: abs(target.longitudinal_m)) if pool else None
    return ego, selected, adjacent_lane


@dataclass(frozen=True)
class RiskResult:
    ttc_s: np.ndarray
    raw_risk: np.ndarray
    omega: np.ndarray
    all_zero_fallback: bool


def compute_ttc_weights(targets: Sequence[TargetObservation], epsilon_s: float = 0.1) -> RiskResult:
    """TTC=d/v for v>0 else infinity; raw=1/(TTC+eps); omega=K*raw/sum(raw)."""
    if epsilon_s <= 0:
        raise ValueError("TTC epsilon must be positive")
    ttc = np.full(len(targets), np.inf, dtype=float)
    for index, target in enumerate(targets):
        if target.closing_speed_mps > 0.0:
            ttc[index] = target.distance_m / target.closing_speed_mps
    raw = 1.0 / (ttc + epsilon_s)
    if raw.size and raw.sum() > 0.0:
        omega = len(raw) * raw / raw.sum()
        fallback = False
    else:
        omega = np.ones(len(raw), dtype=float)
        fallback = True
    return RiskResult(ttc, raw, omega, fallback)


@dataclass(frozen=True)
class BeamSolution:
    tx_steering: np.ndarray
    tx_beam: np.ndarray
    rx_steering: np.ndarray
    rx_beam: np.ndarray
    tx_gain_linear: float
    rx_gain_linear: float
    tx_norm_error: float
    rx_norm_error: float


def steering_vector(azimuth_rad: float, elements: int, spacing_m: float, wavelength_m: float) -> np.ndarray:
    phase = 2.0 * pi * spacing_m / wavelength_m * np.arange(elements) * sin(azimuth_rad)
    return np.exp(1j * phase)


def closed_form_beams(azimuth_rad: float, radar: RadarConfig) -> BeamSolution:
    """w*=a_t/||a_t|| and b*=a_r/||a_r||, recomputed for every target/cycle."""
    a_tx = steering_vector(azimuth_rad, radar.tx_elements, radar.element_spacing_m, radar.wavelength_m)
    a_rx = steering_vector(azimuth_rad, radar.rx_elements, radar.element_spacing_m, radar.wavelength_m)
    w_tx = a_tx / np.linalg.norm(a_tx)
    b_rx = a_rx / np.linalg.norm(a_rx)
    tx_gain = float(abs(np.vdot(a_tx, w_tx)) ** 2)
    rx_gain = float(abs(np.vdot(b_rx, a_rx)) ** 2)
    return BeamSolution(
        a_tx,
        w_tx,
        a_rx,
        b_rx,
        tx_gain,
        rx_gain,
        abs(float(np.linalg.norm(w_tx)) - 1.0),
        abs(float(np.linalg.norm(b_rx)) - 1.0),
    )


def radar_received_power_w(
    sensing_power_w: float,
    distance_m: float,
    beam: BeamSolution,
    radar: RadarConfig,
) -> float:
    """Monostatic radar equation Pr=Ps*Gt*Gr*lambda^2*sigma/((4pi)^3*d^4*L)."""
    if sensing_power_w < 0 or distance_m <= 0:
        raise ValueError("Power must be nonnegative and distance positive")
    return (
        sensing_power_w
        * beam.tx_gain_linear
        * beam.rx_gain_linear
        * radar.wavelength_m**2
        * radar.rcs_m2
        / ((4.0 * pi) ** 3 * distance_m**4 * radar.system_loss_linear)
    )


def lfm_baseband(fast_time_s: np.ndarray, radar: RadarConfig) -> np.ndarray:
    """Unit-energy rectangular LFM pulse x(u)=exp(j*pi*kappa*u^2)/sqrt(Tp)."""
    u = np.asarray(fast_time_s, dtype=float)
    pulse = radar.lfm_pulse_duration_s
    chirp_rate = radar.bandwidth_hz / pulse
    mask = np.abs(u) <= pulse / 2.0
    return mask * np.exp(1j * pi * chirp_rate * u**2) / sqrt(pulse)


def theoretical_lfm_echo(
    fast_time_s: np.ndarray,
    slow_time_s: np.ndarray,
    distance_m: float,
    radial_speed_mps: float,
    received_power_w: float,
    radar: RadarConfig,
) -> np.ndarray:
    """Noiseless LFM echo mu=sqrt(Pr)x(u-2d/c)exp(j2pi*(2vfc/c)t)."""
    delay_s = 2.0 * distance_m / C0
    doppler_hz = 2.0 * radial_speed_mps * radar.carrier_hz / C0
    fast = lfm_baseband(np.asarray(fast_time_s) - delay_s, radar)
    doppler = np.exp(1j * 2.0 * pi * doppler_hz * np.asarray(slow_time_s))
    return sqrt(received_power_w) * fast[:, None] * doppler[None, :]


@dataclass(frozen=True)
class CRBResult:
    fim: np.ndarray
    crb_d_m2: float
    crb_v_m2ps2: float
    scalar_crb: float
    received_power_w: float
    coherent_energy_snr: float


def information_coefficients_per_watt(
    target: TargetObservation,
    beam: BeamSolution,
    radar: RadarConfig,
) -> tuple[float, float, float]:
    """Return Jdd/(Ps*T), Jvv/(Ps*T^3), and Pr/Ps for the centered LFM model."""
    received_per_watt = radar_received_power_w(1.0, target.distance_m, beam, radar)
    ratio_per_watt = received_per_watt / radar.noise_psd_w_per_hz
    jdd_per_watt_second = (
        8.0
        * pi**2
        * ratio_per_watt
        * (2.0 / C0) ** 2
        * radar.rms_bandwidth_hz**2
    )
    jvv_per_watt_second3 = (
        8.0
        * pi**2
        * ratio_per_watt
        * (2.0 * radar.carrier_hz / C0) ** 2
        / 12.0
    )
    return jdd_per_watt_second, jvv_per_watt_second3, received_per_watt


def fim_and_crb(
    target: TargetObservation,
    beam: BeamSolution,
    dwell_s: float,
    sensing_power_w: float,
    radar: RadarConfig,
) -> CRBResult:
    """LFM EFIM and CRBs.

    Centered fast-frequency and centered slow-time coordinates remove the
    delay-Doppler cross moment, hence Jdv=0. The deterministic complex echo
    amplitude is projected out as a nuisance parameter. The range-dependent
    received-power coefficient is frozen at the current SUMO state.
    """
    if dwell_s <= 0 or sensing_power_w < 0:
        raise ValueError("Dwell must be positive and power nonnegative")
    jdd_rate, jvv_rate, received_per_watt = information_coefficients_per_watt(target, beam, radar)
    received = received_per_watt * sensing_power_w
    if sensing_power_w == 0.0:
        return CRBResult(np.zeros((2, 2)), inf, inf, inf, 0.0, 0.0)
    fim = np.array(
        [[jdd_rate * sensing_power_w * dwell_s, 0.0],
         [0.0, jvv_rate * sensing_power_w * dwell_s**3]],
        dtype=float,
    )
    covariance = np.linalg.inv(fim)
    crb_d = float(covariance[0, 0])
    crb_v = float(covariance[1, 1])
    scalar = crb_d / radar.crb_d_reference_m2 + crb_v / radar.crb_v_reference_m2ps2
    return CRBResult(
        fim,
        crb_d,
        crb_v,
        scalar,
        received,
        received * dwell_s / radar.noise_psd_w_per_hz,
    )


def scalar_crb_coefficients(
    targets: Sequence[TargetObservation], beams: Sequence[BeamSolution], radar: RadarConfig
) -> tuple[np.ndarray, np.ndarray]:
    """C_k=a_k/(Ps*T_k)+b_k/(Ps*T_k^3), derived from the FIM rather than a proxy."""
    range_terms, velocity_terms = [], []
    for target, beam in zip(targets, beams):
        jdd, jvv, _ = information_coefficients_per_watt(target, beam, radar)
        range_terms.append(1.0 / (jdd * radar.crb_d_reference_m2))
        velocity_terms.append(1.0 / (jvv * radar.crb_v_reference_m2ps2))
    return np.asarray(range_terms), np.asarray(velocity_terms)


@dataclass(frozen=True)
class P1Result:
    dwell_s: np.ndarray
    crb_results: tuple[CRBResult, ...]
    per_target_scalar_crb: np.ndarray
    ordinary_total_crb: float
    weighted_total_crb: float
    kkt_residual: float
    dwell_sum_error_s: float
    minimum_dwell_margin_s: float
    status: str


def _allocate_convex_dwell(
    range_terms: np.ndarray,
    velocity_terms: np.ndarray,
    weights: np.ndarray,
    total_s: float,
    minimum_s: float,
    sensing_power_w: float,
) -> tuple[np.ndarray, float]:
    """Exact KKT/dual solution of sum w(a/(PT)+b/(PT^3)) with lower bounds."""
    a = np.asarray(range_terms, dtype=float)
    b = np.asarray(velocity_terms, dtype=float)
    w = np.asarray(weights, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.shape != w.shape or not len(a):
        raise ValueError("P1 arrays must be nonempty one-dimensional arrays with matching shapes")
    if np.any(a <= 0) or np.any(b <= 0) or np.any(w < 0):
        raise ValueError("P1 coefficients must be positive and weights nonnegative")
    k = len(a)
    if k * minimum_s > total_s + 1e-12:
        raise ValueError("Infeasible P1: K*Tmin exceeds Ts")
    if abs(k * minimum_s - total_s) <= 1e-12:
        return np.full(k, total_s / k), 0.0
    if sensing_power_w == 0.0 or not np.any(w):
        return np.full(k, total_s / k), 0.0

    # x_k=T_k/Ts. The common 1/Ps factor is retained explicitly; it cancels
    # from the shares but remains in the physical objective.
    floor = minimum_s / total_s
    A = w * a / (sensing_power_w * total_s)
    B = w * b / (sensing_power_w * total_s**3)
    scale = float(np.max(A + B))
    A_scaled, B_scaled = A / scale, B / scale

    def shares(dual: float) -> np.ndarray:
        root = np.sqrt(
            (A_scaled + np.sqrt(A_scaled**2 + 12.0 * dual * B_scaled))
            / (2.0 * dual)
        )
        return np.maximum(floor, root)

    low, high = 0.0, 1.0
    while float(shares(high).sum()) > 1.0:
        high *= 2.0
        if not isfinite(high):
            raise RuntimeError("P1 dual search overflow")
    for _ in range(100):
        dual = (low + high) / 2.0
        if float(shares(dual).sum()) > 1.0:
            low = dual
        else:
            high = dual
    x = shares(high)
    x[int(np.argmax(x))] += 1.0 - float(x.sum())
    dwell = x * total_s

    marginal = w * (
        a / (sensing_power_w * dwell**2)
        + 3.0 * b / (sensing_power_w * dwell**4)
    )
    active = dwell > minimum_s + 1e-10
    dual_physical = float(np.median(marginal[active])) if np.any(active) else float(np.max(marginal))
    active_error = float(np.max(np.abs(marginal[active] - dual_physical), initial=0.0))
    bound_error = float(np.max(np.maximum(marginal[~active] - dual_physical, 0.0), initial=0.0))
    residual = max(active_error, bound_error) / max(abs(dual_physical), 1.0)
    return dwell, residual


def solve_p1(
    targets: Sequence[TargetObservation],
    beams: Sequence[BeamSolution],
    weights: Sequence[float],
    total_sensing_s: float,
    sensing_power_w: float,
    minimum_dwell_s: float,
    radar: RadarConfig,
) -> P1Result:
    """Solve P1 completely for this exact (Ts, Ps); no previous P1 result is reused."""
    k = len(targets)
    if k == 0 or len(beams) != k:
        raise ValueError("P1 requires matching nonempty targets and beams")
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (k,):
        raise ValueError("Risk-weight shape does not match target count")
    range_terms, velocity_terms = scalar_crb_coefficients(targets, beams, radar)
    dwell, residual = _allocate_convex_dwell(
        range_terms,
        velocity_terms,
        weights_array,
        total_sensing_s,
        minimum_dwell_s,
        sensing_power_w,
    )
    results = tuple(
        fim_and_crb(target, beam, float(target_dwell), sensing_power_w, radar)
        for target, beam, target_dwell in zip(targets, beams, dwell)
    )
    if sensing_power_w == 0.0:
        scalar = np.full(k, np.inf)
        ordinary = weighted = inf
        status = "zero_power_unidentifiable"
    else:
        scalar = np.asarray([result.scalar_crb for result in results])
        ordinary = float(scalar.sum())
        weighted = float(np.dot(weights_array, scalar))
        status = "optimal"
    return P1Result(
        dwell,
        results,
        scalar,
        ordinary,
        weighted,
        residual,
        abs(float(dwell.sum()) - total_sensing_s),
        float(np.min(dwell) - minimum_dwell_s),
        status,
    )


def communication_rate_bps(total_sensing_s: float, system: SystemConfig) -> float:
    communication_time = system.frame_s - total_sensing_s
    return (
        communication_time
        / system.frame_s
        * system.communication_bandwidth_hz
        * log2(1.0 + system.communication_snr_linear)
    )


def constraint_margins(
    total_sensing_s: float, sensing_power_w: float, k: int, system: SystemConfig
) -> dict[str, float]:
    """All values are >=0 exactly when every P2 constraint is satisfied."""
    return {
        "time_lower_margin_s": total_sensing_s - k * system.min_dwell_s,
        "time_upper_margin_s": system.sensing_time_qos_max_s - total_sensing_s,
        "energy_margin_j": system.sensing_energy_max_j - sensing_power_w * total_sensing_s,
        "power_lower_margin_w": sensing_power_w,
        "power_upper_margin_w": system.sensing_power_max_w - sensing_power_w,
        "qos_rate_margin_bps": communication_rate_bps(total_sensing_s, system) - system.minimum_rate_bps,
    }


@dataclass(frozen=True)
class CandidateResult:
    total_sensing_s: float
    sensing_power_w: float
    physical_objective: float
    optimizer_objective: float
    time_resource_cost: float
    energy_resource_cost: float
    feasible: bool
    margins: dict[str, float]
    p1: P1Result | None
    stage: str
    ao_round: int


class CandidateEvaluator:
    """Checks feasibility first and invokes a fresh complete P1 for every feasible candidate."""

    def __init__(
        self,
        cycle_index: int,
        targets: Sequence[TargetObservation],
        beams: Sequence[BeamSolution],
        weights: Sequence[float],
        radar: RadarConfig,
        system: SystemConfig,
    ) -> None:
        self.cycle_index = cycle_index
        self.targets = tuple(targets)
        self.beams = tuple(beams)
        self.weights = np.asarray(weights, dtype=float)
        self.radar = radar
        self.system = system
        self.p1_calls = 0
        self.infeasible_evaluations = 0
        self.records: list[dict[str, object]] = []

    def evaluate(self, total_sensing_s: float, sensing_power_w: float, stage: str, ao_round: int) -> CandidateResult:
        margins = constraint_margins(total_sensing_s, sensing_power_w, len(self.targets), self.system)
        finite = isfinite(total_sensing_s) and isfinite(sensing_power_w)
        feasible = finite and min(margins.values()) >= -self.system.feasibility_tolerance
        p1 = None
        if feasible:
            # This call is deliberately unconditional: no cache and no warm-start
            # solution is reused between different P2 candidate evaluations.
            p1 = solve_p1(
                self.targets,
                self.beams,
                self.weights,
                total_sensing_s,
                sensing_power_w,
                self.system.min_dwell_s,
                self.radar,
            )
            self.p1_calls += 1
            physical = p1.weighted_total_crb
            # Strict paper P2: the optimizer objective is the physical
            # risk-weighted CRB. Time and energy are hard constraints only.
            time_cost = 0.0
            energy_cost = 0.0
            optimizer_value = physical
            p1_called = 1
        else:
            self.infeasible_evaluations += 1
            violation = sum(max(0.0, -value) ** 2 for value in margins.values() if isfinite(value))
            physical = float("nan")
            time_cost = float("nan")
            energy_cost = float("nan")
            optimizer_value = self.system.infeasible_penalty * (1.0 + violation)
            p1_called = 0
        record: dict[str, object] = {
            "cycle_index": self.cycle_index,
            "candidate_index": len(self.records) + 1,
            "ao_round": ao_round,
            "stage": stage,
            "Ts_s": total_sensing_s,
            "Ps_w": sensing_power_w,
            "feasible": int(feasible),
            "p1_called": p1_called,
            "physical_weighted_crb": physical,
            "time_resource_cost": time_cost,
            "energy_resource_cost": energy_cost,
            "optimizer_value": optimizer_value,
        }
        record.update(margins)
        self.records.append(record)
        return CandidateResult(
            total_sensing_s,
            sensing_power_w,
            physical,
            optimizer_value,
            time_cost,
            energy_cost,
            feasible,
            margins,
            p1,
            stage,
            ao_round,
        )


@dataclass(frozen=True)
class AOResult:
    final: CandidateResult
    ao_rounds: int
    converged: bool
    last_relative_change: float
    p1_calls: int
    infeasible_evaluations: int
    power_monotonic_theory: bool
    power_monotonic_numeric: bool
    power_scaling_relative_error: float
    dwell_power_invariance_max_s: float
    ao_trace: tuple[dict[str, object], ...]
    candidate_records: tuple[dict[str, object], ...]


def _distinct(values: Sequence[float], tolerance: float = 1e-14) -> list[float]:
    result: list[float] = []
    for value in values:
        if not any(abs(value - existing) <= tolerance for existing in result):
            result.append(float(value))
    return result


def _optimize_ts(
    evaluator: CandidateEvaluator, current_ts: float, fixed_ps: float, ao_round: int
) -> CandidateResult:
    system = evaluator.system
    lower = len(evaluator.targets) * system.min_dwell_s
    qos_safe_upper = system.sensing_time_qos_max_s - 1e-12
    energy_upper = system.sensing_energy_max_j / fixed_ps if fixed_ps > 0.0 else qos_safe_upper
    upper = min(qos_safe_upper, energy_upper)
    if upper < lower - system.feasibility_tolerance:
        return evaluator.evaluate(lower, fixed_ps, "Ts_infeasible_interval", ao_round)
    # For the established CRB model, enlarging total sensing time expands the
    # P1 feasible set and decreases every target CRB.  Evaluate both endpoints
    # and the current point with fresh P1 calls as a numerical theory check.
    candidates = _distinct([lower, upper, min(max(current_ts, lower), upper)])
    evaluations = [evaluator.evaluate(value, fixed_ps, "Ts_exact_block_check", ao_round) for value in candidates]
    return min(evaluations, key=lambda item: item.optimizer_objective)


def _optimize_ps(
    evaluator: CandidateEvaluator, fixed_ts: float, current_ps: float, ao_round: int
) -> CandidateResult:
    system = evaluator.system
    lower = 0.0
    upper = min(system.sensing_power_max_w, system.sensing_energy_max_j / fixed_ts)
    if upper < 0.0:
        return evaluator.evaluate(fixed_ts, 0.0, "Ps_infeasible_interval", ao_round)
    # The FIM is linear in Ps, hence the scalar CRB is proportional to 1/Ps.
    # Endpoint evaluations retain an explicit numerical monotonicity check.
    candidates = _distinct([lower, upper, min(max(current_ps, lower), upper)])
    evaluations = [evaluator.evaluate(fixed_ts, value, "Ps_exact_block_check", ao_round) for value in candidates]
    return min(evaluations, key=lambda item: item.optimizer_objective)


def alternating_optimize_system(
    cycle_index: int,
    targets: Sequence[TargetObservation],
    beams: Sequence[BeamSolution],
    weights: Sequence[float],
    radar: RadarConfig,
    system: SystemConfig,
    initial_total_sensing_s: float | None = None,
    initial_sensing_power_w: float | None = None,
) -> AOResult:
    """P2 AO: fixed Ps optimize Ts (each candidate -> P1), then fixed Ts optimize Ps."""
    evaluator = CandidateEvaluator(cycle_index, targets, beams, weights, radar, system)
    # The previous SUMO cycle may provide only the outer AO starting point.
    # Every candidate still invokes a new P1 and no dwell allocation is reused.
    current_ts = (
        system.initial_sensing_time_s
        if initial_total_sensing_s is None
        else float(initial_total_sensing_s)
    )
    current_ps = (
        system.initial_sensing_power_w
        if initial_sensing_power_w is None
        else float(initial_sensing_power_w)
    )
    minimum_ts = len(targets) * system.min_dwell_s
    qos_safe_upper = system.sensing_time_qos_max_s - 1e-12
    current_ts = min(max(current_ts, minimum_ts), qos_safe_upper)
    current_power_cap = min(system.sensing_power_max_w, system.sensing_energy_max_j / current_ts)
    current_ps = min(max(current_ps, 1e-12), current_power_cap)
    # A previous-cycle point may lie on Ps*Ts=Es_max. If the current channel
    # permits a larger QoS-limited Ts, lower only the starting power enough to
    # make that new time bound reachable. This is causal, feasible, and avoids
    # a coordinate lock on the coupled energy boundary.
    if (
        initial_total_sensing_s is not None
        and current_ts < qos_safe_upper - system.time_search_tolerance_s
    ):
        power_at_new_time_bound = min(
            system.sensing_power_max_w,
            system.sensing_energy_max_j / qos_safe_upper,
        )
        current_ps = min(
            current_ps,
            system.ao_warm_start_power_backoff * power_at_new_time_bound,
        )
    current = evaluator.evaluate(current_ts, current_ps, "AO_initial", 0)
    if not current.feasible or not isfinite(current.physical_objective):
        raise RuntimeError(
            "Configured AO initial point must be feasible and finite: "
            f"cycle={cycle_index}, Ts={current_ts:.16g}, Ps={current_ps:.16g}, "
            f"objective={current.physical_objective}, margins={current.margins}"
        )
    previous_objective = current.optimizer_objective
    ao_trace: list[dict[str, object]] = [{
        "ao_iteration": 0,
        "stage": "initial",
        "Ts_s": current.total_sensing_s,
        "Ps_w": current.sensing_power_w,
        "weighted_crb": current.physical_objective,
    }]
    converged = False
    relative_change = inf
    rounds = 0
    for ao_round in range(1, system.ao_max_rounds + 1):
        rounds = ao_round
        ts_optimum = _optimize_ts(evaluator, current_ts, current_ps, ao_round)
        if not ts_optimum.feasible:
            raise RuntimeError("AO Ts subproblem has no feasible point")
        relaxed_ts = current_ts + system.ao_time_relaxation * (ts_optimum.total_sensing_s - current_ts)
        ts_result = evaluator.evaluate(relaxed_ts, current_ps, "Ts_relaxed_accept", ao_round)
        current_ts = ts_result.total_sensing_s
        ao_trace.append({
            "ao_iteration": ao_round,
            "stage": "Ts_update",
            "Ts_s": ts_result.total_sensing_s,
            "Ps_w": ts_result.sensing_power_w,
            "weighted_crb": ts_result.physical_objective,
        })
        ps_optimum = _optimize_ps(evaluator, current_ts, current_ps, ao_round)
        if not ps_optimum.feasible or not isfinite(ps_optimum.physical_objective):
            raise RuntimeError("AO Ps subproblem failed to find a positive-power feasible point")
        time_bound_gap = (system.sensing_time_qos_max_s - 1e-12) - current_ts
        power_relaxation = (
            system.ao_power_relaxation_near_time_bound
            if time_bound_gap <= system.time_search_tolerance_s
            else system.ao_power_relaxation
        )
        relaxed_ps = current_ps + power_relaxation * (ps_optimum.sensing_power_w - current_ps)
        ps_result = evaluator.evaluate(current_ts, relaxed_ps, "Ps_relaxed_accept", ao_round)
        current_ps = ps_result.sensing_power_w
        ao_trace.append({
            "ao_iteration": ao_round,
            "stage": "Ps_update",
            "Ts_s": ps_result.total_sensing_s,
            "Ps_w": ps_result.sensing_power_w,
            "weighted_crb": ps_result.physical_objective,
        })
        current = evaluator.evaluate(current_ts, current_ps, "AO_round_end", ao_round)
        relative_change = abs(previous_objective - current.optimizer_objective) / max(abs(previous_objective), 1e-30)
        if relative_change < system.ao_eta:
            converged = True
            break
        previous_objective = current.optimizer_objective

    # A fresh final P1 evaluation guarantees final logs correspond exactly to
    # the reported (Ts*, Ps*) rather than any earlier subproblem evaluation.
    final = evaluator.evaluate(current_ts, current_ps, "AO_final", rounds)

    # Current FIM is linear in Ps, so C*(Ts,Ps) is theoretically proportional
    # to 1/Ps and optimal dwell shares are power-invariant. Verify this at every
    # cycle without replacing the requested numerical AO power subproblem.
    feasible_cap = min(system.sensing_power_max_w, system.sensing_energy_max_j / current_ts)
    low_power = 0.5 * feasible_cap
    high_power = feasible_cap
    low = evaluator.evaluate(current_ts, low_power, "power_monotonicity_check_low", rounds)
    high = evaluator.evaluate(current_ts, high_power, "power_monotonicity_check_high", rounds)
    expected_ratio = high_power / low_power
    actual_ratio = low.physical_objective / high.physical_objective
    scaling_error = abs(actual_ratio - expected_ratio) / expected_ratio
    dwell_error = float(np.max(np.abs(low.p1.dwell_s - high.p1.dwell_s)))
    numeric_monotonic = (
        high.physical_objective < low.physical_objective
        and scaling_error < 1e-8
        and dwell_error < 1e-10
    )
    return AOResult(
        final,
        rounds,
        converged,
        relative_change,
        evaluator.p1_calls,
        evaluator.infeasible_evaluations,
        True,
        bool(numeric_monotonic),
        float(scaling_error),
        dwell_error,
        tuple(ao_trace),
        tuple(evaluator.records),
    )


def validate_lfm_fim_numerically(radar: RadarConfig) -> dict[str, float | bool]:
    """Finite-difference the theoretical centered LFM echo and compare with analytic FIM."""
    distance, speed, dwell, received = 55.0, 7.0, 0.012, 2e-15
    frequency = np.linspace(-1.0, 1.0, 21)
    frequency *= radar.rms_bandwidth_hz / sqrt(float(np.mean(frequency**2)))
    slow_time = np.linspace(-1.0, 1.0, 23)
    slow_time *= (dwell / sqrt(12.0)) / sqrt(float(np.mean(slow_time**2)))
    ff, tt = np.meshgrid(frequency, slow_time)
    ff, tt = ff.ravel(), tt.ravel()
    energy = received * dwell

    def echo(candidate_distance: float, candidate_speed: float) -> np.ndarray:
        phase = 2.0 * pi * (
            -ff * (2.0 * candidate_distance / C0)
            + tt * (2.0 * radar.carrier_hz * candidate_speed / C0)
        )
        return sqrt(energy / len(ff)) * np.exp(1j * phase)

    mu = echo(distance, speed)
    d_step, v_step = 1e-4, 1e-5
    derivative = np.column_stack(
        ((echo(distance + d_step, speed) - echo(distance - d_step, speed)) / (2.0 * d_step),
         (echo(distance, speed + v_step) - echo(distance, speed - v_step)) / (2.0 * v_step))
    )
    # Eliminate the unknown deterministic complex reflection amplitude.
    projected = derivative - mu[:, None] * ((mu.conj() @ derivative) / np.vdot(mu, mu))[None, :]
    numerical = 2.0 / radar.noise_psd_w_per_hz * np.real(projected.conj().T @ projected)
    ratio = received / radar.noise_psd_w_per_hz
    analytical = np.diag(
        [8.0 * pi**2 * ratio * (2.0 / C0) ** 2 * radar.rms_bandwidth_hz**2 * dwell,
         8.0 * pi**2 * ratio * (2.0 * radar.carrier_hz / C0) ** 2 * dwell**3 / 12.0]
    )
    relative = float(np.linalg.norm(numerical - analytical) / np.linalg.norm(analytical))
    return {
        "passed": bool(relative < 2e-5),
        "relative_frobenius_error": relative,
        "numerical_cross_term": float(numerical[0, 1]),
        "analytical_cross_term": 0.0,
    }


def dbm(power_w: float) -> float:
    return -inf if power_w <= 0.0 else 10.0 * np.log10(power_w * 1000.0)


def complex_vector_text(vector: np.ndarray) -> str:
    return ";".join(f"{value.real:+.8f}{value.imag:+.8f}j" for value in vector)
