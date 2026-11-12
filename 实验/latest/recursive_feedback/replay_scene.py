"""Target IDs, target geometry and the communication model for the recorded SUMO replay."""

import numpy as np

from model import SLOT_ORDER, TargetObservation, VehicleState, relative_observation


FIXED_IDS = {
    "same_front": "target1_front_same",
    "same_rear": "target2_rear_same",
    "adjacent_front": "target3_front_adj",
    "adjacent_rear": "target4_rear_adj",
}

EXPECTED_LANES = {"same_front": 0, "same_rear": 0, "adjacent_front": 1, "adjacent_rear": 1}

RSU_POSITION_M = (1050.0, 80.0)

COMMUNICATION_SNR_FLOOR_DB = 10.3

RSU_ASSIST_MAX_DB = 0.12

RSU_ASSIST_SCALE_M = 500.0

SHADOWING_LONG_DISTANCE_M = 1000.0

SHADOWING_SHORT_DISTANCE_M = 520.0

SHADOWING_LONG_AMPLITUDE_DB = 0.10

SHADOWING_SHORT_AMPLITUDE_DB = 0.035

def fixed_target_observations(states: list[VehicleState], ego_id: str) -> tuple[VehicleState, list[TargetObservation], list[str]]:
    by_id = {state.vehicle_id: state for state in states}
    if ego_id not in by_id:
        raise RuntimeError("ego is absent from SUMO")
    ego = by_id[ego_id]
    missing: list[str] = []
    observations: list[TargetObservation] = []
    for slot in SLOT_ORDER:
        vehicle_id = FIXED_IDS[slot]
        target = by_id.get(vehicle_id)
        if target is None or target.road_id != ego.road_id or target.lane_index != EXPECTED_LANES[slot]:
            missing.append(slot)
            continue
        observations.append(relative_observation(slot, ego, target))
    return ego, observations, missing

def communication_snr_from_rsu(ego: VehicleState) -> tuple[float, float]:
    """Interference-limited cellular SNR with smooth RSU assistance.

    The floor represents macro-cell coverage. A weak Gaussian-in-distance RSU
    gain and two long-period low-amplitude spatial shadowing components make
    the QoS boundary rise and fall gently over the SUMO route.  The resource
    curves therefore vary at every cycle without random noise or post-processing.
    """
    distance = float(np.hypot(ego.x_m - RSU_POSITION_M[0], ego.y_m - RSU_POSITION_M[1]))
    rsu_gain_db = RSU_ASSIST_MAX_DB * np.exp(-(distance / RSU_ASSIST_SCALE_M) ** 2)
    spatial_shadowing_db = (
        SHADOWING_LONG_AMPLITUDE_DB
        * np.sin(2.0 * np.pi * ego.x_m / SHADOWING_LONG_DISTANCE_M)
        + SHADOWING_SHORT_AMPLITUDE_DB
        * np.sin(2.0 * np.pi * ego.x_m / SHADOWING_SHORT_DISTANCE_M + 0.8)
    )
    snr_db = COMMUNICATION_SNR_FLOOR_DB + rsu_gain_db + spatial_shadowing_db
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(snr_linear), distance
