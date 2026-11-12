"""Per-target posterior TTC precision on causal range and closing-speed predictions."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


class TTCInfeasible(ValueError):
    def __init__(self, reason, required):
        super().__init__(reason)
        self.required = required


@dataclass(frozen=True)
class TTCPrecision:
    state: np.ndarray
    gamma_limit: float

    @property
    def active(self):
        return self.state[:, 1] > 0

    def gamma(self, covariance):
        result = np.full(len(self.state), np.nan)
        active = self.active
        d, v = self.state[active].T
        # Work with v * [1/d, -1/v] to avoid inverse powers of v.
        directions = np.column_stack((v / d, -np.ones(len(v))))
        scaled_variance = np.einsum("ki,kij,kj->k", directions, covariance[active], directions)
        with np.errstate(divide="ignore", over="ignore"):
            result[active] = scaled_variance / v**2
        return result

    def required_dwell(self, total, power, minimum, coefficients, prior_info):
        """Reduce each hard inequality to a lower bound on its own dwell."""
        required = np.full(len(self.state), minimum)
        upper = total - (len(required) - 1) * minimum
        for k in np.flatnonzero(self.active):
            d, v = self.state[k]
            scaled_direction = v / d
            scaled_limit = self.gamma_limit * v**2
            a, b = prior_info[k, 0]
            c = prior_info[k, 1, 1]
            j_d, j_v = power * coefficients[k]

            def residual(dwell):
                dd = a + j_d * dwell
                vv = c + j_v * dwell**3
                # u^T (P^-^{-1}+J)^{-1} u for u = [v/d, -1].
                return (vv * scaled_direction**2 + 2 * b * scaled_direction + dd) / (
                    dd * vv - b**2) - scaled_limit

            if residual(minimum) <= 0:
                continue
            if upper <= minimum or residual(upper) > 0:
                required[k] = np.inf
                continue
            root = brentq(residual, minimum, upper, xtol=1e-15)
            required[k] = min(root + 2e-15, upper)

        impossible = np.flatnonzero(np.isinf(required))
        if impossible.size:
            targets = ",".join(f"k{k + 1}" for k in impossible)
            raise TTCInfeasible(f"individual_dwell_exceeds_budget:{targets}", required)
        if required.sum() > total:
            raise TTCInfeasible("sum_required_dwell_exceeds_budget", required)
        return required
