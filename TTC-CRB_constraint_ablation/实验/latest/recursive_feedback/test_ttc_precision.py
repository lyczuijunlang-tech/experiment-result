import unittest

import numpy as np

from run_feedback import allocate, information, model, posterior
from ttc_precision import TTCInfeasible, TTCPrecision


class TTCPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.states = np.array([[30., 1.], [45., -0.2], [50., 0.], [40., 0.7]])
        self.prior = np.broadcast_to(np.linalg.inv([[0.08, -0.012], [-0.012, 0.02]]), (4, 2, 2))
        self.coeff = np.tile([8e6, 3e9], (4, 1))
        self.power = .025
        self.dwell = np.full(4, .010)

    def covariance(self, dwell):
        return posterior(self.prior, information(dwell, self.power, self.coeff))

    def test_formula_and_derivative(self):
        precision = TTCPrecision(self.states, .01)
        cov = self.covariance(self.dwell)
        gamma = precision.gamma(cov)
        for k in (0, 3):
            d, v = self.states[k]
            g = np.array([1 / v, -d / v**2])
            np.testing.assert_allclose(gamma[k], (g @ cov[k] @ g) / (d / v)**2, rtol=1e-13)
            c = cov[k]
            np.testing.assert_allclose(gamma[k], c[0, 0]/d**2 - 2*c[0, 1]/(d*v) + c[1, 1]/v**2)
        direction = np.array([1., 0., 0., -0.4])
        step = 1e-7
        numeric = (precision.gamma(self.covariance(self.dwell + step*direction)) -
                   precision.gamma(self.covariance(self.dwell - step*direction))) / (2*step)
        for k in (0, 3):
            d, v = self.states[k]
            projected = cov[k] @ np.array([1/d, -1/v])
            j_gradient = self.power * self.coeff[k] * [1, 3*self.dwell[k]**2]
            np.testing.assert_allclose(numeric[k], -(projected**2 @ j_gradient)*direction[k], rtol=1e-7)

    def test_monotonicity_and_nonclosing(self):
        precision = TTCPrecision(self.states, .01)
        result = np.array([precision.gamma(self.covariance(np.full(4, x))) for x in np.linspace(.002, .03, 30)])
        self.assertTrue(np.isnan(result[:, 1:3]).all())
        self.assertTrue((np.diff(result[:, [0, 3]], axis=0) < 0).all())

    def test_boundary_and_individual_infeasibility(self):
        boundary = TTCPrecision(self.states, 1.).gamma(self.covariance(self.dwell))[0]
        precision = TTCPrecision(self.states, boundary)
        floors = precision.required_dwell(.05, self.power, .002, self.coeff, self.prior)
        self.assertAlmostEqual(floors[0], .010, places=12)
        self.assertEqual(floors[1], .002)
        self.assertTrue((precision.gamma(self.covariance(floors))[precision.active] <= boundary*(1+1e-10)).all())
        with self.assertRaisesRegex(TTCInfeasible, "individual"):
            precision.required_dwell(.012, self.power, .002, self.coeff, self.prior)

    def test_aggregate_infeasibility_and_p2(self):
        same = np.tile([30., 1.], (4, 1))
        boundary = TTCPrecision(same, 1.).gamma(self.covariance(self.dwell))[0]
        with self.assertRaisesRegex(TTCInfeasible, "sum_required"):
            TTCPrecision(same, boundary).required_dwell(.039, self.power, .002, self.coeff, self.prior)
        precision = TTCPrecision(self.states, boundary)
        steps = allocate(model.SystemConfig(), self.coeff, self.prior, np.ones(4), np.ones(2), True, None,
                         precision=precision)
        for step in steps:
            self.assertAlmostEqual(step["dwell"].sum(), step["time"])
            self.assertTrue((precision.gamma(step["post"])[precision.active] <= boundary*(1+1e-9)).all())
        self.assertEqual(steps[-1]["stage"], "power")

    def test_near_zero_speed_is_not_clipped(self):
        state = self.states.copy()
        state[0, 1] = 1e-12
        precision = TTCPrecision(state, .01)
        self.assertTrue(precision.active[0])
        self.assertGreater(precision.gamma(self.covariance(self.dwell))[0], 1e18)
        with self.assertRaisesRegex(TTCInfeasible, "individual.*k1"):
            precision.required_dwell(.05, self.power, .002, self.coeff, self.prior)


if __name__ == "__main__":
    unittest.main()
