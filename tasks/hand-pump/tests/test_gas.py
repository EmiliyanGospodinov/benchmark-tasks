"""Checks that pressure comes from displacement and physically directed flow."""
import math
import unittest
from hand_pump.gas import PumpGas


class GasChecks(unittest.TestCase):
    def pump(self):
        return PumpGas(math.pi * .007**2, .8e-6, .08, 100e-6)

    def test_stationary_handle_does_not_create_pressure(self):
        p = self.pump()
        for _ in range(1000):
            p.update(.08)
        self.assertAlmostEqual(p.gauge_kpa, 0.0)

    def test_compression_conserves_gas_and_has_expected_pressure(self):
        p = self.pump()
        q0 = p.cylinder_q + p.receiver_q
        p.update(0)
        expected = p.atmospheric * p.area * p.stroke / (p.receiver_volume + p.dead_volume)
        self.assertAlmostEqual(p.gauge_kpa * 1000, expected)
        self.assertAlmostEqual(p.cylinder_q + p.receiver_q, q0)
        self.assertGreater(p.outward_force, 0)

    def test_return_stroke_keeps_receiver_air(self):
        p = self.pump(); p.update(0); q = p.receiver_q
        p.update(.08)
        self.assertAlmostEqual(p.receiver_q, q)
        self.assertAlmostEqual(p.cylinder_pressure, p.atmospheric)
        self.assertAlmostEqual(p.outward_force, 0)

    def test_same_stroke_resampling_preserves_result(self):
        a, b = self.pump(), self.pump()
        a.update(0)
        for i in range(1, 101):
            b.update(.08 * (1 - i/100))
        self.assertAlmostEqual(a.gauge_kpa, b.gauge_kpa, places=8)

    def test_reset_discards_previous_episode(self):
        p = self.pump(); p.update(0); p.reset()
        self.assertAlmostEqual(p.gauge_kpa, 0)
        self.assertAlmostEqual(p.delivered_q, 0)

    def test_invalid_geometry_does_not_pump(self):
        p = self.pump()
        for x in [float('nan'), float('inf'), -.03, .2]:
            with self.assertRaises(ValueError): p.update(x)


if __name__ == '__main__':
    unittest.main()
