import unittest
from measure_fit.design import Fit
class FitTests(unittest.TestCase):
    def test_nominal_choice(self): self.assertAlmostEqual(Fit().choose(.012),.009)
    def test_choice_has_measurement_margin(self):
        # Held-indicator readings in earlier runs deviated from true travel by under 0.04 mm.
        for x in (.01192,.01196,.01204,.01208): self.assertAlmostEqual(Fit().choose(x),.009)
    def test_changed_gap_changes_choice(self):
        self.assertAlmostEqual(Fit().choose(.0118),.0088)
        self.assertAlmostEqual(Fit().choose(.0122),.0092)
    def test_adjacent_spacers_fall_outside_window(self):
        f=Fit()
        self.assertGreater(f.residual(.0088),f.target_max+.00009)
        self.assertLess(f.residual(.0092),f.target_min-.00009)
    def test_nonunique_or_impossible_measurement(self):
        for x in [float('nan'),float('inf'),.001,.03,.0125]:
            with self.assertRaises(ValueError): Fit().choose(x)
if __name__=='__main__':unittest.main()
