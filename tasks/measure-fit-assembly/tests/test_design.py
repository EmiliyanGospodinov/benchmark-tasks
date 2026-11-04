import unittest
from measure_fit.design import Fit
class FitTests(unittest.TestCase):
    def test_nominal_choice(self): self.assertAlmostEqual(Fit().choose(.012),.009)
    def test_changed_gap_changes_choice(self): self.assertAlmostEqual(Fit(initial_play=.009).choose(.009),.006)
    def test_thin_excess_and_thick_interference(self):
        f=Fit();self.assertGreater(f.residual(.006),f.target_max);self.assertLess(f.residual(.014),0)
    def test_nonunique_or_impossible_measurement(self):
        for x in [float('nan'),float('inf'),.001,.03]:
            with self.assertRaises(ValueError): Fit().choose(x)
if __name__=='__main__':unittest.main()
