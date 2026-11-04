import unittest
from measure_fit.scoring import FitScore
class ScoringTests(unittest.TestCase):
 def tick(self,s,**kwargs):
  values=dict(shaft_mm=0.,probe_mm=-6.,contact_valid=True,reference_valid=True,installed=False,clearance_mm=12.,angle_rad=0.,released=False,stable=True)
  values.update(kwargs);return s.update(**values)
 def test_nearby_probe_or_unreferenced_gauge_cannot_earn_credit(self):
  for flag in ('contact_valid','reference_valid'):
   s=FitScore()
   for x in (0.,12.):
    for _ in range(8): self.tick(s,shaft_mm=x,probe_mm=x-6,**{flag:False})
   self.assertEqual(s.score,0)
 def test_one_endpoint_is_not_a_measurement(self):
  s=FitScore()
  for _ in range(30): self.tick(s)
  self.assertEqual(s.score,0)
 def test_initial_measurement_and_wrong_spacer(self):
  s=FitScore()
  for x in (0.,12.):
   for _ in range(8): self.tick(s,shaft_mm=x,probe_mm=x-6)
  self.assertEqual(s.score,.25)
  self.tick(s,installed=True,clearance_mm=6.)
  self.assertEqual(s.score,.25)
 def test_nan_does_not_preserve_success(self):
  s=FitScore();s.success=True
  self.assertEqual(self.tick(s,shaft_mm=float('nan')),0)
  self.assertFalse(s.success)
 def test_full_score_requires_post_measurement_rotation_and_release(self):
  s=FitScore()
  for x in (0.,12.):
   for _ in range(8): self.tick(s,shaft_mm=x,probe_mm=x-6)
  for x in (0.,3.):
   for _ in range(8): self.tick(s,shaft_mm=x,probe_mm=x-6,installed=True,clearance_mm=3.)
  self.assertEqual(s.score,.75)
  for _ in range(50): self.tick(s,installed=True,clearance_mm=3.,angle_rad=-1.2,released=False)
  self.assertEqual(s.score,.75)
  for _ in range(50): self.tick(s,installed=True,clearance_mm=3.,angle_rad=-1.2,released=True,stable=False)
  self.assertEqual(s.score,.75)
  for _ in range(41): self.tick(s,installed=True,clearance_mm=3.,angle_rad=-1.2,released=True)
  self.assertEqual(s.score,1.)
  self.tick(s,installed=False)
  self.assertEqual(s.score,.25)
  self.assertFalse(s.post_measured)
