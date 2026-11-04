import unittest
from measure_fit.readout import display_digits
class ReadoutTests(unittest.TestCase):
 def test_offset_endpoints_and_difference(self):
  self.assertEqual(display_digits(-20),(0,0,0))
  self.assertEqual(display_digits(20),(4,0,0))
  self.assertEqual(display_digits(-6),(1,4,0))
  self.assertEqual(display_digits(6),(2,6,0))
 def test_invalid_is_blank(self):
  self.assertIsNone(display_digits(float('nan')))
