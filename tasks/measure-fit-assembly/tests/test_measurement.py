import unittest
import numpy as np
from measure_fit.measurement import read_indicator

class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.g=np.array([0,0,0,0,0,0,1.],float)
        self.p=self.g.copy(); self.p[0]=.05
        self.s=self.g.copy(); self.s[0]=.146
    def test_contact_and_displacement(self):
        self.assertTrue(read_indicator(self.g,self.p,self.s)['contact_geometry_valid'])
        self.p[0]+=.012; self.s[0]+=.012
        self.assertAlmostEqual(read_indicator(self.g,self.p,self.s)['extension_mm'],12.)
    def test_whole_instrument_translation_is_not_measurement(self):
        first=read_indicator(self.g,self.p,self.s)
        for p in (self.g,self.p,self.s): p[:3]+=[.3,-.4,1.]
        last=read_indicator(self.g,self.p,self.s)
        self.assertAlmostEqual(first['extension_mm'],last['extension_mm'])
        self.assertTrue(last['contact_geometry_valid'])
    def test_nearby_without_contact_rejected(self):
        self.s[0]+=.005
        self.assertFalse(read_indicator(self.g,self.p,self.s)['contact_geometry_valid'])
        self.s[0]-=.005; self.s[1]+=.01
        self.assertFalse(read_indicator(self.g,self.p,self.s)['contact_geometry_valid'])
    def test_nonfinite_rejected(self):
        self.p[0]=np.nan
        with self.assertRaises(ValueError): read_indicator(self.g,self.p,self.s)
