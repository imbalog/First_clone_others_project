from unittest import TestCase
from EuropeanOptionClass import BlackScholes

__author__ = 'JPC'


class TestBlackScholes(TestCase):
	def test_value(self):
		BlackScholes_test_1 = BlackScholes('call', 0.0001, 100, 1, 0., 0., .35)
		self.assertAlmostEqual(BlackScholes_test_1.value, 0.)
		BlackScholes_test_2 = BlackScholes('call', 100, 0.0001, 1, 0., 0., .35)
		self.assertAlmostEqual(round(BlackScholes_test_2.value, 3), 100)
		BlackScholes_test_3 = BlackScholes('call', 100, 100, 1, 0., 0., 99.)
		self.assertAlmostEqual(round(BlackScholes_test_3.value, 3), 100)
		BlackScholes_test_4 = BlackScholes('call', 100, 100, 1, 0., 0., .0000001)
		self.assertAlmostEqual(round(BlackScholes_test_4.value, 3), 0)

	def test_delta(self):
		BlackScholes_test_5 = BlackScholes('call', 0.0001, 100, 1, 0., 0., .35)
		self.assertAlmostEqual(BlackScholes_test_5.delta, 0.)
		BlackScholes_test_6 = BlackScholes('call', 100, .0001, 1, 0., 0., .35)
		self.assertAlmostEqual(BlackScholes_test_6.delta, 1.)
		BlackScholes_test_7 = BlackScholes('call', 100, 100, 1, 0., 0., 99.)
		self.assertAlmostEqual(BlackScholes_test_7.delta, 1.)
		BlackScholes_test_8 = BlackScholes('call', 100, 100, 1, 0., 0., .0000001)
		self.assertAlmostEqual(BlackScholes_test_8.delta, 0.5)
