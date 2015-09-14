import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod


class EuropeanOption(object):
	""" Abstract Class for European options. Partially implemented.
	s0 : float : initial stock/index level
	strike : float : strike price
	maturity : float : time to maturity (in year fractions)
	int_rates : float : constant risk-free short rate
	dividend_rates :    float : dividend yield
	sigma :  float : volatility factor in diffusion term"""

	__metaclass__ = ABCMeta

	def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, model):
		try:
			self.option_type = option_type
			assert isinstance(option_type, str)
			self.S0 = float(s0)
			self.strike = float(strike)
			self.T = float(maturity)
			self.r = float(int_rates)
			self.div = float(dividend_rates)
			self.sigma = float(sigma)
			self.model = str(model)
		except ValueError:
			print('Error passing Options parameters')

		models = ['BlackScholes', 'MonteCarlo', 'BinomialTree']
		if model not in models:
			raise Exception('Error: Model unknown')
		option_types = ['call', 'put']
		if option_type not in option_types:
			raise ValueError("Error: Option type not valid. Enter 'call' or 'put'")
		if s0 < 0 or strike < 0 or maturity <= 0 or int_rates < 0 or dividend_rates < 0 or sigma < 0:
			raise ValueError('Error: Negative inputs not allowed')

	def getmodel(self):
		return self.model

	def __str__(self):
		return "This EuropeanOption is priced using {0}".format(self.model)

	@abstractmethod
	def value(self):
		pass


class BlackScholes(EuropeanOption):

	def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma):
		EuropeanOption.__init__(self,option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, 'BlackScholes')

	@property
	def value(self):
		d1 = (np.log(self.S0 / self.strike) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
			self.sigma * np.sqrt(self.T))
		d2 = (d1 - self.sigma * np.sqrt(self.T))
		normal_d1 = norm.cdf(d1, 0, 1)
		normal_neg_d1 = norm.cdf(-d1, 0, 1)
		normal_d2 = norm.cdf(d2, 0, 1)
		normal_neg_d2 = norm.cdf(-d2, 0, 1)
		if self.option_type == 'call':
			value = (self.S0 * np.exp(-self.div * self.T) * normal_d1 -
					 self.strike * np.exp(-self.r * self.T) * normal_d2)
		else:
			value = (self.strike * np.exp(-self.r * self.T) * normal_neg_d2 -
					 self.S0 * np.exp(-self.div * self.T) * normal_neg_d1)
		return value


class MonteCarlo(EuropeanOption):

	def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, simulations = 100000):
		EuropeanOption.__init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, "MonteCarlo")
		self.simulations = int(simulations)
		try:
			if self.simulations > 0 :
				assert isinstance(self.simulations, int)
		except:
			raise ValueError("Simulation's number has to be positive integer")

	def generate_payoffs(self, seed = 1234567890):
		np.random.seed(seed)
		brownian = np.random.normal(size = self.simulations)
		price_terminal = self.S0 * np.exp((self.r - self.div - 0.5 * self.sigma ** 2) * self.T +
										  self.sigma * np.sqrt(self.T) * brownian)
		if self.option_type == 'call':
			payoff = np.maximum((price_terminal - self.strike), np.zeros(self.simulations))
		else:
			payoff = np.maximum((self.strike - price_terminal), np.zeros(self.simulations))
		return payoff

	@property
	def value(self):
		payoff = self.generate_payoffs()
		return np.exp(-1.0 * self.r * self.T) * np.sum(payoff) / float(self.simulations)


class BinomialTree(EuropeanOption):

	def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, time_grid ):
		EuropeanOption.__init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma, "BinomialTree")
		self.time_grid = int(time_grid)

	@property
	def value(self):
		delta_time = self.T / self.time_grid
		discount = np.exp(-1.0 *self.r * delta_time)
		move_up = np.exp(self.sigma * np.sqrt(delta_time))
		move_down = 1 / float(move_up)
		probability = (np.exp(self.r * delta_time) - move_down) / float(move_up - move_down)
		index_up = np.arange(self.time_grid + 1)
		index_up = np.resize(index_up, (self.time_grid + 1, self.time_grid + 1))
		index_down = np.transpose(index_up)
		index_up = move_up ** (index_up - index_down)
		index_down = move_down ** index_down
		S = self.S0 * index_up * index_down
		if self.option_type == 'call':
			value = np.maximum(S - self.strike, 0)
		else:
			value = np.maximum(self.strike - S, 0)
		i = 0
		for t in range(self.time_grid - 1, -1, -1):
			value[0:self.time_grid - i, t] = ( probability * value[0:self.time_grid - i, t + 1] +
									 (1 - probability) * value[1:self.time_grid - i + 1, t + 1] ) * discount
			i += 1
		return value[0, 0]

