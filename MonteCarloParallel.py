import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
import multiprocessing
from numpy import ceil, mean
import time


class EuropeanOption(object):
	""" Abstract Class for European options. Partially implemented.
	S0 : float : initial stock/index level
	strike : float : strike price
	T : float : time to maturity (in year fractions)
	r : float : constant risk-free short rate
	div :    float : dividend yield
	sigma :  float : volatility factor in diffusion term
	model: str: name of the model for the pricing"""

	__metaclass__ = ABCMeta

	def __init__(self, option_type, S0, strike, T, r, div, sigma, model):
		try:
			self.option_type = option_type
			assert isinstance(option_type, str)
			self.S0 = float(S0)
			self.strike = float(strike)
			self.T = float(T)
			self.r = float(r)
			self.div = float(div)
			self.sigma = float(sigma)
			self.model = str(model)
		except ValueError:
			print('Error passing Options parameters')

		models = ['BlackScholes', 'MonteCarlo',
				  'BinomialTree', 'TrinomialTree',
				  'FFT', 'PDE']

		if model not in models:
			raise Exception('Error: Model unknown')

		option_types = ['call', 'put']

		if option_type not in option_types:
			raise ValueError("Error: Option type not valid. Enter 'call' or 'put'")
		if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
			raise ValueError('Error: Negative inputs not allowed')

		self.discount = np.exp(-self.r * self.T)

	def getmodel(self):
		return self.model

	def __str__(self):
		return "This European Option is priced using {0}".format(self.getmodel())

	@abstractmethod
	def value(self):
		pass

	@abstractmethod
	def delta(self):
		pass


class BlackScholes(EuropeanOption):
	def __init__(self, option_type, S0, strike, T, r, div, sigma):
		EuropeanOption.__init__(self, option_type, S0, strike,
								T, r, div, sigma, 'BlackScholes')

		d1 = ((np.log(self.S0 / self.strike) +
			   (self.r - self.div + 0.5 * (self.sigma ** 2)) * self.T) /
			  float(self.sigma * np.sqrt(self.T)))
		d2 = float(d1 - self.sigma * np.sqrt(self.T))
		self.Nd1 = norm.cdf(d1, 0, 1)
		self.Nnd1 = norm.cdf(-d1, 0, 1)
		self.Nd2 = norm.cdf(d2, 0, 1)
		self.Nnd2 = norm.cdf(-d2, 0, 1)
		self.pNd1 = norm.pdf(d1, 0, 1)

	@property
	def value(self):
		if self.option_type == 'call':
			value = (self.S0 * np.exp(-self.div * self.T) * self.Nd1 -
					 self.strike * np.exp(-self.r * self.T) * self.Nd2)
		else:
			value = (self.strike * np.exp(-self.r * self.T) * self.Nnd2 -
					 self.S0 * np.exp(-self.div * self.T) * self.Nnd1)
		return value

	@property
	def delta(self):
		if self.option_type == 'call':
			delta = np.exp(-self.div * self.T) * self.Nd1
		else:
			delta = np.exp(-self.div * self.T) * (self.Nd1 - 1)
		return delta


class MonteCarlo(EuropeanOption):
	def __init__(self, simulations, option_type, S0, strike, T, r, div, sigma,
				 antithetic=True,
				 moment_matching=True,
				 fixed_seed=True):
		EuropeanOption.__init__(self, option_type, S0, strike, T, r, div, sigma, "MonteCarlo")
		try:
			self.antithetic = bool(antithetic)
			self.moment_matching = bool(moment_matching)
			self.fixed_seed = bool(fixed_seed)
			if self.simulations > 0:
				self.simulations = int(simulations)
			else:
				raise ValueError("Simulation's number has to be positive integer")
		except:
			raise ValueError("Problem assigning types in MC arguments")

	def simulation_terminal(self, seed=1234567890):
		if self.fixed_seed:
			assert isinstance(seed, int)
			np.random.seed(seed)
		if self.antithetic:
			brownian = np.random.standard_normal(size=int(np.ceil(self.simulations / 2.)))
			brownian = np.concatenate((brownian, -brownian))
		else:
			brownian = np.random.standard_normal(size=self.simulations)
		if self.moment_matching:
			brownian = brownian - np.mean(brownian)
			brownian = brownian / np.std(brownian)

		price_terminal = self.S0 * np.exp((self.r - self.div - 0.5 * self.sigma ** 2) *
										  self.T +
										  self.sigma * np.sqrt(self.T) * brownian)
		return price_terminal

	def generate_payoffs(self):
		price_terminal = self.simulation_terminal()
		if self.option_type == 'call':
			payoff = np.maximum((price_terminal - self.strike), 0)
		else:
			payoff = np.maximum((self.strike - price_terminal), 0)
		return payoff

	@property
	def value(self):
		payoff = self.generate_payoffs()
		return self.discount * np.sum(payoff) / float(len(payoff))

	@property
	def delta(self):
		value_terminal = np.array(self.simulation_terminal() / float(self.S0))
		payoff = self.generate_payoffs()
		delta = np.zeros(len(payoff))
		delta[np.nonzero(payoff)] = value_terminal[np.nonzero(payoff)]
		return self.discount * np.sum(delta) / float(len(payoff))


def eval_in_pool(simulations, method='value'):
	try:
		assert isinstance(simulations, int)
		if method is not None:
			assert isinstance(method, str)
	except ValueError:
		print 'Type in the parameters incorrect'
	myCall = MonteCarlo(simulations, 'call', 100., 100., .5, 0.01, 0., .35)
	if method == 'value':
		return myCall.value
	if method == 'delta':
		return myCall.delta


def value_in_pool(simulations):
	if isinstance(simulations, int):
		simulations = [simulations]
	arguments = ['call', 100., 100., .5, 0.01, 0., .35]
	reduce(lambda x, y: x.extend(y), [simulations, arguments])
	return MonteCarlo(*simulations).value


def delta_in_pool(simulations):
	if isinstance(simulations, int):
		simulations = [simulations]
	arguments = ['call', 100., 100., .5, 0.01, 0., .35]
	reduce(lambda x, y: x.extend(y), [simulations, arguments])
	return MonteCarlo(*simulations).delta


if __name__ == '__main__':
	call = BlackScholes('call', 100., 100., .5, 0.01, 0., .35)
	print '-' * 85
	print 'BS Price:', call.value, 'BS Delta:', call.delta
	print '-' * 85
	scenarios = {'1': [1e4, 1e7],
				 '2': [1e4, 1e7],
				 '3': [1e4, 1e7],
				 '4': [1e4, 1e7],
				 '5': [1e4, 1e7],
				 '6': [1e4, 1e7]}
	for num_processes in scenarios:
		for N in scenarios[num_processes]:
			start = time.time()
			chunks = [int(ceil(N / int(num_processes)))] * int(num_processes)
			chunks[-1] = int(chunks[-1] - sum(chunks) + N)
			p = multiprocessing.Pool(int(num_processes))
			option_value = p.map(value_in_pool, chunks)
			p.close()
			p.join()
			q = multiprocessing.Pool(int(num_processes))
			option_delta = q.map(delta_in_pool, chunks)
			q.close()
			q.join()
			end = time.time()
			print 'Number of processors:', num_processes + ',',
			print 'Number of simulations:', str(int(N))
			print 'Monte Carlo Option Price:', str(mean(option_value)) + ',',
			print 'Monte Carlo Option Delta:', str(mean(option_delta))
			print 'Time, in sec:', str(end - start)
			print '-' * 85
