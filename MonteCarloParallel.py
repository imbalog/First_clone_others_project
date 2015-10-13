class EuropeanCallOption:
	def __init__(self, sigma, S, K, T, r):
		self.sigma = sigma
		self.S = float(S)
		self.K = float(K)
		self.T = float(T)
		self.r = float(r)

	def priceBS(self):
		import numpy as np
		from scipy.stats import norm
		d1 = 1.0 / float(self.sigma * np.sqrt(self.T))
		d1 *= np.log(self.K / self.S) + (self.r + 0.5 * self.sigma ** 2) * self.T
		d2 = d1 - self.sigma * np.sqrt(self.T)
		return norm.cdf(d1) * self.S - norm.cdf(d2) * self.K * np.exp(-1.0 * self.r * self.T)

	def priceMC(self, simulations=1, seed=1234567890, antithetic=True):
		import numpy as np
		from numpy.random import RandomState
		if seed is not None:
			assert type(seed) is int
			rnd = RandomState(seed)
		else:
			rnd = RandomState()
		sum_payoff = 0
		for i in range(simulations):
			brownian = rnd.normal()
			sum_payoff += np.maximum(self.S *
			                         np.exp((self.r - 0.5 * self.sigma**2) * self.T +
			                                self.sigma * np.sqrt(self.T) * brownian) -
			                         self.K, 0)
		return np.exp(-1.0 * self.r * self.T) * sum_payoff / float(simulations)


def eval_price_in_pool(simulations):
	call = EuropeanCallOption(0.2, 100, 100, 1, 0.05)
	return call.priceMC(simulations, seed=os.getpid())


if __name__ == '__main__':
	import multiprocessing
	from numpy import ceil, mean
	import time
	import os
	c = EuropeanCallOption(0.2, 100, 100, 1, 0.05)
	print 'BS Price:', c.priceBS()
	print '-' * 75
	scenarios = {'1': [1e3, 5e6], '2': [1e3, 5e6], '4': [1e3, 5e6], '8': [1e3, 5e6]}
	for num_processes in scenarios:
		for N in scenarios[num_processes]:
			start = time.time()
			chunks = [int(ceil(N / int(num_processes)))] * int(num_processes)
			chunks[-1] = int(chunks[-1] - sum(chunks) + N)
			p = multiprocessing.Pool(int(num_processes))
			option_price = p.map(eval_price_in_pool, chunks)
			p.close()
			p.join()
			end = time.time()
			print 'Number of processors:', num_processes + ',',
			print 'Number of simulations:', str(int(N))
			print 'Monte Carlo Option Price:', str(mean(option_price)) + ',',
			print 'Time, in sec:', str(end - start)
			print '-' * 75
