# Valuation of European call/put options in Black-Scholes (BS) Model


import numpy as np
from scipy.stats import norm


class optionBS(object):
	""" Class for European options in BS model.
	S0 : float : initial stock/index level
	strike : float : strike price
	T : float : time to maturity (in year fractions)
	r : float : constant risk-free short rate
	div :    float : dividend yield
	sigma :  float : volatility factor in diffusion term """

	def __init__(self, option_type, S0, strike, T, r, div, sigma):
		try:
			self.option_type = option_type
			assert isinstance(option_type, str)
			self.S0 = float(S0)
			self.strike = float(strike)
			self.T = float(T)
			self.r = float(r)
			self.div = float(div)
			self.sigma = float(sigma)
		except ValueError:
			print('Error passing Options parameters')

		if option_type != 'call' and option_type != 'put':
			raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
		if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
			raise ValueError('Error: Negative inputs not allowed')

		d1 = (np.log(self.S0 / self.strike) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
			self.sigma * np.sqrt(self.T))
		d2 = (d1 - self.sigma * np.sqrt(self.T))

		self.Nd1 = norm.cdf(d1, 0, 1)
		self.Nnd1 = norm.cdf(-d1, 0, 1)
		self.Nd2 = norm.cdf(d2, 0, 1)
		self.Nnd2 = norm.cdf(-d2, 0, 1)

	def valueBS(self):
		""" Returns BS option value """
		if self.option_type == 'call':
			value = (self.S0 * np.exp(-self.div * self.T) * self.Nd1
			         - self.strike * np.exp(-self.r * self.T) * self.Nd2)
		else:
			value = (self.strike * np.exp(-self.r * self.T) * self.Nnd2
			         - self.S0 * np.exp(-self.div * self.T) * self.Nnd1)
		return value

	def deltaBS(self):
		"""  Returns BS Delta value """
		if self.option_type == 'call':
			delta = np.exp(-self.div * self.T) * self.Nd1
		else:
			delta = np.exp(-self.div * self.T) * (self.Nd1 - 1)
		return delta

	def vegaBS(self):
		""" Returns BS Vega of option. """
		vega = self.S0 * self.Nd1 * np.sqrt(self.T)
		return vega

	def gammaBS(self):
		""" Returns BS Gamma of option. """
		gamma = (self.Nd1 * np.exp(-self.div * self.T)
		         / self.S0 * self.sigma * np.sqrt(self.T))
		return gamma

	def rhoBS(self):
		""" Returns BS Rho of option"""
		if self.option_type == 'call':
			rho = self.S0 * self.T * np.exp(-self.r * self.T) * self.Nd2
		else:
			rho = - self.S0 * self.T * np.exp(-self.r * self.T) * self.Nnd2
		return rho

	def thetaBS(self):
		""" Returns BS Theta of option"""
		if self.option_type == 'call':
			theta = (self.S0 * self.Nd1 * self.sigma * np.exp(-self.r * self.T)) / 2 * np.sqrt(self.T) \
			        + self.div * self.S0 * self.Nd1 * np.exp(-self.r * self.T) \
			        - self.r * self.S0 * np.exp(-self.r * self.T) * self.Nd2
		else:
			theta = (self.S0 * self.Nd1 * self.sigma * np.exp(-self.r * self.T)) / 2 * np.sqrt(self.T) \
			        + self.div * self.S0 * self.Nnd1 * np.exp(-self.r * self.T) \
			        - self.r * self.S0 * np.exp(-self.r * self.T) * self.Nnd2
		return theta

class optionMC(object):

	""" Class for European options using Monte Carlo Simulation
	S0 :     float : initial stock/index level
	strike : float : strike price
	T :      float : time to maturity (in year fractions)
	r :      float : constant risk-free short rate
	div :    float : dividend yield
	sigma :  float : volatility factor in diffusion term
	simul :  int: number of simulations
	t_steps: int: number of time steps per simulation """

	def __init__(self, option_type, S0, strike, T, r, div, sigma, simul, seed = None):
		try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            self.T = float(T)
            self.r = float(r)
            self.div = float(div)
            self.sigma = float(sigma)
            self.simul = int(simul)
            self.seed = seed
            if self.seed is not None:
                assert type (self.seed) is int
                self.rnd = np.random.RandomState(self.seed)
            else :
                self.rnd = np.random.RandomState()
        except ValueError:
            print('Error passing Options parameters')

        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 \
            or div < 0 or sigma < 0 or simul <= 0 :
            raise ValueError('Error: Negative inputs not allowed')

    def MonteCarlo(self):
        """ Returns MC option value """
        z = self.rnd.normal(size = self.simul)
        price_T = self.S0 * np.exp(self.r - self.div - 0.5 * self.sigma ** 2
                                      + self.sigma * np.sqrt(self.T) * z)
        if self.option_type == 'call':
            payoff = np.maximum((price_T - self.strike), np.zeros(self.simul))
        else:
            payoff = np.maximum((self.strike - price_T), np.zeros(self.simul))
        return  np.exp(-1.0 * self.r * self.T) * np.sum(payoff)/self.simul

    def MC_Antith(self):
        """ Returns MC option value using Antithetic variable """
        z1 = self.rnd.normal(size = self.simul)
        price_T_1 = self.S0 * np.exp(self.r - self.div - 0.5 * self.sigma ** 2
                                      + self.sigma * np.sqrt(self.T) * z1)
        price_T_2 = self.S0 * np.exp(self.r - self.div - 0.5 * self.sigma ** 2
                                      + self.sigma * np.sqrt(self.T) * -1.0 * z1)
        if self.option_type == 'call':
            payoff = 0.5 * (np.maximum((price_T_1 - self.strike), np.zeros(self.simul)) +
                            np.maximum((price_T_2 - self.strike), np.zeros(self.simul)) )
        else:
            payoff = 0.5 * (np.maximum((self.strike - price_T_1), np.zeros(self.simul)) +
                            np.maximum((self.strike - price_T_2), np.zeros(self.simul)) )
        return (np.exp(-1.0 * self.r * self.T) * np.sum(payoff)/self.simul)


