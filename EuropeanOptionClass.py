import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod

__author__ = 'Jesus Perez Colino'

class EuropeanOption(object):
    """ Abstract Class for European options.
    Not for direct use. Partially implemented.
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

        models = ['BlackScholes', 'MonteCarlo', 'BinomialTree', 'FFT', 'PDE']

        if model not in models:
            raise Exception('Error: Model unknown')

        option_types = ['call', 'put']

        if option_type not in option_types:
            raise ValueError("Error: Option type not valid. Enter 'call' or 'put'")
        if s0 < 0 or strike < 0 or maturity <= 0 or int_rates < 0 or dividend_rates < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.discount = np.exp(-self.r * self.T)

    @property
    def getmodel(self):
        return self.model

    def __str__(self):
        return "This EuropeanOption is priced using {0}".format(self.getmodel)

    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def delta(self):
        pass

    @abstractmethod
    def gamma(self):
        pass

    @abstractmethod
    def vega(self):
        pass

    @abstractmethod
    def rho(self):
        pass

    @abstractmethod
    def theta(self):
        pass


class BlackScholes(EuropeanOption):

    def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates, sigma):
        EuropeanOption.__init__(self,option_type, s0, strike,
                                maturity, int_rates, dividend_rates,
                                sigma, 'BlackScholes')

        self.d1 = (np.log(self.S0 / self.strike) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T))
        self.d2 = (self.d1 - self.sigma * np.sqrt(self.T))
        self.normal_d1 = norm.cdf(self.d1, 0, 1)
        self.normal_neg_d1 = norm.cdf(-self.d1, 0, 1)
        self.normal_d2 = norm.cdf(self.d2, 0, 1)
        self.normal_neg_d2 = norm.cdf(-self.d2, 0, 1)

    @property
    def value(self):

        if self.option_type == 'call':
            value = (self.S0 * np.exp(-self.div * self.T) * self.normal_d1 -
                     self.strike * np.exp(-self.r * self.T) * self.normal_d2)
        else:
            value = (self.strike * np.exp(-self.r * self.T) * self.normal_neg_d2 -
                     self.S0 * np.exp(-self.div * self.T) * self.normal_neg_d1)
        return value

    @property
    def delta(self):
        if self.option_type == 'call':
            delta = np.exp(-self.div * self.T) * self.normal_d1
        else:
            delta = np.exp(-self.div * self.T) * (self.normal_d1 - 1)
        return delta

    @property
    def vega(self):
        vega = self.S0 * self.normal_d1 * np.sqrt(self.T)
        return vega

    @property
    def gamma(self):
        gamma = (self.normal_d1 * np.exp(-self.div * self.T)
                 / self.S0 * self.sigma * np.sqrt(self.T))
        return gamma

    @property
    def rho(self):
        if self.option_type == 'call':
            rho = self.S0 * self.T * np.exp(-self.r * self.T) * self.normal_d2
        else:
            rho = - self.S0 * self.T * np.exp(-self.r * self.T) * self.normal_neg_d2
        return rho

    @property
    def theta(self):
        if self.option_type == 'call':
            theta = ((self.S0 * self.normal_d1 * self.sigma * np.exp(-self.r * self.T)) / 2 * np.sqrt(self.T) +
                     self.div * self.S0 * self.normal_d1 * np.exp(-self.r * self.T) -
                     self.r * self.S0 * np.exp(-self.r * self.T) * self.normal_d2)
        else:
            theta = ((self.S0 * self.normal_d1 * self.sigma * np.exp(-self.r * self.T)) / 2 * np.sqrt(self.T) +
                    self.div * self.S0 * self.normal_neg_d1 * np.exp(-self.r * self.T) -
                    self.r * self.S0 * np.exp(-self.r * self.T) * self.normal_neg_d2)
        return theta


class MonteCarlo(EuropeanOption):

    def __init__(self, option_type, S0, strike, T, r, div, sigma, simulations = 100000):
        EuropeanOption.__init__(self, option_type, S0, strike,
                                T, r, div, sigma, "MonteCarlo")
        self.simulations = int(simulations)
        try:
            if self.simulations > 0:
                assert isinstance(self.simulations, int)
        except:
            raise ValueError("Simulation's number has to be positive integer")

    def simulation_terminal(self, seed = 123456):
        np.random.seed(seed)
        brownian = np.random.standard_normal(size = self.simulations)
        price_terminal = self.S0 * np.exp((self.r - self.div - 0.5 * self.sigma ** 2) * self.T +
                                          self.sigma * np.sqrt(self.T) * brownian)
        return price_terminal

    def generate_payoffs(self):
        price_terminal = self.simulation_terminal()
        if self.option_type == 'call':
            payoff = np.maximum((price_terminal - self.strike), np.zeros(self.simulations))
        else:
            payoff = np.maximum((self.strike - price_terminal), np.zeros(self.simulations))
        return payoff

    @property
    def value(self):
        payoff = self.generate_payoffs()
        return self.discount * np.sum(payoff) / float(self.simulations)

    @property
    def delta(self):
        value_terminal = np.array(self.simulation_terminal() / float(self.S0))
        payoff = self.generate_payoffs()
        delta = np.zeros(len(payoff))
        delta[np.nonzero(payoff)] = value_terminal[np.nonzero(payoff)]
        return self.discount * np.sum(delta) / float(self.simulations)

    @property
    def gamma(self):  # TODO
        pass

    @property
    def vega(self):  # TODO
        pass

    @property
    def rho(self):  # TODO
        pass

    @property
    def theta(self):  # TODO
        pass


class BinomialTree(EuropeanOption):

    def __init__(self, option_type, S0, strike, T, r, div, sigma, time_grid ):
        EuropeanOption.__init__(self, option_type, S0, strike,
                                T, r, div, sigma, "BinomialTree")
        self.time_grid = int(time_grid)
        if self.time_grid <= 0 :
            raise ValueError("Error: Number of time steps has to be positive integer")

    @property
    def value(self):
        delta_time = self.T / self.time_grid
        discount = np.exp(-self.r * delta_time)
        move_up = np.exp(self.sigma * np.sqrt(delta_time))
        move_down = 1 / float(move_up)
        probability = (np.exp(self.r * delta_time) - move_down) / float(move_up - move_down)
        index_up = np.arange(self.time_grid + 1)
        index_up = np.resize(index_up, (self.time_grid + 1, self.time_grid + 1))
        index_down = np.transpose(index_up)
        index_up = move_up ** (index_up - index_down)
        index_down = move_down ** index_down
        price = self.S0 * index_up * index_down
        if self.option_type == 'call':
            value = np.maximum(price - self.strike, 0)
        else:
            value = np.maximum(self.strike - price, 0)
        z=0
        for t in range(self.time_grid - 1, -1, -1):
            value[0:self.time_grid - z, t] = ((probability *
                                               value[0:self.time_grid - z, t + 1] +
                                               (1 - probability) *
                                               value[1:self.time_grid - z + 1, t + 1]) * discount)
            z += 1
        return value[0, 0]

    @property
    def delta(self):  # TODO
        pass

    @property
    def gamma(self):  # TODO
        pass

    @property
    def vega(self):  # TODO
        pass

    @property
    def rho(self):  # TODO
        pass

    @property
    def theta(self):   # TODO
        pass