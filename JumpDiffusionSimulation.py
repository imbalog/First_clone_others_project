import numpy as np
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
            self.s0 = float(s0)
            self.strike = float(strike)
            self.maturity = float(maturity)
            self.int_rates = float(int_rates)
            self.dividend_rates = float(dividend_rates)
            self.sigma = float(sigma)
            self.model = str(model)
        except ValueError:
            print('Error passing Options parameters')

        models = ['BlackScholes', 'MonteCarlo', 'BinomialTree', 'JumpDiffusion']
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

class JumpDiffusion(EuropeanOption):

    def __init__(self, option_type, s0, strike, maturity, int_rates, dividend_rates,
                 sigma, jump_lambda, jump_size, jump_std, time_intervals, simulations = 10000):
        EuropeanOption.__init__(self,option_type, s0, strike,
                                maturity, int_rates, dividend_rates, sigma, 'JumpDiffusion')
        try:
            self.jump_lambda = float(jump_lambda)
            assert jump_lambda > 0
            self.jump_size = float(jump_size)
            self.jump_std = float(jump_std)
            assert jump_std > 0
            self.time_intervals = int(time_intervals)
            assert time_intervals > 0
            self.simulations = int(simulations)
            assert simulations > 0
        except ValueError:
            print('Error passing the Jump parameters')


    def generate_simulation(self, seed = 1234567890):
        dt = self.maturity / self.time_intervals
        jump_drift = self.jump_lambda * (np.exp(self.jump_size + 0.5 * self.jump_std ** 2) - 1)
        shape = (self.time_intervals + 1, self.simulations)
        prices = np.zeros((self.time_intervals + 1, self.simulations), dtype=np.float)
        prices[0] = self.s0
        np.random.seed(seed)
        gauss_price = np.random.standard_normal(shape)
        gauss_jump = np.random.standard_normal(shape)
        poisson_jump = np.random.poisson(self.jump_lambda * dt, shape)

        for t in xrange(1, self.time_intervals + 1):
                prices[t] = prices[t - 1] * (np.exp((self.int_rates - jump_drift - 0.5 * self.sigma ** 2) * dt +
                                                    self.sigma * np.sqrt(dt) * gauss_price[t]) +
                                             (np.exp(self.jump_size + self.jump_std * gauss_jump[t]) - 1) *
                                             poisson_jump[t])
        return prices

    @property
    def value(self):
        prices_simulation = self.generate_simulation()
        discount = np.exp(-self.int_rates * self.maturity)
        if self.option_type == 'call':
            return discount * np.sum(np.maximum(prices_simulation[-1] - self.strike, 0)) / float(self.simulations)
        else:
            return discount * np.sum(np.maximum(self.strike - prices_simulation[-1], 0)) / float(self.simulations)

