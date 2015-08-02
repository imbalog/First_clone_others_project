import numpy as np
from scipy.special import erf


class AsianOptionMC_MC(object):
    """ Class for Asian options pricing using control variate
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            self.T = float(T)
            self.M = int(M)
            self.r = float(r)
            self.div = float(div)
            self.sigma = float(sigma)
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Options parameters')

        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / self.M
        self.discount = np.exp(-self.r * self.time_unit)

    @property
    def GeometricAsianOption(self):

        sigsqT = (self.sigma ** 2 * self.T *
                  (self.time_unit + 1) * (2 * self.time_unit + 1)
                  / (6 * self.time_unit * self.time_unit))
        muT = (0.5 * sigsqT + (self.r - 0.5 * self.sigma ** 2)
               * self.T * (self.time_unit + 1) / (2 * self.time_unit))

        d1 = (np.log(self.S0 / self.strike) + (muT + 0.5 * sigsqT)
              / (np.sqrt(sigsqT)))
        d2 = d1 - np.sqrt(sigsqT)

        N1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
        N2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))

        geometric_value = np.exp(-self.r * self.T) * (self.S0 * np.exp(muT) * N1 - self.strike * N2)

        return geometric_value

    @property
    def price_path(self, seed = 100):
        np.random.seed(seed)
        price_path = (self.S0 * /
                      np.cumprod (np.exp ((self.r - 0.5 * self.sigma ** 2) * self.time_unit + /
                                    self.sigma * np.sqrt(self.time_unit) /
                                          * np.random.randn(self.simulations, self.M)), 1))
        return price_path

    @property
    def value(self):
        if self.option_type == 'call':
            MCvalue = np.mean(np.exp(-self.r * self.T) /
                              * np.maximum(np.mean(self.price_path,1)-self.strike, 0))
        else:
            MCvalue = np.mean(np.exp(-self.r * self.T) /
                              * np.maximum(self.strike - np.mean(self.price_path,1), 0))

        upper_bound = MCvalue + 1.96 * MCvalue/np.sqrt(self.simulations)
        lower_bound = MCvalue - 1.96 * MCvalue/np.sqrt(self.simulations)
        return MCvalue, lower_bound, upper_bound

