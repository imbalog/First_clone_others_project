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

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(- self.r * self.T)

    @property
    def GeometricAsianOption(self):
        sigsqT = ((self.sigma ** 2 * self.T * (self.M + 1) * (2 * self.M + 1))
                  / (6 * self.M * self.M))
        muT = (0.5 * sigsqT + (self.r - 0.5 * self.sigma ** 2)
               * self.T * (self.M + 1) / (2 * self.M))
        d1 = ((np.log(self.S0 / self.strike) + (muT + 0.5 * sigsqT))
              / np.sqrt(sigsqT))
        d2 = d1 - np.sqrt(sigsqT)
        N1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
        N2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
        geometric_value = self.discount * (self.S0 * np.exp(muT) * N1 - self.strike * N2)
        return geometric_value

    @property
    def price_path(self, seed = 1234567890):
        np.random.seed(seed)
        price_path = (self.S0 *
                      np.cumprod (np.exp ((self.r - 0.5 * self.sigma ** 2) * self.time_unit + /
                                    self.sigma * np.sqrt(self.time_unit) /
                                          * np.random.randn(self.simulations, self.M)), 1))
        return price_path

    @property
    def MCPayoff(self): 
        if self.option_type == 'call':
            MCpayoff = self.discount \
                              * np.maximum(np.mean(self.price_path,1)-self.strike, 0)
        else:
            MCpayoff = self.discount \
                              * np.maximum(self.strike - np.mean(self.price_path,1), 0)
        return MCpayoff
    
    @property 
    def value(self):
        MCvalue = np.mean(self.MCPayoff)
        MCValue_std = np.std(self.MCPayoff)
        upper_bound = MCvalue + 1.96 * MCValue_std/np.sqrt(self.simulations)
        lower_bound = MCvalue - 1.96 * MCValue_std/np.sqrt(self.simulations)
        return MCvalue, lower_bound, upper_bound
    
    @property
    def value_with_control_variate(self):
        
        geometric_average = np.exp( (1/float(self.M)) * np.sum(np.log(self.price_path), 1) )
        if self.option_type == 'call':
            MCpayoff_geometric = self.discount * np.maximum(geometric_average - self.strike, 0)
        else:
            MCpayoff_geometric = self.discount * np.maximum(self.strike - geometric_average, 0)
        value_with_CV = self.MCPayoff + self.GeometricAsianOption - MCpayoff_geometric        
        value_with_control_variate = np.mean(value_with_CV , 0)
        value_with_control_variate_std = np.std(value_with_CV, 0)
        upper_bound_CV = value_with_control_variate + 1.96 * value_with_control_variate_std/np.sqrt(self.simulations)
        lower_bound_CV = value_with_control_variate - 1.96 * value_with_control_variate_std/np.sqrt(self.simulations)        
        return value_with_control_variate, lower_bound_CV, upper_bound_CV