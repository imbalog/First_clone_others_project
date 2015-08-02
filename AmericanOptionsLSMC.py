import numpy as np

class AmericanOptionsLSMC(object):
    """ Class for American options pricing using Longstaff-Schwartz (2001):
    "Valuing American Options by Simulation: A Simple Least-Squares Approach."
    Review of Financial Studies, Vol. 14, 113-147.
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
    def MCprice_matrix(self):
        """ Returns MC price matrix rows: time columns: price-path simulation """
        np.random.seed(123)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations),dtype=np.float64)
        MCprice_matrix[0,:] = self.S0
        for t in xrange(1, self.M + 1):
            ran = np.random.standard_normal( self.simulations / 2)
            ran = np.concatenate((ran, -ran))
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                  * np.exp((self.r - self.sigma ** 2 / 2) * self.time_unit
                                  + self.sigma * ran * np.sqrt(self.time_unit)))
        return MCprice_matrix

    @property
    def MCpayoff(self):
        """Returns the inner-value of American Option"""
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix - self.strike,
                           np.zeros((self.M + 1, self.simulations),dtype=np.float64))
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix,
                            np.zeros((self.M + 1, self.simulations),dtype=np.float64))
        return payoff

    @property
    def value_vector(self):
        value_matrix = np.zeros_like(self.MCpayoff)
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        for t in range(self.M - 1, 0 , -1):
            regression = np.polyfit(self.MCprice_matrix[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value,
                                          self.MCpayoff[t, :],
                                          value_matrix[t + 1, :] * self.discount)

        return value_matrix[1,:] * self.discount


    @property
    def LSMC_value(self): return np.sum(self.value_vector) / self.simulations


