__author__ = 'JPC'

import numpy as np
from scipy.stats import norm

class SpreadOption(object):

    """ Class for valuation of European Spreads Options (Calls/P.uts).
    S1_t : float : initial forward price/index level
    S2_t : float : initial forward price/index level
    K : float : strike price
    T : float : maturity (in year fractions)
    r : float : constant risk-free short rate
    vol1 : float : volatility factor in diffusion term (std)
    vol2 : float : volatility factor in diffusion term (std)
    rho : float:
    CallPut : integer : 1 for a Call, and -1 for a Put
    """

    def __init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, model):
        try:
            self.S1_t = float(S1_t)
            self.S2_t = float(S2_t)
            self.K = float(K)
            self.T = float (T)
            self.r = float(r)
            self.vol1 = float(vol1)
            self.vol2 = float(vol2)
            self.rho = float(rho)
            self.CallPut = int(CallPut)
            self.model = str(model)
            if T < 0 or r < 0 or S1_t < 0 or S2_t < 0:
                raise ValueError('Negative inputs not allowed.')
            if vol1 < 0 or vol2 < 0:
                raise ValueError('Negative volatilities are not allowed.')
            if rho > 1 or rho < -1:
                raise ValueError('Correlation out of range')
            if CallPut != 1 and CallPut != -1:
                raise ValueError('For a Call: CallPut=1, or -1 for a Put')
        except ValueError:
            print('Error passing spread option inputs')

    def getmodel(self):
        return self.model

    def __str__(self):
        return "This SpreadOption is solved using {0}".format(self.model)


class margrabe(SpreadOption):
    def __init__ (self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Margrabe")
        if K != 0:
            raise ValueError('Strike should be null to use Margrabe')

    @property
    def price(self):
        vol = self.vol1 ** 2 + self.vol2 ** 2 - 2 * self.rho* self.vol1 * self.vol2
        d0 = (np.log(self.S2_t / self.S1_t) / (vol * np.sqrt(self.T)) - 0.5 * vol * np.sqrt(self.T))
        d1 = d0 + vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t  * norm.cdf(self.CallPut * d1, 0, 1)
                                  - self.S1_t * norm.cdf(self.CallPut * d0, 0, 1)))
        return price

class kirk(SpreadOption):
    def __init__ (self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Kirk")

    @property
    def price(self):
        z = self.S1_t / (self.S1_t + self.K * np.exp(-self.r * self.T))
        vol = self.vol1 ** 2 * z + self.vol2 ** 2 - 2 * self.rho* self.vol1 * z * self.vol2
        d0 = (np.log(self.S2_t / (self.S1_t + self.K * np.exp(-self.r * self.T)))
              / (vol * np.sqrt(self.T))
              - 0.5 * vol * np.sqrt(self.T))
        d1 = d0 + vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t  * norm.cdf(self.CallPut * d1, 0, 1)
                                  - self.S1_t * norm.cdf(self.CallPut * d0, 0, 1)))
        return price
