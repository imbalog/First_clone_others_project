import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod


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
    __metaclass__ = ABCMeta

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
                raise ValueError('Error: Negative inputs not allowed.')
            if vol1 < 0 or vol2 < 0:
                raise ValueError('Error: Negative volatilities are not allowed.')
            if rho > 1 or rho < -1:
                raise ValueError('Error: Correlation out of range')
            if CallPut != 1 and CallPut != -1:
                raise ValueError('Error: For a Call: CallPut=1, or -1 for a Put')
            models = ['Margrabe', 'Kirk', 'MonteCarlo']
            if model not in models:
                raise Exception('Error: Model Unknown')
        except ValueError:
            print('Error passing spread option inputs')

    def getmodel(self):
        return self.model

    def __str__(self):
        return "This SpreadOption is priced using {0}".format(self.model)

    @abstractmethod
    def price(self):
        pass


class Margrabe(SpreadOption):
    def __init__ (self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Margrabe")
        if K != 0:
            raise ValueError('Strike should be null to use Margrabe')

    @property
    def price(self):
        vol = np.sqrt(self.vol1 ** 2 + self.vol2 ** 2 - 2 * self.rho * self.vol1 * self.vol2)
        d1 = ( np.log(self.S2_t / self.S1_t) / (vol * np.sqrt(self.T)) + 0.5 * vol * np.sqrt(self.T))
        d2 = d1 - vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t  * norm.cdf(self.CallPut * d1, 0, 1)
                                  - self.S1_t * norm.cdf(self.CallPut * d2, 0, 1)))
        return price


class Kirk(SpreadOption):
    def __init__ (self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Kirk")

    @property
    def price(self):

        z = self.S1_t / (self.S1_t + self.K * np.exp(-1. * self.r * self.T))

        vol = np.sqrt( self.vol1 ** 2 * z ** 2 + self.vol2 ** 2 - 2 * self.rho* self.vol1 * self.vol2 * z )
        d1 = (np.log(self.S2_t / (self.S1_t + self.K * np.exp(-self.r * self.T)))
              / (vol * np.sqrt(self.T)) + 0.5 * vol * np.sqrt(self.T))
        d2 = d1 - vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t  * norm.cdf(self.CallPut * d1, 0, 1)
                                 - (self.S1_t + self.K * np.exp(-self.r * self.T))
                                 * norm.cdf(self.CallPut * d2, 0, 1)))
        return price


class MonteCarlo(SpreadOption):

    def __init__ (self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, simulations):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Monte Carlo")
        self.simulations = int(simulations)
        try:
            if self.simulations > 0 :
                assert isinstance(self.simulations, int)
        except:
            raise ValueError("Simulation's number has to be positive integer")

    def generate_spreads(self, seed = 12345678 ):
        try:
            if seed is not None:
                assert isinstance(seed, int)
        except:
            print 'Error passing seed'
        np.random.seed(seed)
        B1 = np.sqrt(self.T) * np.random.randn(self.simulations, 1)
        B2 = np.sqrt(self.T) * np.random.randn(self.simulations, 1)
        S1_T = self.S1_t * np.exp ((self.r - 0.5 * self.vol1 ** 2) * self.T + self.vol1 *  B1)
        S2_T = self.S2_t * np.exp ((self.r - 0.5 * self.vol2 ** 2) * self.T +
                                    self.vol2 * ( self.rho * B1 + np.sqrt(1 - self.rho ** 2) * B2))
        if self.CallPut == 1 :
            payoff = np.maximum((S2_T - S1_T - self.K), 0)
        else:
            payoff = np.maximum((self.K - S2_T - S1_T), 0)

        return np.exp(-1. * self.r * self.T) * payoff

    @property
    def price(self):
        price = np.sum(self.generate_spreads()) / float(self.simulations)
        return price