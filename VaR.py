__author__ = 'JPC'
from pandas.io.data import DataReader, DataFrame
from scipy.stats import norm
from datetime import datetime
import numpy as np


class VaR(object):
    def __init__(self, stock_code, confidence_level, time_horizon, method, volume=1, simulations=100000):
        try:
            self.stock_code = str(stock_code)
            self.confidence_level = float(confidence_level)
            self.time_horizon = int(time_horizon)
            self.method = str(method)
            self.volume = int(volume)
            self.simulations  = int(simulations)

            if confidence_level < 0 or confidence_level > 1:
                raise ValueError('Confidence_level value not allowed.')
            if time_horizon < 0 or time_horizon > 1000:
                raise ValueError('Time_horizon value not allowed.')
            if volume <= 0 :
                raise ValueError('Volumes(number of shares) has to be positive')
            if simulations <= 0 or simulations > 10000000:
                raise ValueError('Simulations value not allowed')
            methods = ['Historical', 'Var_Cov', 'Monte_Carlo']
            if method not in methods:
                raise ValueError('Method unknown')
        except ValueError:
            print('Error passing VaR inputs')

    def getprices(self):
        prices = DataReader(name = self.stock_code,
                            data_source = "yahoo",
                            start = datetime(2013, 1, 1),
                            end = datetime(2014, 12, 31))
        return prices['Adj Close']

    def getreturns(self):
        prices = self.getprices()
        return prices.pct_change().dropna()

    def getmethod(self):
        return self.method

    def __str__(self):
        return 'VaR estimation using {0} method'.format(self.getmethod())


class Historical(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon = 1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'Historical')

    @property
    def value(self):
        returns = VaR.getreturns(self)
        return returns.quantile(self.confidence_level) * self.volume

class Var_Cov(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon = 1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'Var_Cov')

    @property
    def value(self):
        returns = VaR.getreturns(self)
        return norm.ppf(self.confidence_level,
                        returns.mean(),
                        returns.std()) * self.volume


class Monte_Carlo(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon = 1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'Monte_Carlo')

    def random_walk(self, mu, sig, T, initial_price):
        brownian = np.sqrt(T) * np.random.randn(self.simulations, 1)
        price_T = initial_price * np.exp((mu - 0.5 * sig ** 2) * T + sig * brownian)
        return price_T

    @property
    def value(self):
        prices = VaR.getprices(self)
        initial_price = prices[-1]
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sig = returns.std() * np.sqrt(252)
        T = self.time_horizon / 252.
        simulations = self.random_walk(mu, sig, T, initial_price)
        return np.log(np.percentile(simulations, int(self.confidence_level * 100)) / float(initial_price))
