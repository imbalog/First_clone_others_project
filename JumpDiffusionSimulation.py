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

    @property
    def getmodel(self):
        return self.model

    def __str__(self):
        return "This EuropeanOption is priced using {0}".format(self.getmodel)

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
        dt = self.maturity / float(self.time_intervals)
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

class JumpDiffusionFFT(EuropeanOption):

    def __init__(self, option_type, s0, strike, maturity,
                 int_rates, dividend_rates, sigma,
                 jump_lambda, jump_size, jump_std,
                 time_intervals, simulations = 10000):
        EuropeanOption.__init__(self,option_type, s0, strike,
                                maturity, int_rates, dividend_rates,
                                sigma, 'JumpDiffusion')
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

    def characteristic_function(self, ux, x0, delta):
            omega = x0 / self.maturity + self.int_rates - 0.5 * self.sigma ** 2 \
                - self.jump_lambda * (np.exp(self.jump_size + 0.5 * delta ** 2) - 1)
            value = np.exp((1j * ux * omega - 0.5 * ux ** 2 * self.sigma ** 2 +
                self.jump_lambda * (np.exp(1j * ux * self.jump_size - ux ** 2 * delta ** 2 * 0.5) - 1)) * self.maturity)
            return value

    @property
    def value(self):
    ''' Valuation of European call option in M76 model via
    Carr-Madan (1999) Fourier-based approach.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    call_value: float
        European call option present value
    '''
    k = np.log(K / S0)
    x0 = math.log(S0 / S0)
    g = 2  # factor to increase accuracy
    N = g * 4096
    eps = (g * 150.) ** -1
    eta = 2 * math.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
    # Modificatons to Ensure Integrability
    if S0 >= 0.95 * K:  # ITM case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        mod_char_fun = math.exp(-r * T) * characteristic_function(
                                    v, x0, T, r, sigma, lamb, mu, delta) \
                / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
    else:  # OTM case
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo - 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
        v = (vo + 1j * alpha) - 1j
        mod_char_fun_2 = math.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo + 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))

    # Numerical FFT Routine
    delt = np.zeros(N, dtype=np.float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
        payoff = (fft(fft_func)).real
        call_value_m = np.exp(-alpha * k) / math.pi * payoff
    else:
        fft_func = (np.exp(1j * b * vo)
                    * (mod_char_fun_1 - mod_char_fun_2)
                    * 0.5 * eta * SimpsonW)
        payoff = (fft(fft_func)).real
        call_value_m = payoff / (np.sinh(alpha * k) * math.pi)
    pos = int((k + b) / eps)
    call_value = call_value_m[pos]
    return call_value * S0