import numpy as np
import scipy.stats as si


def valueOptionBlackScholes(S_t, K, r, vol, tau):
    '''
    Description
    -----------
    Calculates the value of a European call option 
    using the Black-Scholes formula.

    Parameters
    ----------
    `S_t` : float
        Current stock price at t.
    `K` : float
        Strike price.
    `r` : float
        Risk-free interest rate.
    `vol` : float
        Volatility.
    `tau` : float
        Time to expiration, tau = T - t.
    '''
    d1 = (np.log(S_t/K) + (r + (vol**2)/2)*tau) / (vol*np.sqrt(tau))
    N_d1 = si.norm.cdf(d1)
    d2 = d1 - vol*np.sqrt(tau)
    N_d2 = si.norm.cdf(d2)

    return S_t*N_d1 - np.exp(-r*tau)*K*N_d2


if __name__ == '__main__':

    ### Example Usage ###
    T = 1        # maturity (years)
    K = 99       # strike price at t = T
    r = 0.06     # interest rate
    S_0 = 100    # stock price at t = 0
    sigma = 0.2  # volatility
    N = 50       # timesteps
    
    value = valueOptionBlackScholes(S_0, K, r, sigma, T)
    print(f'Value of the option: {value:.2f}')
    