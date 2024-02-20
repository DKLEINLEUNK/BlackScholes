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


def compute_hedge_parameter_black_scholes(S_t, K, r, vol, tau):

    d1 = (np.log(S_t/K) + (r + (vol**2)/2)*tau) / (vol*np.sqrt(tau))
    N_d1 = si.norm.cdf(d1)

    return N_d1


def generate_stock_path(T,M,S_0,r,vol):

    '''
    Description
    -----------
    Generates a discrete time stock path using Euler's method 
    using the Black-Scholes formula.

    Parameters
    ----------
    `T` : int
        Time till expiration.
    `M` : int
        Number of time steps.
    `S_0` : float
        Stock price today.
    `r` : float
        Risk free rate.
    `vol` : float
        Stock volatility.
    '''

    time_steps = np.arange(0,M,1)

    dt = T/(len(time_steps)) #Length of time interval

    S_m = S_0 #Initialise starting stock price

    stock_prices = [S_m] #A list to store the whole stock path, starting with the initial price of the stock

    for m in range (1,len(time_steps)): #For each time step
        
        Zm = np.random.normal(0,1) #Random number to be used for weiner process

        S_new = S_m + (r*S_m*dt) + (vol*S_m*np.sqrt(dt)*Zm)

        stock_prices.append(S_new)

        S_m = S_new #Set s_m to the new current value of the stock


    return stock_prices, time_steps


def delta_hedge_simulation(stock_prices,num_intervals):
     
    
        '''
        Description
        -----------
        
        Determines the P&L of a short European call hedged with delta hedging

        Parameters
        ----------
        `stock_prices` : int
            Time till expiration.
        `M` : int
            Number of time steps.
        `S_0` : float
            Stock price today.
        `r` : float
            Risk free rate.
        `vol` : float
            Stock volatility.
        '''


        step = len(stock_prices)//num_intervals #// Ensures rounding to whole number

        extracted_stock_prices = [stock_prices[i] for i in range(0,len(stock_prices),step)] #Extracting the stock prices at indexes corresponding to number of intervals
        
             
        return extracted_stock_prices
    

       





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
    