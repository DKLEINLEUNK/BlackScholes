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


class BlackScholesSimulation:
    
    def __init__(self, S_0, K, r, M, stock_volatility, delta_volatility, T=1):
        '''
        Description
        -----------
        Initialises the parameters for the Black-Scholes simulation.
        
        Parameters
        ----------
        `S_0` : float
            Stock price today.
        `K` : float
            Strike price.
        `r` : float
            Risk-free interest rate.
        `M` : int
            Number of time steps.
        `stock_volatility` : float
            Constant for the volatility of the stock price dynamics.
        `delta_volatility` : float
            Constant for the volatility of the delta hedge parameter.
        `T` : float
            Time to expiration.
        '''
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.stock_volatility = stock_volatility
        self.delta_volatility = delta_volatility
        self.T = T
        self.M = M

        self.generate_stock_path()  # Generates stock path & time steps

        self.hedge_taus = []
        self.hedge_delta_params = []


    def generate_stock_path(self):
        '''
        Description
        -----------
        Generates a discrete time stock path using Euler's method.
        '''

        time_steps = np.arange(0, self.M, 1)
        dt = self.T / (len(time_steps))  # length of time interval
        
        self.dt = dt  # store the time step length

        S_m = self.S_0  # initialises starting stock price
        stock_prices = [S_m]  # = list to store the whole stock path, starting with the initial price of the stock

        for _ in range (1, len(time_steps)):  # for each time step
            
            Zm = np.random.normal(0,1) #Random number to be used for weiner process

            S_new = S_m + (self.r * S_m * dt) + (self.stock_volatility * S_m * np.sqrt(dt) * Zm)

            stock_prices.append(S_new)

            S_m = S_new #Set s_m to the new current value of the stock

        self.stock_prices = stock_prices
        self.time_steps = time_steps


    def compute_profit_loss(self, n_hedges):
        '''
        Description
        -----------
        Calculates the profit/loss of the replicating portfolio.
        '''
        return self.rebalance_portfolio(n_hedges) - self.compute_payoff()
    

    def rebalance_portfolio(self, n_hedges):
        '''
        Description
        -----------
        Rebalances the replicating portfolio as stock price progresses.
        
        Parameters
        ----------
        `n_hedges` : int
            Number of hedges to be performed, e.g. `5` for quarterly, `53` for weekly, `366` for daily.
        
        Returns
        -------
        `float`
            Value of the replicated portfolio at t=T.
        '''
        self.hedge_deltas(n_hedges)  # quarterly hedging
        
        # print(f"Debt at t=T is {self.compute_debt()}")
        # print(f"Value of hedged stocks at t=T is {self.sell_hedged_stocks()}")
        # print(f"Value of the option at t=T is {self.compute_invested_premium_value()}")

        return self.compute_invested_premium_value() + self.sell_hedged_stocks() - self.compute_debt() 


    def compute_invested_premium_value(self):
        '''
        Description
        -----------
        Calculates the option premium value at t=T of the European call option.
        '''
        d1 = (np.log(self.S_0 / self.K) + (self.r + (self.delta_volatility**2) / 2) * self.T) / (self.delta_volatility * np.sqrt(self.T))
        N_d1 = si.norm.cdf(d1)
        
        d2 = d1 - self.delta_volatility * np.sqrt(self.T)
        N_d2 = si.norm.cdf(d2)

        C_0 = self.S_0 * N_d1 - np.exp(-self.r * self.T) * self.K * N_d2

        # print(f"Value of the option at t=0 is {C_0}")
        
        C_1 = C_0 * np.exp(self.r * self.T)  # value of the option at t=T

        return C_1


    def hedge_deltas(self, n_hedges):
        '''
        Description
        -----------        
        Determines the P&L of a short European call hedged with delta hedging.
        '''
        hedge_dates = np.floor(np.linspace(0, len(self.stock_prices) - 1, n_hedges)).astype(int)
        hedge_dates = hedge_dates[:-1]  # removing the last date because this is t=T
        
        self.hedge_dates = hedge_dates  # store the hedge dates
        self.hedge_stock_prices = [self.stock_prices[i] for i in hedge_dates]  # store the stock prices at the hedge dates  
        
        self.compute_deltas()


    def compute_deltas(self):
        '''
        Description
        -----------
        Calculates the delta hedge parameter for a European call option.
        '''

        for i in range( len(self.hedge_dates) ):
        
            S_t = self.hedge_stock_prices[i]        
            tau = ( (self.M - 1) - self.hedge_dates[i] ) * self.dt  # time to expiration
        
            d1 = (np.log(S_t / self.K) + (self.r + (self.delta_volatility**2) / 2) * tau) / (self.delta_volatility * np.sqrt(tau))
            N_d1 = si.norm.cdf(d1)
            
            self.hedge_taus.append(tau)
            self.hedge_delta_params.append(N_d1)


    def compute_debt(self):
        '''
        Description
        -----------
        Uses computed deltas to rebalance portfolio as stock price progresses.
        '''
        # start at t = 0
        debt = 0
        prev_hedge_ratio = 0

        for i in range( len(self.hedge_delta_params) ):
            
            curr_hedge_ratio = self.hedge_delta_params[i]
            stock_price = self.hedge_stock_prices[i]
            tau = self.hedge_taus[i]

            debt += (curr_hedge_ratio - prev_hedge_ratio) * stock_price * np.exp(self.r * tau)
            
            # TODO maybe replace the first tau with 1.0 instead of 0.999
            # print(f"Debt at t={tau} is {self.debt}")

            prev_hedge_ratio = self.hedge_delta_params[i]

        return debt

    
    def sell_hedged_stocks(self):
        '''
        Description
        -----------
        Sells the hedged stocks at t=T.
        '''
        return self.hedge_delta_params[-1] * self.stock_prices[-1]  # last delta parameter before T * final stock price from simulation 
    
    
    def compute_payoff(self):
        '''
        Description
        -----------
        Calculates the payoff of the European call option at t=T.
        '''
        return max((self.stock_prices[-1] - self.K), 0)
    

def run_hedge_simulations(n_simulations, S_0, K, r, stock_volatility, delta_volatility, T, N, n_hedges):
    '''
    Description
    -----------
    Runs multiple simulations of the Black-Scholes model to calculate the profit/loss of the replicating portfolio.
    
    Parameters
    ----------
    `n_simulations` : int
        Number of simulations to be run.
    `S_0` : float
        Stock price today.
    `K` : float
        Strike price.
    `r` : float
        Risk-free interest rate.
    `stock_volatility` : float
        Volatility.
    `delta_volatility` : float
        Volatility of the delta hedge parameter.
    `T` : float
        Time to expiration.
    `N` : int
        Number of time steps.
    `n_hedges` : int
        Number of hedges to be performed, e.g. `5` for quarterly, `53` for weekly, `366` for daily.
    
    Returns
    -------
    `list`
        List of profit/loss values for each simulation.
    '''
    profits_losses = []

    for _ in range(n_simulations):
        
        bs_sim = BlackScholesSimulation(
            S_0=S_0, 
            K=K, 
            r=r,
            M=N,
            stock_volatility=stock_volatility, 
            delta_volatility=delta_volatility,
            T=T
        )
        profit_loss = bs_sim.compute_profit_loss(n_hedges)
        profits_losses.append(profit_loss)

    return profits_losses


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ### Example Usage ###
    profits_losses = run_hedge_simulations(
        n_simulations=5000,
        S_0=100,
        K=99,
        r=0.06,
        stock_volatility=0.2,
        delta_volatility=0.2,
        T=1,
        N=1000,
        n_hedges=53  # weekly hedging
    )

    print(f"Mean profit/loss at t=T is {np.mean(profits_losses):.2f}")
    plt.hist(profits_losses, bins=30, density=True)
    plt.xlabel('Profit/Loss')
    plt.ylabel('Density')
    plt.title('${\\sigma}_{S} < {\\sigma}_{\\triangle}$')
    plt.show()
