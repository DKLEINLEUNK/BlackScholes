import numpy as np


class BinomialTreeValuation:
    
    def __init__(self, S, vol , T, N):
        '''
        Description
        -----------
        Initialises the parameters for the Binomial Tree Valuation.

        Parameters
        ----------
        `S` : float
            Stock price today.
        `vol` : float
            Constant for the volatility of the stock price dynamics.
        `T` : float
            Time to expiration.
        `N` : int
            Number of time steps.
        '''
        self.S = S
        self.vol = vol
        self.T = T
        self.N = N

        self.initialize_tree()
    

    def initialize_tree(self): 
        '''
        Description
        -----------
        Initializes the binomial tree.
        '''
        self.dt = self.T / self.N
        self.u = np.exp( self.vol * np.sqrt(self.dt))
        self.d = np.exp(-self.vol * np.sqrt(self.dt)) 
        
        num_up_moves = np.concatenate([np.zeros(self.N), np.arange(self.N + 1)])
        num_down_moves = num_up_moves[::-1]
        
        result = self.S * np.power(self.u, num_up_moves) * np.power(self.d, num_down_moves)
        
        A = np.zeros((self.N+1, self.N+1))
        for i in np.arange(0, result.shape[0], 2):
            i_div = int(i/2)
            i_end = i + (self.N + 1) - i_div    
            A[i_div:, i_div] = result[i:i_end][::-1]
        
        self.tree = A


    def value_option(self, r, K, call_put, option_type, as_tree=False):
        '''
        Description
        -----------
        Calculates the value of a European or American option using the Binomial Tree method.

        Parameters
        ----------
        `r` : float
            Risk-free interest rate.
        `K` : float
            Strike price.
        `call_put` : str
            'Call' or 'Put' option.
        `option_type` : str
            'EU' or 'US' option.
        `as_tree` : bool
            If True, returns the binomial tree and corresponding hedge parameters u & d.
        '''
        self.r = r
        self.K = K
        self.call_put = call_put
        self.option_type = option_type
        self.return_tree = as_tree

        if self.option_type == 'EU':
            return self.value_option_EU()

        elif self.option_type == 'US':
            return self.value_option_US()


    def value_option_EU(self):
        p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        columns = self.tree.shape[1] 
        rows = self.tree.shape[0]
        
        payoff = np.zeros_like(self.tree)

        for c in np.arange(columns): #Loop over columns
            
            S = self.tree[rows - 1, c] #Getting the stock prices in the last row
            payoff[rows - 1, c] = self.compute_payoff(S)
        
        for i in np.arange(rows - 1)[::-1]:  # loops over the row in reverse order
            for j in np.arange(i+1):  # loops over columns 
                down = payoff[i+1, j]
                up = payoff[i+1, j+1]
                payoff[i,j] = np.exp(-self.r * self.dt)*(p * up + (1-p) * down)
        
        if self.return_tree:
            return payoff, self.u, self.d
        
        return payoff[0][0]


    def value_option_US(self):
        p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        columns = self.tree.shape[1]
        rows = self.tree.shape[0]

        payoff = np.zeros_like(self.tree)

        for c in np.arange(columns):  # loops over columns
            S = self.tree[rows - 1, c]
            payoff[rows - 1, c] = self.compute_payoff(S)

        for i in np.arange(rows - 1)[::-1]:  # loops over the rows in reverse order
            for j in np.arange(i + 1):  # loops over columns 
                down = payoff[i+1, j]
                up = payoff[i+1,j+1]
                S = self.tree[i,j]

                continuation = np.exp(-self.r * self.dt) * (p * up + (1-p) * down)
                intrinsic = self.compute_payoff(S)
                payoff[i,j] = max(continuation, intrinsic)

        if self.return_tree:
            return payoff

        return payoff[0][0]


    def compute_payoff(self, S):
        '''
        Description
        -----------
        Computes the payoff of an option depending on if it is a call or a put.
        '''
        if(self.call_put == 'Call'):
            return max(S - self.K, 0)
        
        if(self.call_put == 'Put'):
            return max(self.K - S, 0)
        

def hedge_parameter_binomial(fu, fd, S_0, u, d):
    delta = (fu-fd) / (S_0*u - S_0*d)
    return delta


if __name__ == '__main__':

    ### Example Usage ###
    binom_tree = BinomialTreeValuation(
        S = 100, 
        vol = 0.2, 
        T = 1,
        N = 100
    )

    put_EU = binom_tree.value_option(
        r = 0.06,
        K = 99, 
        call_put = 'Put',
        option_type = 'EU'
    )

    put_US = binom_tree.value_option(
        r = 0.06,
        K = 99, 
        call_put = 'Put',
        option_type = 'US'
    )
    
    print(put_EU)
    print(put_US)
