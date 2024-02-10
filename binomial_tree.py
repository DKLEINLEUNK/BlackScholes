#This file defines the methods required to calculate the price of an option using a binomial tree

import numpy as np


def buildTree(S, vol , T, N): 

    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol*np.sqrt(dt)) #According to formula derived in appendix
    d = np.exp(-vol*np.sqrt(dt)) 

    # Iterate over the lower triangle

    for i_row in np.arange(N + 1): # iterate over rows


        for j_col in np.arange(i_row + 1): # iterate over columns

            num_up_moves = j_col
            num_down_moves = i_row - j_col
            
            #TODO: Recombine down and up movements

            if(i_row - j_col < 0):
                
                matrix[i_row,j_col] = 0
            else:
                 matrix[i_row,j_col] = S*np.power(u, num_up_moves)*np.power(d, num_down_moves)
         
            
            # Hint: express each cell as a combination of up
            # and down moves matrix[i, j]=0#TODO
    
    return matrix


def buildTreeFaster(S, vol , T, N): 
    dt = T / N
    u = np.exp(vol*np.sqrt(dt)) #According to formula derived in appendix
    d = np.exp(-vol*np.sqrt(dt)) 
    
    num_up_moves = np.concatenate([np.zeros(N), np.arange(N + 1)])
    num_down_moves = num_up_moves[::-1]
    
    result = S*np.power(u, num_up_moves)*np.power(d, num_down_moves)
    
    # This code isn't readable, but it's faster than the previous implementation by a lot
    # Take my word for it, I'm a scientist
    A = np.zeros((N+1,N+1))
    for i in np.arange(0, result.shape[0], 2):
        i_div = int(i/2)
        i_end = i+(N+1)-i_div    
        A[i_div:, i_div] = result[i:i_end][::-1]
    
    return A


def valueOptionBinomial(tree , T, r , K, vol, return_tree=False):

    N = tree.shape[1] - 1 #Getting N from the number of columns - 1

    dt = T / N
    u = np.exp(vol*np.sqrt(dt)) #According to formula derived in appendix
    d = np.exp(-vol*np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u-d) #According to formula derived in appendix
    columns = tree.shape[1] 
    rows = tree.shape[0]
    
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row for c in np.arange(columns):

    payoff = np.zeros_like(tree)  # ASK TA IF THIS IS OKAY

    for c in np.arange(columns): #Loop over columns

        S = tree[rows - 1, c] #Getting the first stock price in the last row
        payoff[rows - 1, c] = max(0, S-K)
        
        #TODO #Calculating the payoff of the option using S
    
    # For all other rows , we need to combine from previous rows 
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows - 1)[::-1]: #Loop over the row in reverse order
        for j in np.arange(i+1): # Loop over columns 
            down = payoff[i+1, j]
            up = payoff[i+1,j+1]

            payoff[i,j] = np.exp(-r*dt)*(p*up + (1-p)*down)
    
    if return_tree:
        return payoff
    
    return payoff[0][0]


if __name__ == '__main__':

    ### Example Usage ###
    T = 1        # maturity (years)
    K = 99       # strike price at t = T
    r = 0.06     # interest rate
    S_0 = 100    # stock price at t = 0
    sigma = 0.2  # volatility
    N = 5_000    # timesteps

    import time
    start = time.time()
    tree = buildTree(S_0, sigma, T, N)
    end = time.time()

    print('')
    print('Runtimes')
    print('--------')
    print(f'buildTree(): {end-start:.3f} seconds')
    start = time.time()    
    tree2 = buildTreeFaster(S_0, sigma, T, N)
    end = time.time()
    print(f'buildTreeFaster():  {end-start:.3f}  seconds')
    print('Trees identical?: ', np.allclose(tree, tree2))
    print('N = ', N)
    print('')
    # payoff = valueOptionBinomial(tree, T, r, K, sigma, return_tree=False)

    # approximate_value = payoff[0][0]
    # print(f'Value of the option: {payoff:.2f}')
