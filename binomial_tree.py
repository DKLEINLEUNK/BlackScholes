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


def valueOptionBinomial(tree, T, r , K, vol, option_type, return_tree=False):

    N = tree.shape[1] - 1  # finds N from the number of columns - 1

    dt = T / N
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u-d)
    columns = tree.shape[1] 
    rows = tree.shape[0]
    
    payoff = np.zeros_like(tree)

    for c in np.arange(columns): #Loop over columns
        
        S = tree[rows - 1, c] #Getting the stock prices in the last row
        payoff[rows - 1, c] = compute_payoff(option_type,S,K)
        
        #TODO #Calculating the payoff of the option using S
    
    # For all other rows , we need to combine from previous rows 
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows - 1)[::-1]: #Loop over the row in reverse order
        for j in np.arange(i+1): # Loop over columns 
            down = payoff[i+1, j] #Getting the up and down values at the nodes in the row in the next period
            up = payoff[i+1,j+1]
            payoff[i,j] = np.exp(-r*dt)*(p*up + (1-p)*down)
    
    if return_tree:
        return payoff
    
    return payoff[0][0]


def trying_to_get_the_right_values_for_US(tree, T, r , K, vol, return_tree=False):

    N = tree.shape[1] - 1  # finds N from the number of columns - 1

    dt = T / N
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u-d)
    columns = tree.shape[1] 
    rows = tree.shape[0]
    
    payoff = np.zeros_like(tree)

    for c in np.arange(columns):  # loops over columns
        S = tree[rows - 1, c]
        payoff[rows - 1, c] = max(0, K-S)
        
    for i in np.arange(rows - 1)[::-1]:  # loops over the rows in reverse order
        for j in np.arange(i+1):  # loops over columns 
            down = payoff[i+1, j]
            up = payoff[i+1,j+1]
            continuation = np.exp(-r*dt)*(p*up + (1-p)*down)
            intrinsic = max(0, K-tree[i,j])
            payoff[i,j] = max(continuation, intrinsic)
    
    if return_tree:
        return payoff
    
    return payoff[0][0]


def compute_payoff(option_type,S,K):
    '''Computes the payoff of an option depending on if it is a Call or a Put'''
    if(option_type == "Call"):

        return max(S-K,0)
    
    if(option_type == "Put"):

        return max(K-S,0)



def valueUSOptionBinomial(tree , T, r , K, vol, option_type, return_tree=False):

    '''Values an American option'''

    N = tree.shape[1] - 1 #Getting N from the number of columns - 1

    dt = T / N
    u = np.exp(vol*np.sqrt(dt)) #According to formula derived in appendix
    d = np.exp(-vol*np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u-d) #According to formula derived in appendix
    columns = tree.shape[1] 
    rows = tree.shape[0]
    
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row for c in np.arange(columns):

    payoff = np.zeros_like(tree)


    #Calculate the intrinsic value of the option at every node
    for i in np.arange(rows - 1)[::-1]: #Loop over the row in reverse order
        for c in np.arange(columns): #Loop over columns
            S = tree[i, c] #Getting the stock prices at every node in row i column c
            Intrinsic_value =  compute_payoff(option_type,S,K)
            payoff[i, c] = Intrinsic_value #Append Intrinsic value to payoff array
        
        #TODO #Calculating the payoff of the option using S
    
    # For all other rows , we need to combine from previous rows 
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows - 1)[::-1]: #Loop over the row in reverse order
        for j in np.arange(i+1): # Loop over columns 
            down = payoff[i+1, j] #Getting the up and down values at the nodes in the row in the next period
            up = payoff[i+1,j+1]
            
            discounted_payoff = np.exp(-r*dt)*(p*up + (1-p)*down)
            Intrinsic_value = payoff[i,j] #Get the current intrinsic value at the node in row i and column j
            payoff[i,j] = max(Intrinsic_value,discounted_payoff) #Set the payoff at that node equal to the maximum of the discounted up and down values form the next period and the current intrinsic value
    
    if return_tree:
        return payoff
    
    return payoff[0][0]

def compute_hedge_parameter_binomial(fu, fd,S_0,u,d):

    delta = (fu-fd)/(S_0*u - S_0*d)

    return delta


if __name__ == '__main__':

    ### Example Usage ###
    T = 1        # maturity (years)
    K = 98       # strike price at t = T
    r = 0.06     # interest rate
    S_0 = 100    # stock price at t = 0
    sigma = 0.2  # volatility
    N = 5        # timesteps

    tree = buildTree(S_0, sigma, T, N)
    payoff = valueOptionBinomial(
        tree, 
        T, 
        r, 
        K, 
        sigma, 
        return_tree=False
    )

    treeUS = buildTree(S_0, sigma, T, N)
    payoffUS = trying_to_get_the_right_values_for_US(
        treeUS, 
        T, 
        r, 
        K, 
        sigma,
        return_tree=False
    )
    
    print(payoff)
    print(payoffUS)
    # approximate_value = payoff[0][0]
    # print(f'Value of the option: {payoff:.2f}')
