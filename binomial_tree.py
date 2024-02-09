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


def valueOptionMatrix(tree , T, r , K, vol):

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
    

    return payoff


# Iterate over the lower triangle



