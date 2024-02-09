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




# Iterate over the lower triangle



