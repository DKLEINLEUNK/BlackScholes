#This file defines the methods required to calculate the price of an option using a binomial tree

import numpy as np


def buildTree(S, vol , T, N): 

    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol*np.sqrt(dt)) #According to formula derived in appendix
    d = np.exp(-vol*np.sqrt(dt)) 

    # Iterate over the lower triangle

    for i in np.arange(N + 1): # iterate over rows

        for j in np.arange(i + 1): # iterate over columns
            
            matrix[i,j] = S*np.power(u,j)*np.power(d,i)

            
            
            print("")

            # Hint: express each cell as a combination of up
            # and down moves matrix[i, j]=0#TODO
    
    return matrix



# Iterate over the lower triangle



