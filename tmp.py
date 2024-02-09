import matplotlib.pyplot as plt

from binomial_tree import *
from black_scholes import *

# parameters
T = 1        # maturity (years)
K = 99       # strike price at t = T
r = 0.06     # interest rate
S_0 = 100    # stock price at t = 0
sigma = 0.2  # volatility

Ns = np.arange(1, 501)  # is volatility necessarily positive and bounded to 100%?

binom_ests = []
bs_ests = []

for n in Ns:
    binom_tree = buildTree(S_0, sigma, T, n)
    binom_ests.append(valueOptionBinomial(binom_tree , T, r , K, sigma, return_tree=False))
    bs_ests.append(valueOptionBlackScholes(S_0, K, r, sigma, T))

differences = np.abs(np.array(binom_ests) - np.array(bs_ests))

print("Binomial tree estimates: ", binom_ests)
print("Black-Scholes estimates: ", bs_ests)
print("Differences: ", differences)