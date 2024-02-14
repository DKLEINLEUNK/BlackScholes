import numpy as np

D8 =  0
D7 =  1
D6 =  2
D5 =  3
D4 =  4
D3 =  5
D2 =  6
D1 =  7
UD =  8
U1 =  9
U2 =  10
U3 =  11
U4 =  12
U5 =  13
U6 =  14
U7 =  15
U8 =  16

a = np.array([D8, D7, D6, D5, D4, D3, D2, D1, UD, U1, U2, U3, U4, U5, U6, U7, U8])
A = np.zeros((9, 9))

for i in np.arange(0, a.shape[0], 2):
    i_div = int(i/2)
    i_end = i+9-i_div    
    A[i_div:, i_div] = a[i:i_end][::-1]

print(A)