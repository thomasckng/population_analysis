# # Selection function with linearly increasing probability from 0 to 1 as x increases from 0 to 100 and probability 1 for x>100
# def selection_function(x):
#     p = x/100
#     p[x>100]=1.
#     p[x<0]=0.
#     return p

# # Selection function with probability 1 for x>=50 and 0.1 for x<50
# import numpy as np
# 
# def selection_function(x):
#     p = np.ones(len(x))
#     p[x.flatten()<50]=0.1
#     return p

 # # Selection function with two sigmoid functions on mz and dL
import numpy as np
def selection_function(mz, dL):
    return 1/(1+np.exp(-(mz-50)/10))*(1/(1+np.exp((dL-1500)/100)))
