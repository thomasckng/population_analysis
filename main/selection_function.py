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

# Selection function by reading from a pkl file
import dill

with open('selfunc_detframe.pkl', 'rb') as f:
    selfunc_interp = dill.load(f)

def selection_function(x):
    return selfunc_interp(x)
