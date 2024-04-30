# A test selection function
def selection_function(x):
    p = x/100
    p[x>100]=1.
    p[x<0]=0.
    return p
