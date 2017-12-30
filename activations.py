import numpy as np

#sigmoid function for use as ouput layer activation
def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def tanh(x, deriv=False):
    if (deriv==True):
        return 4/((np.exp(-x) + np.exp(x))**2)

    return (1 - np.exp(-2*x))/(1 + np.exp(-2*x))
