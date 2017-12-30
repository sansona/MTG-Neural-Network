import numpy as np

def initialize(input0):
    #initializes weight matrixes - only used first run
    mat0 = 2*np.random.random((10, int(len(input0)),)) - 1
    #mat1 = 2*np.random.random((5,1)) - 1

    return mat0

def update_weights(weight0, weight1, loss0, loss1, loss2D, loss1D):
    #updates weights with loss
    weight1 += loss1.T.dot(loss2D)
    weight0 += loss0.T.dot(loss1D)
                                    
    return weight0, weight1
