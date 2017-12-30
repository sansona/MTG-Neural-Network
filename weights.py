import numpy as np

w0_rowsize = 10
w1_rowsize = 10

def initialize(input0, w1_rowsize, output_rowsize):
    #initializes weight matrixes - only used first run
    mat0 = 2*np.random.random((w0_rowsize, int(len(input0)))) - 1
    mat1 = 2*np.random.random((w1_rowsize, w1_rowsize)) - 1
    mat2 = 2*np.random.random((output_rowsize, w1_rowsize)) - 1

    return mat0, mat1, mat2

'''
def hidden_layers(w1_rowsize):
    #initialize weight for hidden network
    mat1 = 2*np.random.random((w1_rowsize, w1_rowsize)) - 1

    return mat1
'''

def update_weights(weight0, weight1, loss0, loss1, loss2D, loss1D):
    #updates weights with loss
    weight1 += loss1.T.dot(loss2D)
    weight0 += loss0.T.dot(loss1D)
                                    
    return weight0, weight1
