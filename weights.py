import numpy as np

#nn parameters - chosen semi-randomly
w0_rowsize = 10
w1_rowsize = 10
alpha = 0.5

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

def update_weights(weight0, weight1, weight2,
        loss0, loss1, loss2, loss1D, loss2D, loss3D):
    #updates weights with loss
    #need != 0 term to extract nonzero terms from ill-matched matrix
    weight0 += alpha*(np.dot(loss1D, loss0.T))[np.dot(loss1D, loss0.T) != 0].reshape(w0_rowsize, 1).astype(float)
    weight1 += alpha*(np.dot(loss2D, loss1.T)).astype(float)
    weight2 += alpha*(np.dot(loss3D, loss2.T)).astype(float)

    return weight0, weight1, weight2
