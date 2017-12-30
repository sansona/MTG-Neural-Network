import numpy as np
from param_vec import input_param
from weights import initialize
from activations import sigmoid, tanh

#random test parameters, need to vectorize
x1, x2, x3, x4, win = input_param('DeckParameters', 'TrainingData3')

#first layer of NN, receiving input vector
w0 = (initialize(x1)).reshape(10, int(len(x1)))
#print (w0)

#converts x1 input to matrix 
l0 = np.array(x1, dtype=np.float128).reshape(len(x1), 1)
#transpose & activation functions corresponding to layer 1
l1 = tanh(np.dot(w0, l0))
