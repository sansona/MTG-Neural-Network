import numpy as np
from param_vec import input_param
from weights import initialize
from activations import sigmoid, tanh

layer0_nodes = 10
layer1_nodes = 10
output_nodes = 3
#random test parameters, need to vectorize
x1, x2, x3, x4, win, decklist = input_param('DeckParameters', 'TrainingData3')

#NOTE: this only works for match x1 as of now - have to scale up to iterate through all training data
#initializes weight matrices - reshape to ensure compliance
w0, w1, w2 = initialize(x1, layer1_nodes, output_nodes)
w0 = w0.reshape(layer0_nodes, int(len(x1)))
w1 = w1.reshape(layer1_nodes, layer1_nodes) 
w2 = w2.reshape(output_nodes, layer1_nodes)

#converts x1 input to matrix 
l0 = np.array(x1, dtype=np.float128).reshape(len(x1), 1)

#transpose & activation functions corresponding to layer 1
a1 = tanh(np.dot(w0, l0))

#activation function for layer 2
a2 = tanh(np.dot(w1, a1))

#activation function for output layer
a3 = sigmoid(np.dot(w2, a2))

#index of where in win a 1 shows up for the corresponding match
win_index = list(win[0]).index(1)

print ('The winner is predicted to be: ' + str(decklist[0][list(a3).index(max(a3))]))
print ('The actual winner of this match was: ' + str(decklist[0][win_index]))

#calculate error corresponding to output layer
win_0T = np.transpose([win[0]])
l3_error = win_0T - a3


