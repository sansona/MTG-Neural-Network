import numpy as np
from param_vec import input_param
from weights import initialize, update_weights
from activations import sigmoid, tanh

layer0_nodes = 10
layer1_nodes = 10
output_nodes = 3
#random test parameters, need to vectorize
x1, x2, x3, x4, win, decklist = input_param('DeckParameters', 'TrainingData3')

#NOTE: this only works for match x1 as of now - have to scale up to iterate through all training data
'''
#initializes weight matrices - reshape to ensure compliance
w0, w1, w2 = initialize(x1, layer1_nodes, output_nodes)
w0 = w0.reshape(layer0_nodes, int(len(x1)))
w1 = w1.reshape(layer1_nodes, layer1_nodes) 
w2 = w2.reshape(output_nodes, layer1_nodes)

'''
w0 = np.loadtxt('w0.txt')
w1 = np.loadtxt('w1.txt')
w2 = np.loadtxt('w2.txt')


for i in range(6000):
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

    #print ('The winner is predicted to be: ' + str(decklist[0][list(a3).index(max(a3))]))
    #print ('The actual winner of this match was: ' + str(decklist[0][win_index]))

    #begin backpropagation 
    #calculate error corresponding to output layer
    win_0T = np.transpose([win[0]]) #win_0T is effectively y for this training example - need to scale up
    l3_error = win_0T - a3
    l3D = l3_error*sigmoid(a3, deriv=True)

    #calculate error corresponding to l2 (second hidden layer)
    l2_error = np.dot(w2.T, l3D)
    l2D = l2_error*tanh(a2, deriv=True)

    l1_error = np.dot(w1.T, l2D)
    l1D = l1_error*tanh(a1, deriv=True)
   
    ''' 
    #if want to view error decreasing through iterations
    if (i % 1000) == 0:
        print ("Error: " + str(np.mean(np.abs(l3_error))))
    '''

    #print (np.dot(l1D, win_0T.T))
    #updating weights
    w0, w1, w2 = update_weights(w0, w1, w2, win_0T, a1, a2, l1D, l2D, l3D)


print ('The winner is predicted to be: ' + str(decklist[0][list(a3).index(max(a3))]) + ' with a confidence of ' +
        str(max(list(a3))))
print ('The actual winner of this match was: ' + str(decklist[0][win_index]))

w0_save = np.savetxt('w0.txt', w0)
w1_save = np.savetxt('w1.txt', w1)
w2_save = np.savetxt('w2.txt', w2)



