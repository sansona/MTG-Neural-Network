import numpy as np
from param_vec import input_param
from weights import initialize, update_weights
from activations import sigmoid, tanh

#change this when re-training with different dataset
number_training_examples = 9
TrainingData = 'TrainingTest'

layer0_nodes = 10
layer1_nodes = 10
output_nodes = 3
#random test parameters, need to vectorize - input .csv file names w/o .csv extension
#x1, x2, x3, x4, win, decklist = input_param('DeckParameters', 'TrainingData3')



#initializes weight matrices - reshape to ensure compliance
x, win, decklist = input_param('DeckParameters', TrainingData, 0, number_training_examples)

w0, w1, w2 = initialize(x, layer1_nodes, output_nodes)
w0 = w0.reshape(layer0_nodes, int(len(x)))
w1 = w1.reshape(layer1_nodes, layer1_nodes) 
w2 = w2.reshape(output_nodes, layer1_nodes)

'''
#loads previous weight matrices - use if not initializing weights
w0 = np.loadtxt('w0.txt')
w1 = np.loadtxt('w1.txt')
w2 = np.loadtxt('w2.txt')
'''

#functional albeit unelegant iterative. First loop iterates through number of matches, second loop is a training loop (updating weights every loop)
deck = -1

for train_match in range(number_training_examples):
    deck += 1
    for i in range(5):
        #converts x1 input to matrix 

        x, win, decklist = input_param('DeckParameters', TrainingData, deck, number_training_examples)

        l0 = np.array(x, dtype=np.float128).reshape(len(x), 1)

        a1 = tanh(np.dot(w0, l0))

        a2 = tanh(np.dot(w1, a1))

        a3 = sigmoid(np.dot(w2, a2))

        #index of where in win a 1 shows up for the corresponding match
        win_index = list(win[deck]).index(1)

        #print ('The winner is predicted to be: ' + str(decklist[0][list(a3).index(max(a3))]))
        #print ('The actual winner of this match was: ' + str(decklist[0][win_index]))

        #begin backpropagation 
        #calculate error corresponding to output layer
        win_0T = np.transpose([win[deck]]) #win_0T is effectively y for this training example - need to scale up
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


    print ('The winner is predicted to be: ' + str(decklist[deck][list(a3).index(max(a3))]) + ' with a confidence of ' +
            str(max(list(a3))))
    print ('The actual winner of this match was: ' + str(decklist[deck][win_index]) + '\n')

#saves updated weight matrices to be loaded on future training
w0_save = np.savetxt('w0.txt', w0)
w1_save = np.savetxt('w1.txt', w1)
w2_save = np.savetxt('w2.txt', w2)



