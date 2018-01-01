import itertools
import numpy as np
from param_vec import input_param
from weights import initialize, update_weights
from activations import sigmoid, tanh

#change these when re-training with different datasets
number_training_examples = 9
TrainingData = 'TrainingTest'

#can be scaled up/down if desired
layer0_nodes = 10
layer1_nodes = 10
output_nodes = 3
number_features = 7

def neural_network(test = False):
    #trains network if test == False, runs nn on test data if test == True
    if (test == True):
        #weight matrices must be initialized first before loaded
        w0_test = np.loadtxt('w0.txt')
        w1_test = np.loadtxt('w1.txt')
        w2_test = np.loadtxt('w2.txt')

        #care about values & train_vec to vectorize user inputs
        x_train, win_train, train_mat, values, train_vec = input_param('DeckParameters', TrainingData, 0, number_training_examples)

        train_deck1 = str(input('Enter the name of the first deck: \n'))
        train_deck2 = str(input('\nEnter the name of the second deck: \n'))
        train_deck3 = str(input('\nEnter the name of the third deck: \n'))
        
        test_decks = [train_deck1, train_deck2, train_deck3]
        #substitutes commander name with corresponding parameters - to parameterize deck data
        for i in range(len(test_decks)):
            test_decks[i] = (values[values[:, 0] == str(test_decks[i])]).tolist()
            del test_decks[i][0][0]


        x_test = [float(n) for n in list(itertools.chain.from_iterable(
            list(itertools.chain.from_iterable(test_decks))))]
        #divides x_test into features for each corresponding deck and feeds into training loop
        x_test_deck1 = x_test[0:number_features]
        x_test_deck2 = x_test[number_features:2*number_features]
        x_test_deck3 = x_test[2*number_features:3*number_features]

        #sends data through nn
        maxValue = 0
        for deck_test in [x_test_deck1, x_test_deck2, x_test_deck3]:
            l0_test = np.array(deck_test, dtype=np.float128).reshape(len(deck_test), 1)
            a1_test = tanh(np.dot(w0_test, l0_test))
            a2_test = tanh(np.dot(w1_test, a1_test))
            a3_test = sigmoid(np.dot(w2_test, a2_test))
            deck_test.append(max(a3_test)) #to compare maximum a3_test values
            
            #saves a3_test value and name of predicted deck for display later
            if deck_test[-1] > maxValue:
                maxValue = deck_test[-1]
                best_deck_index = [x_test_deck1, x_test_deck2, x_test_deck3].index(deck_test)

        #to be completely honest, not sure why the code only works when this is initialized here
        test_decks_name = [train_deck1, train_deck2, train_deck3]

        print ('The winner is predicted to be: ' + str(test_decks_name[best_deck_index]) + ' with a confidence of ' +
                str(maxValue))


    elif (test == False):
        #for training nn - run when initialized with test=False 
        #initializes weight matrices - reshape to ensure compliance
        x, win, decklist, values, train_vec = input_param('DeckParameters', TrainingData, 0, number_training_examples)
        '''
        w0, w1, w2 = initialize(x, layer1_nodes, output_nodes)
        w0 = w0.reshape(layer0_nodes, int(len(x)))
        w1 = w1.reshape(layer1_nodes, layer1_nodes) 
        w2 = w2.reshape(output_nodes, layer1_nodes)
        ''' 
        #loads previous weight matrices - use if not initializing weights
        w0 = np.loadtxt('w0.txt')
        w1 = np.loadtxt('w1.txt')
        w2 = np.loadtxt('w2.txt')

        #functional albeit unelegant iterative. First loop iterates through number of matches, second loop is a training loop (updating weights every loop)
        deck = -1
        for train_match in range(number_training_examples):
            deck += 1
            for i in range(100):
                #converts x1 input to matrix 

                x, win, decklist, values, train_vec = input_param('DeckParameters', TrainingData, deck, number_training_examples)

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
                win_0T = np.transpose([win[deck]]) 
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



