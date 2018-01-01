import itertools
import numpy as np
import pandas as pd

def input_param(DeckParameters, TrainingData, param_segment, number_training_examples):
    #function for returning vectors of deck parameters & winners vector given .csv of parameters & match history data
    #loads parameters
    df = pd.read_csv(str(DeckParameters)+'.csv', header=None)

    #strips first row corresponding to commander names since not directly used
    df = df.drop(df.index[0])
    values = df.values

    #load raw training data
    training = pd.read_csv(str(TrainingData)+ '.csv')
    training = training.drop('Date',1)

    #convert training data to matrix & unroll
    train_mat = training.as_matrix()
    train_vec_raw = train_mat.ravel()

    train_vec = []

    #separates train_vec_raw into list of input data (train_vec) and winners (classif)
    for i in range(train_vec_raw.size):
        #%4 == 3 due to intiial formatting of match data - every 4th point corresponds to winner
        if i%4 != 3:
            train_vec.append(train_vec_raw[i])
    
    #save train_vec prior to deletions in newlist decklists, used to call back to deck names
    decklists = train_vec_raw

    #print ('The number of elements in the raw data set is ' + str(len(train_vec_raw)) + '. The number of elements in the training vecto   r are ' + str(len(train_vec)) + '. The number of elements in the winners vector is ' + str(len(winners)) + '.\n')


    #sets each element of train_vec equal to the corresponding stats from values - done this way to parameterize deck names 
    for i in range(len(train_vec)):
        #substitutes commander name with corresponding parameters
        train_vec[i] = (values[values[:, 0] == str(train_vec[i])]).tolist()
        #removes first element (commander name) from list
        del train_vec[i][0][0]

    #converts nparray to single list containing floats of all parameters of all decks - ugly, but functional
    vectorized_parameters = [float(n) for n in list(itertools.chain.from_iterable(
        list(itertools.chain.from_iterable(train_vec))))]


    #initialize list of classes (winners) and reshapes to a more usable format (1s corresponding to the winner & 0s otherwise) so that can have a vectorized solution vector 
    winners = np.asarray(train_vec_raw)
    winners = winners.reshape((winners.size//4, 4))

    #substitutes winners matrix w/ 1s for position of winning deck and 0s for others
    for match in range(len(winners)):
        for deck in range(len(winners[0])):
            if winners[match][deck] == winners[match][len(winners[match]) - 1]:
                winners[match][deck] = 1
                winners[match][len(winners[match]) - 1] = 2
            else:
                winners[match][deck] = 0

    winners = np.delete(winners, [3], axis=1)

    #selects segment of unrolled parameter list to feed into NN for training
    deck_param_size = 7 #number of features for each deck
    
    if param_segment == 0:
        deck_param = vectorized_parameters[0:deck_param_size]
    elif param_segment > 0:
        deck_param = vectorized_parameters[i*deck_param_size : (i+1)*deck_param_size]

    return deck_param, winners, train_mat, values, train_vec

